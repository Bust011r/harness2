# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JobBench Kimi launcher: stage a task workspace, run `kimi -p` in it, keep the artifacts.

The adapter supplies KIMI_CODE_HOME (provider config + harness hooks) and HOME in the
environment; this script owns the workspace and the attempt loop.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from harness2.benchmarks.jb_opencode_runner import (
    PROMPT,
    TRAJ_DIR_NAME,
    _hms,
    _log,
    _nonempty,
    _stamp,
)
from harness2.substrate.kimi import KIMI_ROOT_ENTRIES

TEMP_BASE = Path(tempfile.gettempdir()) / "harness2_kimi_ws"


@dataclass(frozen=True)
class Settings:
    task_ids: tuple[str, ...]
    label: str
    render_dir: Path
    tasks_base: Path
    kimi_bin: str
    model: str
    timeout_per_task: int
    max_retries: int
    log_dir: Path


def _settings() -> Settings:
    """Resolve the environment contract the adapter writes; a bad one exits non-zero."""
    env = os.environ
    need = ("KIMI_BIN", "LABEL", "HARNESS_RENDER_DIR", "TASK_IDS", "TASKS_BASE_DIR")
    missing = [n for n in need + ("KIMI_CODE_HOME",) if not env.get(n)]
    if missing:
        raise SystemExit(f"[ERROR] unset: {', '.join(missing)}")
    tasks_base = Path(env["TASKS_BASE_DIR"])
    return Settings(
        task_ids=tuple(env["TASK_IDS"].split()),
        label=env["LABEL"],
        render_dir=Path(env["HARNESS_RENDER_DIR"]),
        tasks_base=tasks_base,
        kimi_bin=env["KIMI_BIN"],
        model=env.get("KIMI_MODEL", ""),
        timeout_per_task=int(env.get("TIMEOUT_PER_TASK") or 3600),
        max_retries=int(env.get("MAX_RETRIES") or 2),
        log_dir=tasks_base / ".harness2-logs",
    )


def _stage(task_dir: Path, render_dir: Path, workspace: Path) -> None:
    """Fresh task copy, empty output/, harness files at the root, and a git root:
    kimi resolves project skills and agents from the nearest directory holding .git."""
    shutil.rmtree(workspace, ignore_errors=True)
    (workspace / "output").mkdir(parents=True)
    shutil.copytree(task_dir, workspace / "task_folder")
    for entry in KIMI_ROOT_ENTRIES:
        src = render_dir / entry
        if src.is_dir():
            shutil.copytree(src, workspace / entry)
        elif src.is_file():
            shutil.copy2(src, workspace / entry)
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)


def _command(cfg: Settings, prompt: str) -> list[str]:
    """`kimi -p`: non-interactive and auto-approved (it refuses --yolo/--auto), one JSON
    message per line on stdout; the workspace is the process cwd."""
    cmd = [cfg.kimi_bin, "-p", prompt, "--output-format", "stream-json"]
    return cmd + (["-m", cfg.model] if cfg.model else [])


def _attempt(cfg: Settings, workspace: Path, traj_file: Path, log_file: Path) -> int:
    """Run kimi once; its stream lands in `traj_file`, its progress on stderr in the log.
    A timeout kills the whole process group (kimi's background tasks included): 124."""
    prompt = PROMPT.format(
        task_folder=workspace / "task_folder",
        output_dir=workspace / "output",
        workspace=workspace,
    )
    with traj_file.open("w", encoding="utf-8") as out, log_file.open("a", encoding="utf-8") as err:
        proc = subprocess.Popen(
            _command(cfg, prompt),
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        try:
            return proc.wait(timeout=cfg.timeout_per_task)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return 124


def run_one(cfg: Settings, task_id: str) -> bool:
    """Roll one task; True only when deliverables landed in `model_output/<label>/`."""
    task_parent = cfg.tasks_base / task_id
    task_dir = task_parent / "task_folder"
    task_name = task_id.replace("/", "_")
    final_output_dir = task_parent / "model_output" / cfg.label
    traj_dir = task_parent / TRAJ_DIR_NAME / cfg.label
    log_file = cfg.log_dir / f"kimi_{task_name}_{cfg.label}_{_stamp()}.log"

    if not task_dir.is_dir():
        _log(log_file, f"[ERROR] no task_folder: {task_dir}")
        return False
    if _nonempty(final_output_dir):
        print(f"[{_hms()}] SKIP (done): {task_id} ({cfg.label})", flush=True)
        return True
    traj_dir.mkdir(parents=True, exist_ok=True)

    workspace = TEMP_BASE / f"{task_name}_{cfg.label}_{os.getpid()}"
    output_dir = workspace / "output"
    for attempt in range(1, cfg.max_retries + 1):
        _stage(task_dir, cfg.render_dir, workspace)
        traj_file = traj_dir / f"{task_name}_{cfg.label}_attempt{attempt}_{_stamp()}.jsonl"
        _log(log_file, f"[{_hms()}] run {task_id} ({cfg.label}) attempt {attempt} -> {traj_file}")
        code = _attempt(cfg, workspace, traj_file, log_file)
        if _nonempty(output_dir):
            break
        _log(log_file, f"[{_hms()}] attempt {attempt} exit {code}, no output")

    delivered = _nonempty(output_dir)
    if delivered:
        final_output_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(output_dir.iterdir()):
            if not item.name.startswith("."):
                shutil.move(str(item), str(final_output_dir / item.name))
        _log(log_file, f"[{_hms()}] output -> {final_output_dir}")
    else:
        _log(log_file, f"[{_hms()}] NO OUTPUT for {task_id} ({cfg.label})")
    shutil.rmtree(workspace, ignore_errors=True)
    return delivered and _nonempty(final_output_dir)


def main() -> int:
    cfg = _settings()
    print(f"=== kimi eval: LABEL={cfg.label} MODEL={cfg.model or '<default>'} ===", flush=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    TEMP_BASE.mkdir(parents=True, exist_ok=True)
    failed = [task_id for task_id in cfg.task_ids if not run_one(cfg, task_id)]
    print(f"=== kimi eval done: {cfg.label} ===", flush=True)
    if failed:
        print(f"no output for: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
