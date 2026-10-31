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

"""KimiSubstrate: Kimi Code CLI (`kimi -p`) as a solver substrate.

Carries the c1..c8 surface (KIMI_SURFACE), the one install mode (kimi_root), the
per-rollout KIMI_CODE_HOME that carries the harness hooks (Kimi has no project-level
hook file), and the role-based stream-json trajectory renderer.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

from .. import config
from ..core import runmark as RM
from ..core import tree as HT
from ..core.tree import SurfaceSpec
from .base import Install, Launcher, RunRequest
from .codex import _TRUNC_FMT, _clip, _emit_if, _text_of

KIMI_HOOKS_FILE = ".kimi/hooks.json"

KIMI_SURFACE = SurfaceSpec(
    name="kimi",
    always_on=("AGENTS.md",),
    always_on_max=8_000,
    always_on_cap_reason="AGENTS.md is injected whole into the system prompt of EVERY turn",
    skill_dir=".agents/skills",
    skill_name_rule="must_match",
    trigger_hint=(
        'Fix it by adding a trigger line naming `{name}` to AGENTS.md, e.g. "When '
        "<the situation this skill handles>, read .agents/skills/{name}/SKILL.md "
        'and follow it."'
    ),
    autotrigger_file="AGENTS.md",
    trigger_line="- When {desc}, read .agents/skills/{name}/SKILL.md and follow it.\n",
    subagent_requires_mode=False,
    plugin_dir=None,
    hooks_file=KIMI_HOOKS_FILE,
    hook_events=frozenset(
        {
            "PreToolUse",
            "PostToolUse",
            "PostToolUseFailure",
            "PreCompact",
            "PostCompact",
            "SessionStart",
            "SubagentStart",
            "SubagentStop",
            "UserPromptSubmit",
            "Stop",
            "StopFailure",
        }
    ),
    compile_skill_scripts=True,
    reject_empty_always_on=True,
    stubs=(
        (
            "scripts/README.md",
            "<!-- c7: deterministic helpers, run as `python3 scripts/<name>.py`. "
            "Add one by emitting scripts/<name>.py. -->\n",
        ),
        (
            "sub_agents/README.md",
            "<!-- c5: delegated kimi agents, each spawned in its OWN context. Add one as "
            "sub_agents/<name>/AGENT.md with frontmatter `description:` (required); the body "
            "is its system prompt. The runner installs it as .agents/agents/<name>.md. -->\n",
        ),
        (
            ".kimi/README.md",
            "<!-- c8: kimi lifecycle hooks -- the only lever that acts INSIDE the tool loop, "
            "so the model cannot skip it. Add ONE file, .kimi/hooks.json, Claude-Code hook "
            'schema: {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": '
            '"command", "command": "..."}]}]}}. The hook reads the event as JSON on stdin; '
            "matcher is a regex on the tool name. Only PreToolUse, Stop and UserPromptSubmit "
            "can block: exit 2 with the reason on stderr (a blocked Stop makes the agent "
            "continue). On UserPromptSubmit, stdout is appended to the context. Every other "
            "event is observe-only, and a crashing hook is ignored (fail-open). -->\n",
        ),
        (
            ".agents/skills/README.md",
            "<!-- c6: reusable workflows. Add one as .agents/skills/<name>/SKILL.md with YAML "
            "frontmatter: `name: <name>` (matching the directory) and `description:` are both "
            "REQUIRED, or kimi fails to parse it. Only name+description are advertised to the "
            "agent; the body is read on demand. -->\n",
        ),
    ),
)

KIMI_ROOT_ENTRIES = ("AGENTS.md", ".agents", "scripts")

_HOOK_TIMEOUT_RANGE = (1, 600)


def _toml_str(s: str) -> str:
    """A TOML basic string; JSON's escapes (ASCII-only) are a valid subset."""
    return json.dumps(str(s), ensure_ascii=True)


def hooks_toml(src: str) -> str:
    """Translate a Claude-Code-schema hooks.json into kimi `[[hooks]]` tables.

    One table per command; a `*` or empty matcher means "match all", which kimi spells
    by omitting the key. Timeouts are clamped to kimi's 1..600s range.
    """
    doc = json.loads(src) if src.strip() else {}
    out = []
    for event, entries in (doc.get("hooks") or {}).items():
        for entry in entries or []:
            matcher = str(entry.get("matcher") or "").strip()
            for h in entry.get("hooks") or []:
                if h.get("type") != "command" or not str(h.get("command", "")).strip():
                    continue
                rows = [f"event = {_toml_str(event)}"]
                if matcher and matcher != "*":
                    rows.append(f"matcher = {_toml_str(matcher)}")
                rows.append(f"command = {_toml_str(h['command'])}")
                if isinstance(h.get("timeout"), (int, float)):
                    lo, hi = _HOOK_TIMEOUT_RANGE
                    rows.append(f"timeout = {min(hi, max(lo, int(h['timeout'])))}")
                out.append("[[hooks]]\n" + "\n".join(rows) + "\n")
    return "\n".join(out)


# Shared with the base home by symlink, so a token refresh lands where the user's own
# `kimi` reads it; everything else (sessions, logs, config) stays per rollout.
_SHARED_HOME_ENTRIES = ("credentials", "oauth", "bin", "device_id", "region")
_TABLE_HEADER = re.compile(r"^\s*\[")
_HOOKS_HEADER = re.compile(r"^\s*\[\[\s*hooks\s*\]\]\s*(#.*)?$")


def strip_hooks(config_text: str) -> str:
    """Drop every `[[hooks]]` table from a kimi config.toml; the user's own hooks (session
    archivers, notifiers) must not fire inside benchmark rollouts."""
    out, in_hooks = [], False
    for ln in config_text.splitlines(keepends=True):
        if _TABLE_HEADER.match(ln):
            in_hooks = bool(_HOOKS_HEADER.match(ln))
        if not in_hooks:
            out.append(ln)
    return "".join(out)


def make_home(base_home: Path, hooks: str, root: Path) -> Path:
    """Build `root/.kimi-code`, a KIMI_CODE_HOME carrying the base provider config with the
    harness hooks in place of the user's, and the base OAuth credentials shared by link.

    Kimi reads hooks only from the user-level config.toml, so this is where c8 lands.
    `root` doubles as HOME, which keeps the user's ~/.agents skills out of the rollout.
    """
    base_home, home = Path(base_home), Path(root) / ".kimi-code"
    home.mkdir(parents=True)
    cfg = strip_hooks((base_home / "config.toml").read_text(encoding="utf-8")).rstrip()
    (home / "config.toml").write_text(
        cfg + "\n" + ("\n" + hooks if hooks.strip() else ""), encoding="utf-8"
    )
    for entry in _SHARED_HOME_ENTRIES:
        if (base_home / entry).exists():
            (home / entry).symlink_to(base_home / entry)
    return home


class KimiSubstrate:
    """kimi-code behind the one Substrate interface."""

    name = "ki"

    def __init__(self, launcher: Launcher | None = None):
        self._launcher = launcher

    def surface(self) -> SurfaceSpec:
        return KIMI_SURFACE

    def materialise(self, tree_dir: Path, spec, ctx: Mapping[str, str]) -> Install:
        """Stage the workspace-root entries kimi discovers itself (AGENTS.md, .agents/skills,
        .agents/agents), and carry the hooks as TOML for the per-rollout KIMI_CODE_HOME."""
        if spec.install_mode != "kimi_root":
            raise ValueError(
                f"install_mode {spec.install_mode!r} is not the kimi mode; this substrate "
                f"has exactly one: 'kimi_root'"
            )
        tree_dir = Path(tree_dir)
        if not HT.is_tree(KIMI_SURFACE, tree_dir):
            raise ValueError(f"{tree_dir} is not a kimi harness tree (no AGENTS.md)")
        staged = Path(tempfile.mkdtemp(prefix="adapt-ki-root-"))
        for entry in KIMI_ROOT_ENTRIES:
            src = tree_dir / entry
            if src.is_dir():
                shutil.copytree(src, staged / entry, ignore=shutil.ignore_patterns("README.md"))
            elif src.is_file():
                shutil.copy2(src, staged / entry)
        for agent in sorted((tree_dir / "sub_agents").glob("*/AGENT.md")):
            dest = staged / ".agents" / "agents" / f"{agent.parent.name}.md"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(agent, dest)
        hooks = tree_dir / KIMI_HOOKS_FILE
        return Install(
            staging_dir=staged,
            fingerprint=RM.fingerprint(tree_dir),
            extras={
                "install_mode": "kimi_root",
                "kimi_hooks": hooks_toml(hooks.read_text(encoding="utf-8"))
                if hooks.is_file()
                else "",
                **{k: v for k, v in ctx.items() if not k.startswith("file:")},
            },
        )

    def exec_env(self, base_env: Mapping[str, str], *, kimi_home: Path) -> dict[str, str]:
        """The `kimi -p` environment: a per-rollout data root, and HOME next to it."""
        env = dict(base_env)
        env["KIMI_CODE_HOME"] = str(kimi_home)
        env["HOME"] = str(Path(kimi_home).parent)
        env["KIMI_DISABLE_TELEMETRY"] = "1"
        return env

    def execute(
        self, task_ref: str, install: Install, run_ids: list[str], timeout: int
    ) -> list[Path]:
        if self._launcher is None:
            raise RuntimeError(
                "KimiSubstrate was constructed without a launcher body; the benchmark "
                "adapter must supply one"
            )
        return self._launcher(
            RunRequest(task_ref=task_ref, run_ids=tuple(run_ids), timeout=int(timeout)), install
        )

    def read_trajectory(
        self, stream_path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000
    ) -> str:
        return render_events(
            Path(stream_path), obs_trunc=obs_trunc, max_chars=max_chars, args_trunc=args_trunc
        )

    def preflight(self) -> None:
        """Check the CLI and the base data root the per-rollout homes are built from."""
        if not Path(config.KIMI_BIN).is_file():
            raise RuntimeError(
                f"Kimi Code CLI not found at {config.KIMI_BIN}; install it or set HARNESS2_KIMI_BIN"
            )
        if not (config.KIMI_BASE_HOME / "config.toml").is_file():
            raise RuntimeError(
                f"no {config.KIMI_BASE_HOME}/config.toml; run `kimi` once and /login, or set "
                f"HARNESS2_KIMI_HOME to a configured KIMI_CODE_HOME"
            )


def parse_events(path: Path) -> list[dict]:
    """Read the stream-json capture (junk tolerated: progress lines, a stray stderr)."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return [{"role": "_read_error", "content": str(e)}]
    events = []
    for ln in raw.replace("\x00", "").splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            ev = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events


def _tool_args(fn: dict) -> str:
    """Tool arguments as compact JSON; kimi sends them as a JSON-encoded string."""
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args
    return json.dumps(args, ensure_ascii=False)


def render_events(path: Path, *, obs_trunc: int, max_chars: int, args_trunc: int = 2000) -> str:
    """Render a kimi stream-json capture into the shared grammar.

    Kimi writes one chat message per line (`role` = assistant | tool | user | meta);
    thinking and token usage are not part of the stream.
    """
    lines: list[str] = []
    for ev in parse_events(Path(path)):
        role = ev.get("role", "")
        if role == "_read_error":
            return f"(failed to read trajectory: {ev.get('content')})"
        if role == "assistant":
            _emit_if(lines, "REASONING", _text_of(ev.get("reasoning_content")), obs_trunc)
            _emit_if(lines, "ASSISTANT", _text_of(ev.get("content")), obs_trunc)
            for call in ev.get("tool_calls") or []:
                fn = call.get("function") if isinstance(call, dict) else None
                if isinstance(fn, dict):
                    name = fn.get("name") or "tool"
                    lines.append(f"[ACTION] {name}: {_clip(_tool_args(fn), args_trunc)}")
        elif role == "tool":
            failed = bool(ev.get("is_error"))
            obs = _clip(_text_of(ev.get("content")).rstrip("\n"), obs_trunc)
            lines.append(f"[OBSERVATION{' ERROR' if failed else ''}] {obs}")
        elif role == "user":
            _emit_if(lines, "USER", _text_of(ev.get("content")), obs_trunc)
        elif role == "meta":
            etype = str(ev.get("type", ""))
            if "error" in etype or ev.get("error"):
                msg = _text_of(ev.get("error")) or _text_of(ev) or json.dumps(ev)[:300]
                lines.append(f"[ERROR] {_clip(msg, 1000)}")
        elif ev.get("error"):
            msg = _text_of(ev.get("error")) or json.dumps(ev)[:300]
            lines.append(f"[ERROR] {_clip(msg, 1000)}")

    rendered = "\n".join(lines).strip()
    if not rendered:
        return "(no parseable events in trajectory)"
    if len(rendered) > max_chars:
        dropped = len(rendered) - max_chars
        rendered = rendered[:max_chars] + "\n" + _TRUNC_FMT.format(n=dropped)
    return rendered
