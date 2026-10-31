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

import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness2 import config
from harness2.benchmarks.jobbench import JobBenchAdapter
from harness2.core import tree as HT
from harness2.substrate.kimi import KIMI_SURFACE, KimiSubstrate, hooks_toml, make_home

# A kimi 2.4.1 `-p --output-format stream-json` capture, trimmed from a real run.
STREAM = [
    {"role": "meta", "type": "system.version", "version": "2.4.1"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "tool_1",
                "function": {
                    "name": "Write",
                    "arguments": json.dumps({"path": "hello.txt", "content": "hi\n"}),
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "tool_1", "content": "Wrote 3 bytes to hello.txt"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "tool_2",
                "function": {"name": "Bash", "arguments": json.dumps({"command": "cat hello.txt"})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "tool_2", "content": "hi\n"},
    {"role": "assistant", "content": "Done. Created hello.txt."},
    {
        "role": "meta",
        "type": "session.resume_hint",
        "session_id": "session_x",
        "command": "kimi -r session_x",
        "content": "To resume this session: kimi -r session_x",
    },
]


def _write_stream(d: Path, events: list, junk: str = "") -> Path:
    p = d / "stream.jsonl"
    p.write_text(junk + "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    return p


class KimiRendererTests(unittest.TestCase):
    def test_renders_tool_calls_observations_and_final_message(self):
        with tempfile.TemporaryDirectory() as d:
            out = KimiSubstrate().read_trajectory(
                _write_stream(Path(d), STREAM, junk="progress noise\n"),
                obs_trunc=1000,
                max_chars=100_000,
            )
        self.assertEqual(
            out.splitlines(),
            [
                '[ACTION] Write: {"path": "hello.txt", "content": "hi\\n"}',
                "[OBSERVATION] Wrote 3 bytes to hello.txt",
                '[ACTION] Bash: {"command": "cat hello.txt"}',
                "[OBSERVATION] hi",
                "[ASSISTANT] Done. Created hello.txt.",
            ],
        )

    def test_truncates_long_observations_and_whole_render(self):
        events = [{"role": "tool", "tool_call_id": "t", "content": "x" * 50}]
        with tempfile.TemporaryDirectory() as d:
            p = _write_stream(Path(d), events)
            clipped = KimiSubstrate().read_trajectory(p, obs_trunc=10, max_chars=100_000)
            capped = KimiSubstrate().read_trajectory(p, obs_trunc=1000, max_chars=20)
        self.assertEqual(clipped, "[OBSERVATION] xxxxxxxxxx…(+40 chars — render truncation)")
        self.assertTrue(capped.startswith("[OBSERVATION] xxxxxx\n…(+"))

    def test_empty_stream_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            out = KimiSubstrate().read_trajectory(
                _write_stream(Path(d), [], junk="not json\n"), obs_trunc=10, max_chars=100
            )
        self.assertEqual(out, "(no parseable events in trajectory)")


HOOKS = {
    "hooks": {
        "PreToolUse": [
            {
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "python3 scripts/guard.py", "timeout": 5}],
            }
        ],
        "Stop": [{"matcher": "*", "hooks": [{"type": "command", "command": 'echo "done"'}]}],
    }
}


def _tree(d: Path) -> Path:
    files = {
        "AGENTS.md": "Follow the task.\n",
        ".agents/skills/xlsx/SKILL.md": (
            "---\nname: xlsx\ndescription: read spreadsheets\n---\nUse pandas.\n"
        ),
        "scripts/guard.py": "print('ok')\n",
        "sub_agents/reviewer/AGENT.md": "---\ndescription: reviews\n---\nReview it.\n",
        ".kimi/hooks.json": json.dumps(HOOKS),
    }
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    return d


class KimiMaterialiseTests(unittest.TestCase):
    def test_stages_workspace_files_and_kimi_agents(self):
        with tempfile.TemporaryDirectory() as d:
            inst = KimiSubstrate().materialise(
                _tree(Path(d)), SimpleNamespace(install_mode="kimi_root"), {}
            )
            staged = sorted(
                str(p.relative_to(inst.staging_dir))
                for p in inst.staging_dir.rglob("*")
                if p.is_file()
            )
        self.assertEqual(
            staged,
            [
                ".agents/agents/reviewer.md",
                ".agents/skills/xlsx/SKILL.md",
                "AGENTS.md",
                "scripts/guard.py",
            ],
        )

    def test_translates_hooks_json_to_kimi_hook_tables(self):
        with tempfile.TemporaryDirectory() as d:
            inst = KimiSubstrate().materialise(
                _tree(Path(d)), SimpleNamespace(install_mode="kimi_root"), {}
            )
        self.assertEqual(
            tomllib.loads(inst.extras["kimi_hooks"])["hooks"],
            [
                {
                    "event": "PreToolUse",
                    "matcher": "Bash",
                    "command": "python3 scripts/guard.py",
                    "timeout": 5,
                },
                {"event": "Stop", "command": 'echo "done"'},
            ],
        )

    def test_skill_without_name_is_rejected_because_kimi_fails_to_parse_it(self):
        tree = {
            "AGENTS.md": "When reading spreadsheets, read .agents/skills/xlsx/SKILL.md.\n",
            ".agents/skills/xlsx/SKILL.md": "---\ndescription: read spreadsheets\n---\nBody.\n",
        }
        with self.assertRaisesRegex(ValueError, "name: xlsx"):
            HT.validate(KIMI_SURFACE, tree)

    def test_non_numeric_hook_timeout_falls_back_to_kimis_default(self):
        src = json.dumps(
            {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "x", "timeout": "5s"}]}]}}
        )
        self.assertEqual(
            tomllib.loads(hooks_toml(src))["hooks"], [{"event": "Stop", "command": "x"}]
        )

    def test_rejects_other_install_modes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                KimiSubstrate().materialise(
                    _tree(Path(d)), SimpleNamespace(install_mode="codex_root"), {}
                )


BASE_CONFIG = """default_model = "kimi-code/kimi-for-coding"

[[hooks]]
event = "SessionStart"
command = "/home/me/htb-archive start"

[providers."managed:kimi-code"]
type = "kimi"
base_url = "https://api.kimi.ai/coding/v1"
api_key = ""

[[hooks]]
event = "SessionEnd"
matcher = "exit"
command = "/home/me/htb-archive end"

[thinking]
effort = "high"

[[ hooks ]]  # spaced header, trailing comment
event = "Stop"
command = "/home/me/notify"
"""


def _base_home(d: Path) -> Path:
    base = d / "base"
    (base / "credentials").mkdir(parents=True)
    (base / "credentials" / "kimi-code.json").write_text("{}", encoding="utf-8")
    (base / "config.toml").write_text(BASE_CONFIG, encoding="utf-8")
    return base


class KimiHomeTests(unittest.TestCase):
    def test_keeps_provider_config_but_replaces_user_hooks_with_harness_hooks(self):
        harness_hooks = '[[hooks]]\nevent = "Stop"\ncommand = "true"\n'
        with tempfile.TemporaryDirectory() as d:
            home = make_home(_base_home(Path(d)), harness_hooks, Path(d) / "rollout")
            cfg = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["default_model"], "kimi-code/kimi-for-coding")
        self.assertEqual(cfg["providers"]["managed:kimi-code"]["type"], "kimi")
        self.assertEqual(cfg["thinking"], {"effort": "high"})
        self.assertEqual(cfg["hooks"], [{"event": "Stop", "command": "true"}])

    def test_without_harness_hooks_no_user_hook_survives(self):
        with tempfile.TemporaryDirectory() as d:
            home = make_home(_base_home(Path(d)), "", Path(d) / "rollout")
            cfg = tomllib.loads((home / "config.toml").read_text(encoding="utf-8"))
        self.assertNotIn("hooks", cfg)
        self.assertEqual(cfg["thinking"], {"effort": "high"})

    def test_shares_oauth_credentials_with_the_base_home(self):
        with tempfile.TemporaryDirectory() as d:
            base = _base_home(Path(d))
            home = make_home(base, "", Path(d) / "rollout")
            (home / "credentials" / "kimi-code.json").write_text('{"refreshed": 1}')
            self.assertEqual(
                (base / "credentials" / "kimi-code.json").read_text(), '{"refreshed": 1}'
            )

    def test_env_isolates_home_from_the_users_global_agents_dir(self):
        with tempfile.TemporaryDirectory() as d:
            home = make_home(_base_home(Path(d)), "", Path(d) / "rollout")
            env = KimiSubstrate().exec_env({"HOME": "/Users/me", "PATH": "/bin"}, kimi_home=home)
        self.assertEqual(env["KIMI_CODE_HOME"], str(home))
        self.assertEqual(env["HOME"], str(home.parent))
        self.assertEqual(env["PATH"], "/bin")
        self.assertEqual(env["KIMI_DISABLE_TELEMETRY"], "1")


# Stands in for `kimi -p`: reports what the rollout looked like from inside, writes a
# deliverable to the prompt's output directory, and emits a stream-json trajectory.
FAKE_KIMI = """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
prompt = args[args.index("-p") + 1]
out = Path(prompt.split("=== OUTPUT DIRECTORY ===")[1].split()[0])
ws = Path.cwd()
seen = {
    "args": [a for a in args if a != prompt],
    "agents_md": (ws / "AGENTS.md").read_text(),
    "git_root": (ws / ".git").is_dir(),
    "home": os.environ["HOME"],
    "kimi_home": os.environ["KIMI_CODE_HOME"],
    "config": Path(os.environ["KIMI_CODE_HOME"], "config.toml").read_text(),
}
(out / "seen.json").write_text(json.dumps(seen))
print(json.dumps({"role": "meta", "type": "system.version", "version": "2.4.1"}))
print(json.dumps({"role": "assistant", "content": "Saved the report."}))
"""


class KimiJobBenchLauncherTests(unittest.TestCase):
    def test_rolls_a_task_through_kimi_and_keeps_deliverables_and_trajectory(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            kimi = d / "kimi"
            kimi.write_text(FAKE_KIMI, encoding="utf-8")
            kimi.chmod(0o755)
            store = d / "store"
            task = store / "analyst" / "task1" / "task_folder"
            task.mkdir(parents=True)
            (task / "TASK_INSTRUCTIONS.txt").write_text("Write a report.", encoding="utf-8")
            base = _base_home(d)
            with (
                patch.object(config, "KIMI_BIN", str(kimi)),
                patch.object(config, "KIMI_BASE_HOME", base),
                patch.object(config, "SOLVER_MODEL_KI", "kimi-code/k3"),
            ):
                adapter = JobBenchAdapter("ki", store_root=store)
                install = adapter.substrate.materialise(_tree(d / "tree"), adapter.spec, {})
                (run_dir,) = adapter.substrate.execute("analyst/task1", install, ["r/1"], 60)
            seen = json.loads((run_dir / "seen.json").read_text())
            rendered = adapter.trajectory(run_dir, obs_trunc=1000)

        self.assertEqual(run_dir, store / "analyst" / "task1" / "model_output" / "r__1")
        self.assertEqual(
            seen["args"], ["-p", "--output-format", "stream-json", "-m", "kimi-code/k3"]
        )
        self.assertEqual(seen["agents_md"], "Follow the task.\n")
        self.assertTrue(seen["git_root"], "kimi only finds project skills under a .git root")
        self.assertNotEqual(seen["home"], os.environ.get("HOME"))
        self.assertIn("python3 scripts/guard.py", seen["config"])
        self.assertNotIn("htb-archive", seen["config"])
        self.assertEqual(rendered, "[ASSISTANT] Saved the report.")


if __name__ == "__main__":
    unittest.main()
