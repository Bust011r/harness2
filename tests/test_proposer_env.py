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

import unittest
from unittest.mock import patch

from harness2.core.proposer import Proposer
from harness2.core.spec import resolve_channel_model


class ProposerEnvTests(unittest.TestCase):
    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
    def test_opencode_channel_keeps_the_anthropic_key_for_an_anthropic_model(self):
        spec = resolve_channel_model("opencode", "anthropic/claude-sonnet-5-5")
        self.assertEqual(Proposer(spec).env().get("ANTHROPIC_API_KEY"), "sk-test")

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"})
    def test_claude_cli_channel_drops_it_so_vertex_billing_is_not_bypassed(self):
        spec = resolve_channel_model("claude_cli", "claude-sonnet-5-5")
        self.assertNotIn("ANTHROPIC_API_KEY", Proposer(spec).env())


if __name__ == "__main__":
    unittest.main()
