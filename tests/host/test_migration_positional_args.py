# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[2] / "convert_tf2npu"
SCRIPT = """
import ast
import json
import sys
sys.path.insert(0, sys.argv[1])
import pasta
import util_global
import ast_util
util_global.init()
util_global.set_value('distributed_mode', 'tf_strategy')
util_global.set_value('path', 'fixture.py')
tree = pasta.parse(sys.argv[2])
ast_util.add_npu_func_to_params(tree.body[0].value, 1, 'Estimator.train', 'hooks',
                               'npu_hooks_append', 'hooks_list')
converted = pasta.dump(tree)
events = []
def tick(value):
    events.append(value)
    return value
def npu_hooks_append(hooks_list=None):
    events.append('wrap')
    return ['wrapped', hooks_list]
def f(input_fn, hooks=None, steps=None):
    return [input_fn, hooks, steps]
result = eval(compile(ast.Expression(ast.parse(converted).body[0].value), 'converted.py', 'eval'))
print(json.dumps({'result': result, 'events': events}))
"""


@unittest.skipUnless(
    importlib.util.find_spec("pasta") and importlib.util.find_spec("pandas"),
    "migration tool dependencies are not installed",
)
class MigrationPositionalArgsTest(unittest.TestCase):
    def run_case(self, source):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", SCRIPT, str(TOOL_ROOT), source],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_trailing_positional_arguments_keep_position_and_order(self):
        data = self.run_case("f(tick('input'), tick('hooks'), tick('steps'))")
        self.assertEqual(data["result"], ["input", ["wrapped", "hooks"], "steps"])
        self.assertEqual(data["events"], ["input", "hooks", "wrap", "steps"])

    def test_keyword_argument_remains_supported(self):
        data = self.run_case(
            "f(tick('input'), hooks=tick('hooks'), steps=tick('steps'))"
        )
        self.assertEqual(data["result"], ["input", ["wrapped", "hooks"], "steps"])
        self.assertEqual(data["events"], ["input", "hooks", "wrap", "steps"])

    def test_omitted_argument_still_gets_default_wrapper(self):
        data = self.run_case("f(tick('input'))")
        self.assertEqual(data["result"], ["input", ["wrapped", None], None])
        self.assertEqual(data["events"], ["input", "wrap"])


if __name__ == "__main__":
    unittest.main()
