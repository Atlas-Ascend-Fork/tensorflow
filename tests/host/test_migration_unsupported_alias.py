# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib
from pathlib import Path
import sys
import tempfile
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[2] / "convert_tf2npu"
sys.path.insert(0, str(TOOL_ROOT))
visit_by_ast = importlib.import_module("visit_by_ast")


class UnsupportedAliasTest(unittest.TestCase):
    def test_imported_api_aliases_are_reported(self):
        source = (
            "from cupy import array as gpu_array\n"
            "from cupy import random as gpu_random\n"
            "from cupy.linalg import norm as gpu_norm\n"
            "from pynvml import nvmlInit as init_gpu\n"
            "gpu_array([1])\n"
            "gpu_random.rand(2)\n"
            "gpu_norm([1])\n"
            "init_gpu()\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.py"
            path.write_text(source, encoding="utf-8")
            apis, modules, lines = visit_by_ast.get_unsupported_api(str(path))
        self.assertEqual(apis, ["gpu_array", "gpu_random.rand", "gpu_norm", "init_gpu"])
        self.assertEqual(modules, ["cupy", "cupy", "cupy", "pynvml"])
        self.assertEqual(lines, [5, 6, 7, 8])

    def test_unaliased_and_module_imports_remain_supported(self):
        source = (
            "from cupy import array\nimport cupy as cp\narray([1])\ncp.array([2])\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.py"
            path.write_text(source, encoding="utf-8")
            apis, modules, lines = visit_by_ast.get_unsupported_api(str(path))
        self.assertEqual(apis, ["array", "cp.array"])
        self.assertEqual(modules, ["cupy", "cupy"])
        self.assertEqual(lines, [3, 4])


if __name__ == "__main__":
    unittest.main()
