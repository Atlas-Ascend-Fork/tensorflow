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
util_global = importlib.import_module("util_global")
visit_by_ast = importlib.import_module("visit_by_ast")


class MigrationUtf8BomTest(unittest.TestCase):
    def test_scanners_accept_python_files_with_or_without_bom(self):
        util_global.init()
        code = "import tensorflow as tf\nimport cupy\nx = tf.constant(1)\ny = tf.float32\ncupy.array([1])\n"
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "plain.py"
            marked = Path(tmp) / "marked.py"
            plain.write_text(code, encoding="utf-8")
            marked.write_text(code, encoding="utf-8-sig")
            compile(marked.read_bytes(), str(marked), "exec")
            for scan in (
                visit_by_ast.preprocess_visit,
                visit_by_ast.get_tf_api,
                lambda path: visit_by_ast.get_tf_enume(path, ["tf.float32"]),
                visit_by_ast.get_unsupported_api,
            ):
                with self.subTest(scanner=getattr(scan, "__name__", "enum")):
                    self.assertEqual(scan(str(marked)), scan(str(plain)))


if __name__ == "__main__":
    unittest.main()
