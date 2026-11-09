# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import getopt
import importlib.util
from pathlib import Path
import sys
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[2] / "convert_tf2npu"
sys.path.insert(0, str(TOOL_ROOT))


def load_config(path):
    spec = importlib.util.spec_from_file_location("test_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.param_config


class DistributedModeOptionTest(unittest.TestCase):
    def test_option_forms_include_the_mode_argument(self):
        for path in (TOOL_ROOT / "config.py", TOOL_ROOT / "v2" / "config_v2.py"):
            config = load_config(path)
            for args, expected in (
                (
                    ["--distributed_mode=tf_strategy"],
                    ("--distributed_mode", "tf_strategy"),
                ),
                (
                    ["--distributed_mode", "tf_strategy"],
                    ("--distributed_mode", "tf_strategy"),
                ),
                (["-d", "tf_strategy"], ("-d", "tf_strategy")),
            ):
                with self.subTest(config=path.name, args=args):
                    options, remaining = getopt.getopt(
                        args, config.short_opts, config.long_opts
                    )
                    self.assertEqual(options, [expected])
                    self.assertEqual(remaining, [])

    def test_v2_compatibility_flag_remains_argument_free(self):
        config = load_config(TOOL_ROOT / "v2" / "config_v2.py")
        options, remaining = getopt.getopt(
            ["--compat", "--distributed_mode=horovod"],
            config.short_opts,
            config.long_opts,
        )
        self.assertEqual(options, [("--compat", ""), ("--distributed_mode", "horovod")])
        self.assertEqual(remaining, [])


if __name__ == "__main__":
    unittest.main()
