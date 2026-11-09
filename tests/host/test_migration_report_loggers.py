# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL_ROOT = Path(__file__).resolve().parents[2] / "convert_tf2npu"
SCRIPT = """
import io
import logging
import sys
sys.path.insert(0, sys.argv[1])
import log
first, second = sys.argv[2:]
capture = io.StringIO()
observer = logging.StreamHandler(capture)
logging.getLogger('success_report.txt').addHandler(observer)
log.init_loggers(first)
logging.getLogger('success_report.txt').info('first run')
log.init_loggers(second)
logging.getLogger('success_report.txt').info('second run')
with open('observer.txt', 'w', encoding='utf-8') as output:
    output.write(capture.getvalue())
"""
FAILURE_SCRIPT = """
import logging
import sys
sys.path.insert(0, sys.argv[1])
import log
log.init_loggers(sys.argv[2])
try:
    log.logger_create('success_report.txt', sys.argv[3])
except OSError:
    logging.getLogger('success_report.txt').info('still active')
else:
    raise AssertionError('invalid report directory was accepted')
"""


class MigrationReportLoggerTest(unittest.TestCase):
    def run_script(self, root, script, *args):
        result = subprocess.run(
            [sys.executable, "-c", script, str(TOOL_ROOT), *(str(arg) for arg in args)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def report_lines(self, directory):
        return (
            (directory / "success_report.txt").read_text(encoding="utf-8").splitlines()
        )

    def test_new_run_does_not_write_to_old_report_or_initial_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            self.run_script(root, SCRIPT, first, second)
            self.assertEqual(self.report_lines(first), ["first run"])
            self.assertEqual(self.report_lines(second), ["second run"])
            self.assertEqual(
                (root / "success_report.txt").read_text(encoding="utf-8"), ""
            )
            self.assertEqual(
                (root / "observer.txt").read_text(encoding="utf-8").splitlines(),
                ["first run", "second run"],
            )

    def test_reinitializing_same_report_does_not_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            self.run_script(root, SCRIPT, report, report)
            self.assertEqual(self.report_lines(report), ["first run", "second run"])

    def test_failed_new_handler_keeps_previous_report_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report"
            invalid = root / "not-a-directory"
            invalid.write_text("file", encoding="utf-8")
            self.run_script(root, FAILURE_SCRIPT, report, invalid)
            self.assertEqual(self.report_lines(report), ["still active"])


if __name__ == "__main__":
    unittest.main()
