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
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_module(relative):
    spec = importlib.util.spec_from_file_location(
        "host_config_" + relative.replace("/", "_"), ROOT / relative
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_python37_with_empty_response_returns_failure(tmp_path, monkeypatch):
    module = load_module("configure.py")
    monkeypatch.setattr(module, "which", lambda _: None)
    monkeypatch.setattr(module, "get_input", lambda _: "")
    monkeypatch.chdir(tmp_path)
    assert module.setup_python(None) is False
    assert not (tmp_path / "tools").exists()


def test_explicit_missing_path_retains_retry_failure(tmp_path, monkeypatch):
    module = load_module("configure.py")
    missing = str(tmp_path / "missing-python")
    monkeypatch.setattr(module, "which", lambda _: None)
    monkeypatch.setattr(module, "get_input", lambda _: missing)
    monkeypatch.chdir(tmp_path)
    assert module.setup_python(missing) is False
    assert not (tmp_path / "tools").exists()


@pytest.mark.parametrize("relative", ["configure.py"])
def test_valid_python_configuration_retains_all_generated_flags(
    tmp_path, monkeypatch, relative
):
    module = load_module(relative)
    import sys

    executable = sys.executable
    tf_path = str(tmp_path / "tensorflow")
    include_path = str(tmp_path / "include")

    def tensorflow_dependency_boundary(command):
        if command[-1] == "--version":
            return "Python 3.7.0"
        return "1.15.0" + "|" + tf_path + "|-DFLAG=1|" + include_path

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "run_command", tensorflow_dependency_boundary)
    assert module.setup_python(executable) is True
    assert (tmp_path / "tools/PYTHON_BIN_PATH").read_text() == executable
    assert (
        tmp_path / "tools/COMPILE_FLAGS"
    ).read_text() == "-DFLAG=1\n-I" + include_path + "\n"
    if relative == "configure.py":
        expected = (
            tf_path
            + "/libtensorflow_framework.so.1\n"
            + tf_path
            + "/python/_pywrap_tensorflow_internal.so\n"
        )
        for name in ("LINK_FLAGS", "UT_LINK_FLAGS", "ST_LINK_FLAGS"):
            assert (tmp_path / "tools" / name).read_text() == expected
    else:
        assert (tmp_path / "tools/TF_INSTALLED_PATH").read_text() == tf_path
