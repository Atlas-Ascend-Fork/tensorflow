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


@pytest.mark.parametrize("relative", ["configure.py", "tf_adapter_2.x/configure.py"])
@pytest.mark.parametrize("kind", ["file", "directory", "missing"])
def test_ascend_installation_requires_searchable_directory(
    tmp_path, monkeypatch, relative, kind
):
    module = load_module(relative)
    target = tmp_path / "ascend"
    if kind == "file":
        target.write_text("not an installation directory")
    elif kind == "directory":
        target.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "get_input", lambda _: str(target))
    result = module.setup_ascend(str(target))
    assert result is (kind == "directory")
    if kind != "directory":
        assert not (tmp_path / "tools").exists()


@pytest.mark.parametrize("relative", ["configure.py", "tf_adapter_2.x/configure.py"])
def test_valid_python_configuration_retains_all_generated_flags(
    tmp_path, monkeypatch, relative
):
    module = load_module(relative)
    import sys

    executable = sys.executable
    tf_path = str(tmp_path / "tensorflow")
    include_path = str(tmp_path / "include")
    expected_tf_version = "1.15.0" if relative == "configure.py" else "2.6"

    def tensorflow_dependency_boundary(command):
        if command[-1] == "--version":
            return "Python 3.7.0"
        return expected_tf_version + "|" + tf_path + "|-DFLAG=1|" + include_path

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
