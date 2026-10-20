#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Basic configurations for installing Tensorflow adaptor"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys

try:
    from shutil import which
except ImportError:
    from distutils.spawn import find_executable as which

_COMPAT_TENSORFLOW_VERSION = "1.15.0"
_COMPAT_PYTHON_VERSION = "Python 3.7"
_COMPAT_SWIG_VERSION = "SWIG Version "
_PYTHON_BIN_PATH_ENV = "ADAPTER_TARGET_PYTHON_PATH"
_ASCEND_HOME_PATH_ENV = "ASCEND_HOME_PATH"
_RETRY_COUNT_MAX = 5


def run_command(cmd):
    """Execute command"""
    output = subprocess.check_output(cmd)
    return output.decode("UTF-8").strip()


def get_input(q):
    """Get user input from the keyboard"""
    try:
        try:
            ans = raw_input(q)
        except NameError:
            ans = input(q)
    except EOFError:
        ans = ""
    return ans


def real_config_path(file):
    """Get complete file path"""
    os.makedirs("tools", exist_ok=True)
    return os.path.join("tools", file)


def _write_python_configuration(python_bin_path, compile_args):
    with open(real_config_path("PYTHON_BIN_PATH"), "w") as f:
        f.write(python_bin_path)
    with open(real_config_path("COMPILE_FLAGS"), "w") as f:
        for flag in compile_args[2:-1]:
            f.write("".join([flag, "\n"]))
        f.write("".join(["-I", compile_args[-1], "\n"]))
    print("tensorflow path: %s." % compile_args[1])
    with open(real_config_path("LINK_FLAGS"), "w") as f:
        f.write(os.path.join(compile_args[1], "libtensorflow_framework.so.1\n"))
        f.write(
            os.path.join(compile_args[1], "python", "_pywrap_tensorflow_internal.so\n")
        )
    with open(real_config_path("UT_LINK_FLAGS"), "w") as f:
        f.write(os.path.join(compile_args[1], "libtensorflow_framework.so.1\n"))
        f.write(
            os.path.join(compile_args[1], "python", "_pywrap_tensorflow_internal.so\n")
        )
    with open(real_config_path("ST_LINK_FLAGS"), "w") as f:
        f.write(os.path.join(compile_args[1], "libtensorflow_framework.so.1\n"))
        f.write(
            os.path.join(compile_args[1], "python", "_pywrap_tensorflow_internal.so\n")
        )


def setup_python(env_path):
    """Get python install path."""
    default_python_bin_path = which("python3.7")
    ask_python_bin_path = (
        "Please specify the location of python 3.7 with valid "
        "tensorflow 1.15.0 site-packages installed. [Default "
        "is %s]\n(You can make this quiet by set env "
        "[ADAPTER_TARGET_PYTHON_PATH]): "
    ) % default_python_bin_path
    custom_python_bin_path = env_path if env_path else default_python_bin_path

    is_success = False
    retry_count = 0
    while not is_success:
        retry_count += 1
        if retry_count > _RETRY_COUNT_MAX:
            break

        if not custom_python_bin_path:
            python_bin_path = get_input(ask_python_bin_path)
        else:
            python_bin_path = custom_python_bin_path
            custom_python_bin_path = None
        if not python_bin_path:
            python_bin_path = default_python_bin_path
        # Check if the path is valid
        if os.path.isfile(python_bin_path) and os.access(python_bin_path, os.X_OK):
            compile_args = run_command([python_bin_path, "--version"])
            if _COMPAT_PYTHON_VERSION not in compile_args:
                print(
                    "Invalid default python version: %s, only support Python 3.7."
                    % compile_args
                )
                continue
        elif not os.path.exists(python_bin_path):
            print("Invalid python path: %s cannot be found." % python_bin_path)
            continue
        else:
            print("%s is not executable.  Is it the python binary?" % python_bin_path)
            continue

        try:
            compile_args = run_command(
                [
                    python_bin_path,
                    "-c",
                    'import distutils.sysconfig; import tensorflow as tf; print(tf.__version__ + "|" + '
                    "tf.sysconfig.get_lib("
                    ') + "|" + "|".join(tf.sysconfig.get_compile_flags()) + "|" + '
                    "distutils.sysconfig.get_python_inc())",
                ]
            ).split("|")
        except subprocess.CalledProcessError:
            print("Invalid python path: %s tensorflow not installed." % python_bin_path)
            continue
        if not compile_args[0].startswith(_COMPAT_TENSORFLOW_VERSION):
            print(
                "Invalid python path: %s compat tensorflow version is %s"
                " got %s."
                % (python_bin_path, _COMPAT_TENSORFLOW_VERSION, compile_args[0])
            )
            continue
        is_success = True

    if is_success:
        _write_python_configuration(python_bin_path, compile_args)

    return is_success


def setup_ascend(env_path):
    """Get ascend install path."""
    default_ascend_path = os.path.realpath("/usr/local/Ascend/cann")
    ask_ascend_path = (
        "Please specify the location of ascend. [Default is "
        "%s]\n(You can make this quiet by set env [ASCEND_HOME_PATH]): "
    ) % default_ascend_path
    custom_ascend_path = env_path

    is_success = False
    retry_count = 0
    while not is_success:
        retry_count += 1
        if retry_count > _RETRY_COUNT_MAX:
            break

        if not custom_ascend_path:
            ascend_path = get_input(ask_ascend_path)
        else:
            ascend_path = custom_ascend_path
            custom_ascend_path = None
        if not ascend_path:
            ascend_path = default_ascend_path
        # Check if the path is valid
        if not os.path.isdir(ascend_path) or not os.access(ascend_path, os.X_OK):
            print(
                "Invalid ascend path: %s must be a searchable directory." % ascend_path
            )
            continue
        is_success = True

    if is_success:
        with open(real_config_path("ASCEND_INSTALL_PATH"), "w") as f:
            f.write(ascend_path)
        with open(real_config_path("LINK_FLAGS"), "a") as f:
            f.write(os.path.join(ascend_path, "lib64", "libge_runner.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libfmk_parser.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libfmk_onnx_parser.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libdatatransfer.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libindextransform.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libascendcl.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libascendalog.so\n"))
            f.write(os.path.join(ascend_path, "lib64", "libacl_tdt_channel.so\n"))

    return is_success


def setup_swig():
    """Get swig install path."""
    default_swig_path = which("swig")
    ask_swig_path = (
        "Please specify the location of swig. [Default is "
        "%s]\n(Please enter the correct swig path: "
    ) % default_swig_path
    custom_swig_path = default_swig_path

    is_success = False
    retry_count = 0
    while not is_success:
        retry_count += 1
        if retry_count > _RETRY_COUNT_MAX:
            break

        if not custom_swig_path:
            swig_path = get_input(ask_swig_path)
        else:
            swig_path = custom_swig_path
            custom_swig_path = None
        # Check if the path is valid
        if os.path.isfile(swig_path) and os.access(swig_path, os.X_OK):
            compile_args = run_command([swig_path, "-version"])
            if _COMPAT_SWIG_VERSION not in compile_args:
                print("Invalid default swig version: %s." % compile_args)
                continue
        elif not os.path.exists(swig_path):
            print("Invalid swig path: %s cannot be found." % swig_path)
            continue
        else:
            print("%s is not executable.  Is it the swig binary?" % swig_path)
            continue
        is_success = True

    if is_success:
        # Write tools/SWIG_BIN_PATH
        with open(real_config_path("SWIG_BIN_PATH"), "w") as f:
            f.write(swig_path)

    return is_success


def main():
    """Entry point for configuration"""
    env_snapshot = dict(os.environ)
    if not setup_python(env_snapshot.get(_PYTHON_BIN_PATH_ENV)):
        sys.exit(1)
    if not setup_ascend(env_snapshot.get(_ASCEND_HOME_PATH_ENV)):
        sys.exit(1)
    if not setup_swig():
        sys.exit(1)


if __name__ == "__main__":
    main()
