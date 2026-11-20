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

"""Generate npu_bridge handle"""

import os
import tensorflow
import npu_bridge

try:
    npu_bridge_handle = tensorflow.load_op_library(
        os.path.join(os.path.dirname(npu_bridge.__file__), "_tf_adapter.so")
    )
except Exception as e:
    tensorflow.get_logger().error("Failed to load _tf_adapter.so: %s", e)


def get_gen_ops():
    """Get npu_bridge handle"""
    return npu_bridge_handle


VERSION = "v1.15.0"
