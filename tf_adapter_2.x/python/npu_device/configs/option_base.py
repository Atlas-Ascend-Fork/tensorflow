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

"""NPU basic configurations"""

import logging


class OptionValue:
    """Options for setting npu basic configurations"""

    def __init__(self, default, optional):
        self.__default = default
        self.__optional = optional
        self.__value = default

    @property
    def default(self):
        """Return property"""
        return self.__default

    @property
    def optional(self):
        """Return property"""
        return self.__optional

    @property
    def value(self):
        """Return option value"""
        if self.__value is None:
            return None
        if str(self.__value) == str(True):
            return "1"
        if str(self.__value) == str(False):
            return "0"
        return str(self.__value)

    @value.setter
    def value(self, v):
        if (
            isinstance(
                self.__optional,
                (
                    tuple,
                    list,
                ),
            )
            and v not in self.__optional
        ):
            if self.__default is not None and not isinstance(v, type(self.__default)):
                raise TypeError(
                    "Expected "
                    + type(self.__default).__name__
                    + ", got "
                    + type(v).__name__
                )
            else:
                raise ValueError(
                    "'" + str(v) + "' not in optional list " + str(self.__optional)
                )
        self.__value = v


class DeprecatedValue(OptionValue):
    def __init__(self, optional, *, replacement):
        super().__init__(None, optional)
        self.replacement = replacement


class NpuBaseConfig:
    """NPU basic configurations"""

    def __init__(self):
        self._fixed_attrs = []
        for k, v in self.__dict__.items():
            if isinstance(v, (OptionValue, NpuBaseConfig)):
                self._fixed_attrs.append(k)

    def __setattr__(self, key, value):
        if hasattr(self, "_fixed_attrs"):
            if key not in self._fixed_attrs:
                raise ValueError(
                    self.__class__.__name__
                    + " has no option "
                    + key
                    + ", all options "
                    + str(self._fixed_attrs)
                )
            if isinstance(getattr(self, key), OptionValue):
                getattr(self, key).value = value
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def as_dict(self):
        """Return updated option in dictionary format"""
        options = {}
        for k, v in self.__dict__.items():
            if k in self._fixed_attrs:
                if isinstance(v, DeprecatedValue) and v.value is not None:
                    if v.replacement is None:
                        logging.warning(
                            "[tf_adapter] Option '%s' is deprecated and will be removed "
                            "in future version. Please do not configure this option in the future.",
                            k,
                        )
                    else:
                        logging.warning(
                            "[tf_adapter] Option '%s' is deprecated and will be removed "
                            "in future version. Please use '%s' instead.",
                            k,
                            v.replacement,
                        )
                    options.update({k: v.value})
                elif isinstance(v, OptionValue) and v.value is not None:
                    options.update({k: v.value})
                elif isinstance(v, NpuBaseConfig):
                    options.update(v.as_dict())
        return options
