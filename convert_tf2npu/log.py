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

"""log class"""

import os
import logging
from logging.handlers import RotatingFileHandler

_report_handlers = {}


def logger_create(logger_name, log_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.INFO)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    rotating = RotatingFileHandler(
        filename=log_dir + "/" + logger_name,
        maxBytes=1024**2,
        backupCount=10,
        encoding="utf-8",
    )
    rotating.setFormatter(logging.Formatter("%(message)s"))
    rotating.setLevel(logging.INFO)
    previous = _report_handlers.get(logger_name)
    if previous is not None:
        logger.removeHandler(previous)
        previous.close()
    logger.addHandler(rotating)
    _report_handlers[logger_name] = rotating
    return logger


def init_loggers(log_dir="."):
    global logger_success_report
    logger_success_report = logger_create("success_report.txt", log_dir)

    global logger_failed_report
    logger_failed_report = logger_create("failed_report.txt", log_dir)

    global logger_need_migration_doc
    logger_need_migration_doc = logger_create("need_migration_doc.txt", log_dir)

    global logger_api_brief_report
    logger_api_brief_report = logger_create("api_brief_report.txt", log_dir)


init_loggers()
