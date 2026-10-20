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

"""Basic class to visit ast node"""

import ast
import util_global
import config


class VisitCall(ast.NodeVisitor):
    """Class for visiting python ast call node"""
    def __init__(self):
        self._in_call = False
        self._current_visit = []
        self.call_nodes = []
        self.line_nos = []

    def visit_Call(self, node):
        """Visit call node"""
        self._in_call = True
        self._current_visit = []
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Visit attr node"""
        if self._in_call:
            self._current_visit.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Visit name in call node"""
        if self._in_call:
            self._current_visit.append(node.id)
            self.call_nodes.append('.'.join(self._current_visit[::-1]))
            self.line_nos.append(getattr(node, "lineno", "None"))
            # Reset the state
            self._in_call = False
            self._current_visit = []
        self.generic_visit(node)


class VisitAttr(ast.NodeVisitor):
    """Class for visiting python ast attr node"""
    def __init__(self):
        self._in_attr = False
        self._current_visit = []
        self.attr_nodes = []
        self.line_nos = []

    def visit_Attribute(self, node):
        """Visit attr node"""
        self._in_attr = True
        self._current_visit.append(node.attr)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Visit name in attr node"""
        if self._in_attr:
            self._current_visit.append(node.id)
            self.attr_nodes.append('.'.join(self._current_visit[::-1]))
            self.line_nos.append(getattr(node, "lineno", "None"))
            # Reset the state
            self._in_attr = False
            self._current_visit = []
        self.generic_visit(node)


class VisitUnsupportedImport(ast.NodeVisitor):
    """Class for visiting python ast import node"""
    def __init__(self):
        self.imports = []
        self.import_modules = []
        self.modules = []
        self.unsupported = ['cupy', 'cupyx', 'pynvml']

    def visit_ImportFrom(self, node):
        """Visit importfrom node"""
        if node.module is not None:
            self.modules = node.module.split('.')
        for value in node.names:
            if isinstance(value, ast.alias):
                classes = value.name.split('.')
                if len(self.modules) == 0:
                    break
                if self.modules[0] in self.unsupported:
                    self.imports.append(classes[0])
                    self.import_modules.append(self.modules[0])
        self.generic_visit(node)

    def visit_Import(self, node):
        """Visit import node"""
        for value in node.names:
            if isinstance(value, ast.alias):
                self.modules = value.name.split('.')
                if len(self.modules) == 0:
                    break
                if self.modules[0] in self.unsupported:
                    self.import_modules.append(self.modules[0])
                    if value.asname is not None:
                        self.imports.append(value.asname)
                    else:
                        self.imports.append(self.modules[0])
        self.generic_visit(node)


def get_tf_api(file_name):
    """
    Args:
        file_name: script file

    Returns:
        list objects containing all the tensorflow apis extracted from file_name
        and corrosponding line number
    """
    util_global.set_value('use_keras_dropout', False)
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitCall()
    visitor.visit(tree)

    # get tensorflow related api
    api = []
    lineno = []
    import_list = config.import_list
    keras_dropout_api = config.keras_dropout_api
    for module in import_list:
        for i in range(len(visitor.call_nodes)):
            if "".join([module, '.']) in visitor.call_nodes[i] and visitor.call_nodes[i].split('.')[0] == module:
                api.append(visitor.call_nodes[i])
                lineno.append(visitor.line_nos[i])

            # get tf api using keras dropout
            if visitor.call_nodes[i] in keras_dropout_api:
                util_global.set_value('use_keras_dropout', True)
    return api, lineno


def get_tf_enume(file_name, enume_list):
    """
    Args:
        file_name: script file
        enume_list: list of all the Tensorflow enumeration apis

    Returns:
        list objects containing all the tensorflow enumeration apis extracted from file_name
        and corrosponding line number
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitAttr()
    visitor.visit(tree)

    # get tensorflow enume api
    api = []
    lineno = []
    for i in range(len(visitor.attr_nodes)):
        if visitor.attr_nodes[i] in enume_list:
            api.append(visitor.attr_nodes[i])
            lineno.append(visitor.line_nos[i])
    return api, lineno


def get_unsupported_api(file_name):
    """
    Args:
        file_name: script file

    Returns:
        list objects containing the unsupported apis and module extracted from file_name
        and corrosponding line number
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = VisitCall()
    visitor.visit(tree)
    unsupported_visitor = VisitUnsupportedImport()
    unsupported_visitor.visit(tree)

    # get unsupported api
    api = []
    lineno = []
    module = []
    for i in range(len(visitor.call_nodes)):
        imports = visitor.call_nodes[i].split('.')[0]
        if imports in unsupported_visitor.imports or visitor.call_nodes[i].startswith('nvml'):
            if visitor.call_nodes[i].startswith('nvml'):
                module.append('pynvml')
            else:
                index = unsupported_visitor.imports.index(imports)
                module.append(unsupported_visitor.import_modules[index])
            api.append(visitor.call_nodes[i])
            lineno.append(visitor.line_nos[i])
    return api, module, lineno


class PreProcessVisitor(ast.NodeVisitor):
    """Class for visiting python ast before convert"""
    def visit_ClassDef(self, node):
        """visit ast node classdef"""
        self.generic_visit(node)

        for base in node.bases:
            if isinstance(base, ast.Attribute) and \
                base.attr == "Model":
                custom_keras_model_name = node.name
                custom_keras_models = util_global.get_value('custom_keras_models', [])
                custom_keras_models.append(custom_keras_model_name)
                util_global.set_value('custom_keras_models', custom_keras_models)
                break


def preprocess_visit(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        source = file.read()
    tree = ast.parse(source)
    visitor = PreProcessVisitor()
    visitor.visit(tree)
