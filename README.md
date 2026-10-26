# TensorFlow Adapter For Huawei Ascend

## 概述

TensorFlow Adapter For Huawei Ascend（简称TF Adapter）是昇腾提供的TensorFlow框架适配插件，让Tensorflow框架的开发者可以使用NPU的算力。

开发者只需安装TF Adapter插件，并在现有TensorFlow脚本中添加少量配置，即可实现在昇腾AI处理器上加速自己的训练任务。

![tfadapter](docs/zh/figures/tfadapter_overview.png)

> [!NOTE]说明
> TensorFlow是谷歌公司的商标。

## 支持的TensorFlow版本

TF Adapter支持的TensorFlow版本为TensorFlow 1.15与TensorFlow 2.6.5。

## 支持的产品型号

- Ascend 950PR&950DT系列产品
- Atlas A3系列产品
- Atlas A2训练系列产品
- Atlas训练系列产品
- Atlas推理系列产品（仅支持TensorFlow 1.15在线推理特性）

## 如何使用源码

若您的TensorFlow框架版本是1.15，本源码仓的编译安装等详细使用方法请参见[tf_adapter 1.x](./tf_adapter/README.md)。

若您的TensorFlow框架版本是2.6.5，本源码仓的编译安装等详细使用方法请参见[tf_adapter 2.x](./tf_adapter_2.x/README.md)。

## 学习教程

TF Adapter提供了模型迁移指南、API参考、培训视频等参考材料，详细可参见[TF Adapter资料书架](./docs/README.md)。

## 相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
