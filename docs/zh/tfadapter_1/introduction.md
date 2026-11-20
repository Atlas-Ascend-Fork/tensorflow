# 前言

描述本手册的读者对象，TensorFlow 1.15模型迁移的全流程，以及使用本手册的一些注意点。

## 读者对象

本文档适用于AI算法工程师，用于将基于TensorFlow 1.15的Python API开发的网络脚本迁移到NPU上执行。当前NPU支持TensorFlow 1.15的三种API开发的网络脚本迁移：分别是Estimator，Session，Keras。

掌握以下经验和技能可以更好地理解本文档：

- 熟悉CANN软件基本架构以及特性。
- 熟悉TensorFlow 1.15的API。
- 对机器学习、深度学习有一定的了解，熟悉训练网络的基本知识与流程。
- 具有熟练的Python语言编程能力。

## 支持的产品

- Ascend 950PR&950DT系列产品
- Atlas A3系列产品
- Atlas A2系列产品
- Atlas训练系列产品
- Atlas推理系列产品（仅支持在线推理特性）

## 使用前须知

- 在NPU进行模型迁移之前，建议用户事先准备好基于TensorFlow 1.15开发的训练模型以及配套的数据集，并要求在GPU或CPU上运行通过，达到预期精度和性能要求，同时记录相关精度和性能指标，用于后续在NPU进行精度和性能对比。
- 本文中的代码片段仅为示例，用户使用时请注意修改适配。

## 系统约束与限制

1. 该文档仅配套TensorFlow 1.15版本使用。
2. 当前版本不支持float64、complex64、complex128、DT_VARIANT数据类型。
3. 目前系统支持的数据format主要有NCHW、NHWC、NC、HWCN、CN。
4. 条件分支、循环分支只支持tf.cond、tf.while_loop、tf.case。
5. 多卡训练时，NPURunConfig不支持tf.estimator.RunConfig中的配置参数save_checkpoints_secs。
6. 多卡训练时，不支持仅保存单卡的Summary信息（tf.summary接口）。
7. 针对Atlas训练系列产品，算子不支持inf/nan输入。
8. 数据预处理约束：当前不支持queue方式读取数据，仅支持dataset和placeholder方式。
9. 如果使用python的多进程包multiprocessing创建多进程，请不要使用fork方法，建议使用forkserver方法。

   因为在Python3.8~Python3.11版本中如果使用fork方法，在创建子进程时可能会复制主进程的锁状态，而在子进程里再触发获取锁时，就会导致死锁，进而导致业务进程卡死。
