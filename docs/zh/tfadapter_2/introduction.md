# 前言

描述本手册的读者对象，以及使用本手册的一些注意点。

## 读者对象

本文档适用于AI算法工程师，将基于TensorFlow 2.6.5的Python API开发的训练脚本迁移到NPU上执行训练，并进行精度性能的调试调优。

掌握以下经验和技能可以更好地理解本文档：

- 熟悉CANN软件基本架构以及特性。
- 熟悉TensorFlow的API。
- 对机器学习、深度学习有一定的了解，熟悉训练网络的基本知识与流程。
- 具有熟练的Python语言编程能力。

## 支持的产品

Ascend 950PR&950DT系列产品

Atlas A3系列产品

Atlas A2系列产品

Atlas训练系列产品

## 使用前须知

- 在进行模型迁移之前，建议用户事先准备好基于TensorFlow 2.6.5开发的训练模型以及配套的数据集，并要求在GPU或CPU上正常运行，达到预期精度和性能要求。同时记录相关精度和性能指标，用于后续在AI处理器进行精度和性能对比。
- 本文中的代码片段仅为示例，请用户使用时注意修改适配。

## 系统约束与限制

- 该文档仅配套TensorFlow 2.6.5版本使用。
- 当前版本不支持float64/complex64/complex128/DT_VARIANT数据类型。
- 只支持变量（tf.Variable）资源相关操作在NPU执行。
- 只支持tf.function修饰的函数算子在NPU执行。
- 不支持训练脚本中同时使用tf.compat.v1接口和TF 2.6.5中Eager功能相关的API。
- TensorFlow 2.6.5数据预处理过程默认在Host上执行，而变量需要下沉到Device上初始化。因此若数据预处理脚本中包含了变量，可能会在NPU上执行失败。为了解决此问题，应将包含变量的数据预处理代码嵌套在context.device\('CPU:0'\)中，确保预处理代码中的变量在Host上初始化。

    ```python
    import tensorflow as tf
    from tensorflow.python.eager import context
    with context.device('cpu:0'):
        # 此处写数据预处理代码，以下仅为示例
        x = tf.Variable([1, 2, 3])
        y = tf.square(x)
    ```

- 如果使用Python的多进程包multiprocessing创建多进程，请不要使用fork方法，建议使用forkserver方法。

  因为在Python3.8~Python3.11版本中如果使用fork方法，在创建子进程时可能会复制主进程的锁状态，而在子进程里再触发获取锁时，就会导致死锁，进而导致业务进程卡死。
