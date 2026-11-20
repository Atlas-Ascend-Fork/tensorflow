# TF Adapter简介

Ascend Adapter for TensorFlow 2.x （下称TF Adapter）是TensorFlow 2.x（下称TF2）框架与CANN软件栈间的适配层，用于帮助TF2训练框架的使用者便捷地将训练迁移到AI处理器（简称NPU）上执行。TF Adapter是无侵入的且与TF2有配套关系的Ascend发布件。

TF Adapter在昇腾AI软件栈中的位置如下图所示。

**图 1**  昇腾AI软件栈架构图
![昇腾AI软件栈架构图](./migration/figures/ascend_architecture.png)

## TF2关键概念

TF Adapter涉及的TF2关键概念：

- **Eager模式**

    TF2默认执行方式，运算即时执行并返回具体的值，而非构建供稍后运行的计算图，更多介绍请参考[链接](https://tensorflow.google.cn/guide/eager)。

- **Eager Context**

    TF2 Eager模式下的运行上下文，全局唯一，context中持有线程变量成员，满足不同线程内执行上下文的差异化需求。

- **tf.function**

    TF2提供的Python函数装饰器，用于将Python函数中调用的TF2运算封装成graph执行，从而提升性能，更多介绍请参考[链接](https://tensorflow.google.cn/api_docs/python/tf/function)。

- **TF2自定义设备**

    TF2提供C接口TFE_RegisterCustomDevice提供注册自定义设备的能力，TF Adapter调用该接口将AI处理器注册成为TF的自定义设备，自定义设备与内置的CPU和GPU地位相当。TF2源码信息请参考[链接](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/c_api.cc)。

## TF Adapter对接原理

TF Adapter将AI处理器注册成为TF2自定义设备，并且设置为默认设备，注册成为默认设备后，所有用户指定到AI处理器或未指定执行设备的运算操作，都将被TF2框架分发到AI处理器执行，AI处理器的算子执行接口内部实现时调用CANN的算子/图执行能力，完成AI处理器上的算子执行。

**图 2**  TF Adapter对接框架图
![TF-Adapter对接框架图](./migration/figures/TF-Adapter_architecture.png)

TF Adapter对接时序图：

下图以一次典型的训练流程为例：包括设备初始化，模型（变量）初始化，执行训练，保存Checkpoint几个阶段。

**图 3**  TF Adapter对接时序图
![TF-Adapter对接时序图](./migration/figures/TF-Adapter_sequence_diagram.png)

时序图中涉及的概念说明：

- **CANN**

    AI处理器用户编程接口体系，详情请参考[链接](https://www.hiascend.com/software/cann)。

- **TF2 Runtime**

    这里指原生TensorFlow的运行时接口。

- **Iterator**

    TensorFlow数据输入Pipeline的迭代器，通过Iterator访问数据集，是TensorFlow的推荐方式也是AI处理器上性能亲和的方式，详情请参考[链接](https://tensorflow.google.cn/guide/data)。

- **HDC（Host Device Communication）通道**

    TensorFlow进程到AI处理器硬件内存的数据传输通道，TF Adapter2.x在TensorFlow进程中通过HDC通道，异步地为AI处理器上的训练任务供给训练数据。

## 方案优势

TF Adapter当前对接方案优势：

- NPU成为TF2的自定义设备，从用户视角看来，NPU与GPU/CPU的存在形式一致，且能保持对TF2框架后续演进的兼容性。
- 算子级适配，兼容TF2框架原始特性。特别是函数算子，可充分发挥CANN的图处理优势，加速执行。
- 插件式无侵入对接CANN，无需重新编译部署TF2，快速完成在不同平台上的TF Adapter安装。
