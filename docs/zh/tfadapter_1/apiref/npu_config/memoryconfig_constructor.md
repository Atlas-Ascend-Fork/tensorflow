# MemoryConfig构造函数

## 功能说明

MemoryConfig类的构造函数，用于配置系统内存使用方式。

## 函数原型

```python
class MemoryConfig():
    def __init__(self,
                 atomic_clean_policy=0,
                 static_memory_policy=0,
                 memory_optimization_policy=None,
                 variable_use_1g_huge_page=0):
        ...
```

## 参数说明

- **atomic_clean_policy**：输入，是否集中清理网络中所有memset算子占用的内存（含有memset属性的算子都是memset算子）。
  - 0（默认值）：集中清理。
  - 1：单独清理，对网络每一个memset算子进行单独清理。当网络中memset算子内存过大时可尝试此种清理方式，但可能会导致一定的性能损耗。

- **static_memory_policy**：输入，网络运行时使用的内存分配方式。

  - 0（默认值）：动态分配内存，即按照实际大小动态分配。
  - 2：静态shape支持内存动态扩展。网络运行时，可以通过此取值实现同一session中多张图之间的内存复用，即以最大图所需内存进行分配。例如，假设当前执行图所需内存超过前一张图的内存时，直接释放前一张图的内存，按照当前图所需内存重新分配。
  - 3：动态shape支持内存动态扩展，解决内存动态分配时的碎片问题，降低动态shape网络内存占用。
  - 4：静态shape和动态shape同时支持内存动态扩展。

    > [!NOTE]说明
    >- 多张图并发执行时，不支持配置为“2”和“4”。
    >- 为兼容历史版本配置，配置为“1”的场景下，系统会按照“2”的方式进行处理。
    >- 配置为“3”和“4”的场景下，将带来内存收益，但可能导致性能损失。

- **memory_optimization_policy**：输入，可选内存优化策略，默认为None；未设置时不向NpuOptimizer传入该配置。此参数不是大页配置参数；1G大页由`variable_use_1g_huge_page`控制，其取值见下方说明。建议使用关键字参数，例如`MemoryConfig(variable_use_1g_huge_page=1)`，避免位置参数混淆。

- **variable_use_1g_huge_page**：输入，在推荐模型中，嵌入层\(Embedding层\)在TensorFlow中使用的是变量，当嵌入层作为索引类算子\(Gather、ScatterNd等\)的输入或输出地址时，若内存较大会存在大范围的离散访问，可能会出现算子性能下降问题。此时可尝试通过配置此参数，为变量和常量使用1G大页申请内存，从而提升访存性能。

    该参数取值包括：

  - 0（默认值）：使用系统默认的4K或者2M页申请内存。
  - 1：使用1G大页申请内存，如果申请失败立即打印ERROR日志，并终止业务执行。
  - 2：使用1G大页申请内存，如果申请失败会打印ERROR日志，但不终止业务执行，而是转为使用2M页申请内存，如果尝试申请成功，则业务继续执行。如果尝试申请失败，则终止业务执行。

    使用1G大页申请内存，可以有效降低页表数量，有效扩大TLB（Translation Lookaside Buffer）缓存的地址范围，从而提升离散访问的性能。TLB是昇腾AI处理器中用于高速缓存的硬件模块，用于存储最近使用的虚拟地址到物理地址的映射。

    > [!NOTE]说明
    >此参数仅支持以下产品：
    >
    >- Ascend 950PR&950DT系列产品
    >- Atlas A3系列产品
    >- Atlas A2系列产品

## 返回值

返回MemoryConfig类对象，作为NPURunConfig的参数传入。

## 约束说明

无

## 调用示例

```python
from npu_bridge.npu_init import *
...
# 使用1G大页；申请失败时终止业务执行。
mem_config = MemoryConfig(atomic_clean_policy=0,
                          static_memory_policy=0,
                          variable_use_1g_huge_page=1)
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(memory_config=mem_config, session_config=session_config)
```
