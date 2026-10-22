# 内存管理

## memory_config.atomic_clean_policy

是否集中清理网络中所有memset算子占用的内存（含有memset属性的算子都是memset算子）

- 0（默认值）：集中清理。
- 1：单独清理，对网络每一个memset算子进行单独清理。当网络中memset算子内存过大时可尝试此种清理方式，但可能会导致一定的性能损耗。

配置示例：

```python
npu.global_options().memory_config.atomic_clean_policy=1
```

## memory_config.static_memory_policy

网络运行时使用的内存分配方式。

- 0（默认值）：动态分配内存，即按照实际大小动态分配。
- 2：静态shape支持内存动态扩展。网络运行时，可以通过此取值实现同一session中多张图之间的内存复用，即以最大图所需内存进行分配。例如，假设当前执行图所需内存超过前一张图的内存时，直接释放前一张图的内存，按照当前图所需内存重新分配。
- 3：动态shape支持内存动态扩展，解决内存动态分配时的碎片问题，降低动态shape网络内存占用。
- 4：静态shape和动态shape同时支持内存动态扩展。

配置示例：

```python
npu.global_options().memory_config.static_memory_policy=0
```

> [!NOTE]说明
>
> - 多张图并发执行时，不支持配置为“2”和“4”。
> - 为兼容历史版本配置，配置为“1”的场景下，系统会按照“2”的方式进行处理。
> - 配置为“3”和“4”的场景下，将带来内存收益，但可能导致性能损失。

## memory_config.variable_use_1g_huge_page

在推荐模型中，嵌入层\(Embedding层\)在TensorFlow中使用的是变量，当嵌入层作为索引类算子\(Gather、ScatterNd等\)的输入或输出地址时，若内存较大会存在大范围的离散访问，可能会出现算子性能下降问题。此时可尝试通过配置此参数，为变量和常量使用1G大页申请内存，从而提升访存性能。

该参数取值包括：

- 0（默认值）：使用系统默认的4K或者2M页申请内存。
- 1：使用1G大页申请内存，如果申请失败立即打印ERROR日志，并终止业务执行。
- 2：使用1G大页申请内存，如果申请失败会打印ERROR日志，但不终止业务执行，而是转为使用2M页申请内存，如果尝试申请成功，则业务继续执行。如果尝试申请失败，则终止业务执行。

使用1G大页申请内存，可以有效降低页表数量，有效扩大TLB（Translation Lookaside Buffer）缓存的地址范围，从而提升离散访问的性能。TLB是昇腾AI处理器中用于高速缓存的硬件模块，用于存储最近使用的虚拟地址到物理地址的映射。

配置示例：

```python
npu.global_options().memory_config.variable_use_1g_huge_page=1
```

> [!NOTE]说明
>此参数仅支持以下产品：
>
> - Ascend 950PR&950DT系列产品
> - Atlas A3系列产品
> - Atlas A2系列产品

## external_weight

同一个session内同时加载多个模型时，如果多个模型间的权重能够复用，建议通过此配置项将网络中Const/Constant节点的权重外置，实现多个模型间的权重复用，从而减少权重的内存占用。

- False（默认值）：权重不外置，保存在图中。
- True：权重外置，将网络中所有Const/Constant节点的权重文件落盘，并将Const/Constant类型转换为FileConstant。权重文件以“weight_<hash值\>”命名。

  若环境中未配置环境变量ASCEND_WORK_PATH，则权重文件落盘至当前执行目录“tmp_weight_<pid\>_<sessionid\>”下。

  若环境中配置了环境变量ASCEND_WORK_PATH，则权重文件会落盘至$\{ASCEND_WORK_PATH\}/tmp_weight_<pid\>_<sessionid\>目录下。

  模型卸载时，会自动删除“tmp_weight_<pid\>_<sessionid\>”目录。

**说明**：一般场景下不需要配置此参数，针对模型加载环境有内存限制的场景，可以将权重外置。

配置示例：

```python
npu.global_options().external_weight=True
```

## input_fusion_size

Host侧输入数据搬运到Device侧时，将用户离散多个输入数据合并拷贝的阈值。单位为Byte，最小值为0 Byte，最大值为33554432 Byte（32MB），默认值为131072 Byte（128KB）。

- 输入数据大小**<=**阈值，则合并输入，然后从Host搬运到Device。
- 输入数据大小**\>**阈值，或者阈值=0（功能关闭），则不合并，直接从Host搬运到Device。

例如有10个输入，有2个输入数据大小为100KB，2个输入数据大小为50KB，其余输入大于100KB，若设置：

- “input_fusion_size”设置为100KB，则上述4个输入合并为300KB，执行搬运；其他6个输入，直接从Host搬运到Device。
- “input_fusion_size”设置为0KB，则该功能关闭，不进行输入合并，即10个输入直接从Host搬运到Device。

**说明：该参数仅针对静态shape图生效。**

配置示例：

```python
npu.global_options().input_fusion_size=25600
```

## input_batch_cpy

Host侧输入数据搬运到Device时，是否开启批量内存拷贝功能。

- True：开启批量内存拷贝功能。该配置仅在用户输入个数大于1时生效。
- False（默认值）：关闭批量内存拷贝功能。

说明：

- 该参数仅支持以下产品：
  - Ascend 950PR&950DT系列产品
  - Atlas A3系列产品
  - Atlas A2系列产品
- 该参数可以提升Host到Device的数据搬运性能，适用于需要频繁搬运数据且PCIe带宽利用率较低的场景。通过该参数使能批量拷贝功能后，可提升带宽利用率。
- 若网络初始输入个数仅有1个，即使配置了批量拷贝功能也不会生效。
- 当同时配置了“input_fusion_size”参数以启用合并拷贝功能和“input_batch_cpy”参数以启用批量拷贝功能时，合并拷贝的阈值可能会影响批量拷贝功能。

  例如，如果用户有5个输入，其中有4个输入数据小于合并拷贝阈值，满足数据合并条件，那么这4个输入会执行合并拷贝，剩余的1个输入由于不满足批量拷贝的输入个数，则不会执行批量拷贝。

配置示例：

```python
npu.global_options().input_batch_cpy=True
```
