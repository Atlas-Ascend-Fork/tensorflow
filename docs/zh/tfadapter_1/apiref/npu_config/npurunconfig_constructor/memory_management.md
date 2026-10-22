# 内存管理

## memory_config

用于配置系统内存使用方式，用户在创建NPURunConfig之前，可以实例化一个MemoryConfig类进行功能配置。MemoryConfig类的构造函数，请参见[MemoryConfig构造函数](../memoryconfig_constructor.md)。

## external_weight

同一个session内同时加载多个模型时，如果多个模型间的权重能够复用，建议通过此配置项将网络中Const/Constant节点的权重外置，实现多个模型间的权重复用，从而减少权重的内存占用。

- False（默认值）：权重不外置，保存在图中。
- True：权重外置，将网络中所有Const/Constant节点的权重文件落盘，并将Const/Constant类型转换为FileConstant。权重文件以“weight_<hash值\>”命名。

若环境中未配置环境变量ASCEND_WORK_PATH，则权重文件落盘至当前执行目录“tmp_weight_<pid\>_<sessionid\>”下。

若环境中配置了环境变量ASCEND_WORK_PATH，则权重文件会落盘至\$\{ASCEND_WORK_PATH\}/tmp_weight_<pid\>_<sessionid\>目录下，关于ASCEND_WORK_PATH的详细说明，可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“安装配置相关”章节。

模型卸载时，会自动删除“tmp_weight_<pid\>_<sessionid\>”目录。

说明：一般场景下不需要配置此参数，针对模型加载环境有内存限制的场景，可以将权重外置。

配置示例：

```python
config = NPURunConfig(external_weight=True)
```

## input_fusion_size

Host侧输入数据搬运到Device侧时，将用户离散多个输入数据合并拷贝的阈值。单位为Byte，最小值为0 Byte，最大值为33554432 Byte（32MB），默认值为131072 Byte（128KB）。

- 若输入数据大小`<=`阈值，则合并输入，然后从Host搬运到Device。
- 若输入数据大小`>`阈值，或者阈值=0（功能关闭），则不合并，直接从Host搬运到Device。

例如用户有10个输入，有2个输入数据大小为100KB，2个输入数据大小为50KB，其余输入大于100KB，若设置：

- “input_fusion_size”设置为100KB，则上述4个输入合并为300KB，执行搬运；其他6个输入，直接从Host搬运到Device。
- “input_fusion_size”设置为0KB，则该功能关闭，不进行输入合并，即10个输入直接从Host搬运到Device。

**说明：该参数仅针对静态shape图生效。**

配置示例：

```python
config = NPURunConfig(input_fusion_size=25600)
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
config = NPURunConfig(input_batch_cpy=True)
```
