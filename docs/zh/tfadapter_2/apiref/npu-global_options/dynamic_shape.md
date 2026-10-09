# 动态shape

## ac_parallel_enable

动态shape图中，是否允许AI CPU算子和AI Core算子并行运行。

动态shape图中，开关开启时，系统自动识别图中可以和AI Core并发的AI CPU算子，不同引擎的算子下发到不同流上，实现多引擎间的并行，从而提升资源利用效率和动态shape执行性能。

- 1：允许AI CPU和AI Core算子间的并行运行。
- 0（默认值）：AI CPU算子不会单独分流。

配置示例：

```python
npu.global_options().ac_parallel_enable="1"
```

## compile_dynamic_mode

是否需要泛化图中所有的输入shape。

- True：将所有的输入shape泛化为-1，如果是静态shape图，则会泛化为动态shape图。
- False（默认值）：不泛化输入shape。

配置示例：

```python
npu.global_options().compile_dynamic_mode=True
```

## all_tensor_not_empty

动态shape计算图场景，为避免将空tensor节点下发到device，执行图通常会插入控制节点用于判断当前节点是否为空。如果用户确认计算图中不存在空tensor，可通过开启此配置移除这些控制节点，从而提升图执行性能。

- True：移除执行图中用于空tensor判断的控制节点。仅在确认计算图中不存在tensor节点时开启，否则可能导致部分算子执行出错。
- False（默认值）：保留执行图中用于空tensor判断的控制节点。

配置示例：

```python
npu.global_options().all_tensor_not_empty=True
```
