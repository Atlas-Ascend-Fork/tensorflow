# Dynamic Shape

## ac_parallel_enable

Whether to allow AI CPU operators and AI Core operators to run in parallel in a dynamic shape graph.

In a dynamic shape graph, when this parameter is enabled, the system automatically identifies AI CPU operators that can be concurrently executed with the AI Core operators in the graph. Operators of different engines are distributed to different streams to implement parallel execution among multiple engines, improving resource utilization and dynamic shape execution performance.

- 1: AI CPU operators and AI Core operators are allowed to run in parallel.
- 0 (default): AI CPU operators are not separately distributed.

Example:

```python
npu.global_options().ac_parallel_enable="1"
```

## compile_dynamic_mode

Whether to generalize all input shapes in the graph.

- True: All input shapes are generalized to -1. Also, static shape graphs are generalized to dynamic ones.
- False (default): Input shapes are not generalized.

Example:

```python
npu.global_options().compile_dynamic_mode=True
```

## all_tensor_not_empty

Whether to remove control nodes for empty tensor checks in the execution graph. In dynamic shape graph scenarios, control nodes are typically inserted to check whether a node is empty to prevent empty tensor nodes from being sent to the device. If you are certain that the graph does not contain empty tensors, you can enable this configuration to remove these control nodes and improve graph execution performance.

- True: Removes the control nodes used for empty tensor checks in the execution graph. Set it to True only when you are sure that the graph does not contain empty tensor nodes; otherwise, some operators may fail.
- False (default): Retains the control nodes used for empty tensor checks in the execution graph.

Configuration example:

```python
npu.global_options().all_tensor_not_empty=True
```
