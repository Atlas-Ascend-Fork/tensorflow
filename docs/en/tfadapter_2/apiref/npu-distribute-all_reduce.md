# npu.distribute.all_reduce

## Description

Performs aggregation operation between workers in distributed NPU training.

## Prototype

```python
npu.distribute.all_reduce(values, reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| values | Input | TensorFlow tensor type.<br>For the Atlas A3 products, the supported data types are int8, int32, float16, float32, and bfloat16 (not supported by prod).<br>For the Atlas A2 training products, the supported data types are int8, int32, float16, float32, and bfloat16 (not supported by prod).<br>For the Atlas training products, the supported data types are int8, int32, float16, and float32. |
| reduction | Input | A string.<br>Aggregation operation type. The value can be mean, max, min, prod, or sum. |
| fusion | Input | An int.<br>AllReduce operator fusion flag. The value can be one of the following:<br><br>  - 0: The AllReduce operator is not fused with other AllReduce operators during network compilation.<br>  - 1: The AllReduce operator is fused based on the gradient splitting policy during network compilation.<br>  - 2: AllReduce operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | An int.<br>AllReduce operator fusion ID.<br>If fusion is set to 2, AllReduce operators with the same fusion_id are fused during network compilation. |
| group | Input | A string of up to 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

Result tensor, whose values are consistent with those in the  **values**  input with ordering preserved. It has the same type as  **values**.

## Example

To aggregate a value on multiple devices:

```python
# rank_id = 0  rank_size = 8
import npu_device as npu
v = tf.constant(1.0)
x = npu.distribute.all_reduce([v], 'sum') # 8.0
y = npu.distribute.all_reduce([v], 'mean') # 1.0
```
