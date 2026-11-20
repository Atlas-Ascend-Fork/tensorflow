# npu.distribute.broadcast

## Description

Synchronizes variables between workers in distributed NPU training.

## Prototype

```python
npu.distribute.broadcast(values, root_rank, fusion=2, fusion_id=0, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| values | Input | A TensorFlow variable or variable set.<br>For the Atlas training products, the supported data types are int8, int32, float16, float32, int64, and uint64.<br>For the Atlas A2 training products, the supported data types are int8, int32, float16, float32, int64, uint64, and bfloat16.<br>For the Atlas A3 products, the supported data types are int8, int32, float16, float32, int64, uint64, and bfloat16. |
| root_rank | Input | An int.<br>Rank ID of the root rank. Must be a rank ID in the group. |
| fusion | Input | An int.<br>Broadcast operator fusion flag. The value can be one of the following:<br><br>  - 0: The Broadcast operator is not fused with other Broadcast operators during network compilation.<br>  - 2: Broadcast operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | An int.<br>Broadcast operator fusion ID.<br>If fusion is set to 2, Broadcast operators with the same fusion_id are fused during network compilation. |
| group | Input | A string of up to 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

None

## Example

To broadcast the variables on device 0 to the rest devices:

```python
# rank_id = 0  rank_size = 8
import npu_device as npu
x = tf.Variable(tf.random.normal(shape=()))
print("before broadcast", x)
npu.distribute.broadcast(x, root_rank=0)
print("after_broadcast", x)
```

Before the broadcast

![](figures/before_broadcast.png)

After the broadcast

![](figures/after_broadcast.png)
