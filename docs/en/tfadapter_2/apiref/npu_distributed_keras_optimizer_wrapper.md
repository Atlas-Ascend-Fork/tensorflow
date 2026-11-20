# npu.distribute.npu_distributed_keras_optimizer_wrapper

## Description

Adds the AllReduce operation of the NPU to aggregate the gradients, and then updates the gradients. This API applies only to distributed training.

## Prototype

```python
def npu_distributed_keras_optimizer_wrapper(optimizer, reduce_reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| optimizer | Input | TensorFlow gradient training optimizer. |
| reduce_reduction | Input | A string.<br>Aggregation operation type. The value can be mean, max, min, prod, or sum. |
| fusion | Input | An int.<br>AllReduce operator fusion flag. The value can be one of the following:<br><br>  - 0: The AllReduce operator is not fused with other AllReduce operators during network compilation.<br>  - 1: The AllReduce operator is fused based on the gradient splitting policy during network compilation.<br>  - 2: AllReduce operators with the same fusion_id are fused during network compilation. |
| fusion_id | Input | An int.<br>AllReduce operator fusion ID.<br>If fusion is set to 2, AllReduce operators with the same fusion_id are fused during network compilation. |
| group | Input | A string of up to 128 bytes, including the end character.<br>Group name, which can be a user-defined value or hccl_world_group. |

## Returns

TensorFlow gradient training optimizer.

## Example

```python
import npu_device as npu
optimizer = tf.keras.optimizers.SGD()
optimizer = npu.distribute.npu_distributed_keras_optimizer_wrapper(optimizer) # Use NPU-based distributed computing to update gradients.
model.compile(optimizer = optimizer)
```
