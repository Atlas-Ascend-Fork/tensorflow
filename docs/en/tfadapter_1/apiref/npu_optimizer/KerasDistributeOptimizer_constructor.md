# KerasDistributeOptimizer Constructor

## Description

Constructs an object of class  **KerasDistributeOptimizer**, which wraps around the single-server training optimizer constructed by  **tf.keras**  to an NPU distributed training optimizer.

## Prototype

```python
class KerasDistributeOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, optimizer, name="NpuKerasOptimizer", **kwargs)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| optimizer | Input | Single-server training optimizer for gradient calculation and weight update. |
| name | Input | Name of the optimizer. |

## Returns

An object of the  **KerasDistributeOptimizer**  class

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

model=xxx
model.compile(loss='mean_squared_error', optimizer=KerasDistributeOptimizer(tf.keras.optimizers.SGD()))
```
