# NPUBroadcastGlobalVariablesCallback Constructor

## Description

Broadcasts variables in Keras scenarios to ensure that the initial values of variables on each device are the same in distributed scenarios.

## Prototype

```python
class NPUBroadcastGlobalVariablesCallback(BroadcastGlobalVariablesCallbackImpl, keras.callbacks.Callback):
    def __init__(self, root_rank)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| root_rank | Input | Identifies the device whose variables are to be broadcast to other devices. |

## Returns

An object of the  **NPUBroadcastGlobalVariablesCallback**  class

## Example

Before porting

```python
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))
model.fit(data, labels, epochs=10, batch_size=32, callbacks=callbacks)
```

After porting

```python
from npu_bridge.npu_init import *
callbacks = [NPUBroadcastGlobalVariablesCallback(0)]

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))
model.fit(data, labels, epochs=10, batch_size=32, callbacks=callbacks)
```
