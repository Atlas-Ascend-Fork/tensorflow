# get_schedule_aicore_task_early

## Description

Queries whether the AI Core task early start mode is enabled.

## Prototype

```python
def get_schedule_aicore_task_early()
```

## Parameters

None

## Returns

True: the AI Core task early start mode is enabled. False: the mode is disabled.

## Constraints

None

## Example

```python
import tensorflow as tf
import npu_device as npu
# Enable the AI Core task early start mode.
npu.npu_device.set_schedule_aicore_task_early(True)
# Query whether the mode is enabled.
print(npu.npu_device.get_schedule_aicore_task_early())  # True
# Initialize the NPU as the default device.
npu.open().as_default()
```
