# set_schedule_aicore_task_early

## Description

Sets whether to enable the AI Core task early start mode. When this mode is enabled, AI Core tasks can be scheduled in advance to reduce the scheduling latency and improve the execution efficiency.

## Prototype

```python
def set_schedule_aicore_task_early(value)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| value | Input | Whether to enable the AI Core task early start mode.<br><br>  - True: enables the mode.<br>  - False: disables the mode. |

## Returns

0: success. Other values: failure.

## Constraints

This API needs to be called before the network script is executed.

## Example

```python
import tensorflow as tf
import npu_device as npu
# Enable the AI Core task early start mode.
npu.npu_device.set_schedule_aicore_task_early(True)
# Initialize the NPU as the default device.
npu.open().as_default()
```
