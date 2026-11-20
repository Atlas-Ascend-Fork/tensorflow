# set_device_sat_mode

## Description

Sets the process-level overflow mode for floating-point compute. Two overflow modes are supported: saturation mode and Inf/NaN mode.

- Saturation mode: When overflow occurs during compute, the compute result is saturated as the floating-point extremum \(+-MAX\).
- INF/NaN mode: Complies with IEEE 754 and outputs the INF/NaN compute result based on the definition.

For the Atlas training products, the default (and the only supported) mode is saturation mode.

For other series products, the overflow/underflow mode can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable

## Prototype

```python
def set_device_sat_mode(mode)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| mode | Input | Specified overflow mode.<br><br>  - 0: saturation mode.<br>  - 1: INF/NaN mode.<br><br>For the Atlas training products, the default (and the only supported) value is 0.<br>For other series products, use the default value 1.|

## Returns

None

## Constraints

None

## Example

```python
import tensorflow as tf
import npu_device as npu
# Initialize the NPU as the default device.
npu.open().as_default()
# For the Atlas A2 products, the following API is called to set the overflow mode during network execution:
npu.npu_device.set_device_sat_mode(1)
```
