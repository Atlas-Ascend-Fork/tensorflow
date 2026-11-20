# npu.ops.gelu

## Description

Computes the GELU activation function. Each input tensor is multiplied by one P\(X <= x\), where P\(X\) follows N\(0, 1\).

## Prototype

```python
npu.ops.gelu(x)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| x | Input | Input tensor of type float. |

## Returns

Result tensor after the GELU operation is performed on input **x**. The data type is the same as the input.

## Example

```python
import npu_device as npu
output = npu.ops.gelu(x)
```
