# npu.ops.gelu

## 功能说明

计算高斯误差线性单元（GELU）激活函数。将输入Tensor乘以1个P\(X <= x\)，其中P\(X\) ~ N\(0, 1\)。

## 函数原型

```python
npu.ops.gelu(x)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 输入Tensor，float类型。 |

## 返回值

tensor：对输入x执行完GELU操作之后的输出tensor。数据类型和输入相同。

## 调用示例

```python
import npu_device as npu
output = npu.ops.gelu(x)
```
