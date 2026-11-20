# KerasDistributeOptimizer构造函数

## 功能说明

KerasDistributeOptimizer类的构造函数，用于包装用户使用tf.keras构造的脚本中的单机训练优化器，构造NPU分布式训练优化器。

## 函数原型

```python
class KerasDistributeOptimizer(optimizer_v2.OptimizerV2):
    def __init__(self, optimizer, name="NpuKerasOptimizer", **kwargs)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| optimizer | 输入 | 用于梯度计算和更新权重的单机版训练优化器。 |
| name | 输入 | 优化器名称。 |

## 返回值

返回KerasDistributeOptimizer类对象。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

model=xxx
model.compile(loss='mean_squared_error', optimizer=KerasDistributeOptimizer(tf.keras.optimizers.SGD()))
```
