# NPUBroadcastGlobalVariablesCallback构造函数

## 功能说明

Keras场景下对变量进行广播，使得在分布式场景下每个device上的变量初始值保持一致。

## 函数原型

```python
class NPUBroadcastGlobalVariablesCallback(BroadcastGlobalVariablesCallbackImpl, keras.callbacks.Callback):
    def __init__(self, root_rank)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| root_rank | 输入 | 标识将哪个device的变量广播到其他的device上。 |

## 返回值

返回NPUBroadcastGlobalVariablesCallback类对象。

## 调用示例

迁移前：

```python
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))
model.fit(data, labels, epochs=10, batch_size=32, callbacks=callbacks)
```

迁移后：

```python
from npu_bridge.npu_init import *
callbacks = [NPUBroadcastGlobalVariablesCallback(0)]

import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000,1))
model.fit(data, labels, epochs=10, batch_size=32, callbacks=callbacks)
```
