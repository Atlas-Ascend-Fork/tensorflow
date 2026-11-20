# keep_dtype_scope

## 功能说明

指定哪些算子保持原有精度，如果原始网络模型中的算子精度在AI处理器上不支持，则系统内部自动采用算子支持的高精度来计算。

## 函数原型

```python
def keep_dtype_scope()
```

## 参数说明

无

## 使用约束

算子精度模式为保持原图精度（即precision_mode指定为must_keep_origin_dtype）时，该接口不生效。

## 返回值

无

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

x = tf.placeholder(tf.float32, (1, 4, 4, 3))
w = tf.Variable(tf.random_normal([3, 3, 3, 1]))

with npu_scope.keep_dtype_scope():
    X = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
```
