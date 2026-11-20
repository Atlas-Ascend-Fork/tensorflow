# disable_autofuse

## 功能说明

TensorFlow网络在CANN平台运行时，会自动根据图结构进行多个算子的自动融合，以获取更优的性能。若开发者不想让某些算子进行自动融合，可通过此接口进行标识。

## 函数原型

```python
def disable_autofuse()
```

## 参数说明

无

## 返回值

无

## 约束说明

“disable_autofuse”接口需要通过with语句调用，仅在对应作用域内的算子不进行自动融合。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

a = tf.placeholder(tf.float32, (5, 1))
b = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32, shape=(5, 1))
add_0 = tf.add(a, b)
mul_0 = tf.multiply(add_0, b)

with disable_autofuse():
    # 此作用域中的算子不参与自动融合
    abs_0 = tf.abs(mul_0)
    div_0 = tf.divide(a, abs_0)

with tf.Session() as sess:
    result = sess.run(div_0, feed_dict={a: [[1], [2], [3], [4], [5]]})
    print(result)
```
