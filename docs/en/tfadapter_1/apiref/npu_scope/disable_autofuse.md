# disable_autofuse

## Description

When a TensorFlow network runs on the CANN platform, multiple operators are automatically fused based on the graph structure to obtain better performance. If you do not want to perform automatic fusion on some operators, you can use this API to mark them.

## Prototype

```python
def disable_autofuse()
```

## Parameters

None

## Returns

None

## Restrictions

The  **disable_autofuse**  API needs to be called using the  **with**  statement. Only operators in the corresponding scope are not automatically fused.

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

a = tf.placeholder(tf.float32, (5, 1))
b = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32, shape=(5, 1))
add_0 = tf.add(a, b)
mul_0 = tf.multiply(add_0, b)

with disable_autofuse():
    # Operators in this scope are not automatically fused.
    abs_0 = tf.abs(mul_0)
    div_0 = tf.divide(a, abs_0)

with tf.Session() as sess:
    result = sess.run(div_0, feed_dict={a: [[1], [2], [3], [4], [5]]})
    print(result)
```
