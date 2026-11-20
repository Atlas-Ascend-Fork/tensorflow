# Mixed Computing

Mixed computing is provided to facilitate flexibility and scalability in a scenario where the computational graph has unsupported operators \(such as py_func\). Unsupported operators are executed within the frontend framework on the host.

## Overview

For the  AI processor, the full offload mode is used by default, that is, all compute operators are offloaded to the device. As a supplement to the full offload mode, mixed computing allows certain operators \(such as resource operators\) to be executed online within the frontend framework, improving the  AI processor's adaptability to TensorFlow.

## Principles

In mixed computing scenarios, after identifying offloadable operators, TF Adapter partitions the entire graph into multiple GEOPs. Data transfer for unoffloadable operators is performed via  **memcpy**.

![](../figures/hybrid_compute_principle.png)

## Precautions

- In mixed computing mode,  [iteration offload](../performance_tuning/iteration_offload.md#enabling-iteration-offload-in-keras-mode)  is not supported. That is,  **iterations_per_loop**  must retain the default value  **1**.
- In addition to the operators that are not offloaded by default, you can also configure additional operators that are not offloaded by using  **without_npu_compile_scope**.
- The FusedBatchNormV3 operator was introduced in 2019. Its fifth output is a CUDA-related optimized output, which is not supported on the  AI processor  in mixed computing mode. If  **tf.layers.batch_normalization**  is used in your training script, you can use  **with compat.forward_compatibility_horizon\(2019, 5, 1\):**  to skip this operator.

## In Estimator Mode

In  **Estimator**  mode, use  **mix_compile_mode**  of  **NPURunConfig**  to enable the mixed computing function.

```python
from npu_bridge.npu_init import *

session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(session_config=session_config, mix_compile_mode=True, iterations_per_loop=1)
```

## In sess.run Mode

In  **sess.run**  mode, set the session configuration option  **mix_compile_mode**  to enable mixed computing.

```python
import tensorflow as tf
from npu_bridge.npu_init import *
config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b =  True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
```

## In Keras Mode

The configuration method is similar to that in  **sess.run**  mode.

## Operator Retaining

With mixed computing enabled, operators that cannot be offloaded are retained on the host. You can also specify the operators that are not to be offloaded by using  **without_npu_compile_scope**.

```python
import tensorflow as tf
from npu_bridge.npu_init import *
X = tf.random_normal([2,])
Y = tf.random_normal([2,])

with npu_scope.without_npu_compile_scope():
  pred = tf.add(tf.multiply(X, 1.), 0.) # Specify tf.add and tf.multiply as operators not offloaded.
cost = tf.reduce_sum(tf.abs(pred-Y))

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b =  True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
  print(sess.run(cost))
```
