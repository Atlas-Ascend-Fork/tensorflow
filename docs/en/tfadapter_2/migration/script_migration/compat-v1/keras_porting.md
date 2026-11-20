# Porting with Keras

## About Keras

Similar to  **Estimator**,  **Keras**  is another high-level API of TensorFlow. It constructs graphs efficiently, and provides APIs for training, evaluation, validation, and export.

TensorFlow 2.6 continues the support for  **Keras**  APIs. To use it in the same way as in TF1, call it using the  **compat.v1**  module as follows:

```python
tf.compat.v1.Session
```

Develop your training script with the Keras API as follows:

1. Preprocess data.
2. Construct your model.
3. Build your model.
4. Train your model.

> [!CAUTION]NOTICE
> Currently, only training scripts compiled using TensorFlow Keras APIs are supported. Native Keras APIs are not supported.

The following describes how to port the  **Keras**  training scripts for training on the  AI processor.

## Header File Inclusion

To import NPU-related libraries, add this header file reference in related Python files as follows:

```python
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
```

## Porting Configuration

If you are using a Keras training script, the script migrated to the NPU will lose support of certain features such as the dynamic learning rate. Therefore, you are advised not to port Keras scripts to the NPU. To run a Keras script on the NPU, you need to edit the script as follows:

To train your model on the  AI processor, create a TensorFlow session, register Keras, and add related configurations. When the training ends, close the session.

```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from npu_device.compat.v1.npu_init import *

sess_config = tf.compat.v1.ConfigProto()
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.compat.v1.Session(config=sess_config)
K.set_session(sess)

# Preprocess data.
# Construct your model.
# Build your model.
# Train your model.

sess.close()
```
