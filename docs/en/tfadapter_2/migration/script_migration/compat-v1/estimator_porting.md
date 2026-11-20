# Porting with Estimator

## About Estimator

The  **Estimator**  API is a high-level API of TensorFlow, which can simplify the programming process of machine learning.  **Estimator**  has many advantages, for example, good support for distribution, simplified model creation, and code sharing between model developers.

TensorFlow 2.6 continues the support for this high-level API. To use it in the same way as in TF1, call it using the  **compat.v1**  module as follows:

```python
tf.compat.v1.estimator.Estimator
```

Develop your training script with the  **Estimator**  API of  **compat.v1**  as follows:

1. Data preprocessing: Create an input function  **input_fn**.
2. Model construction: Create a model function  **model_fn**.
3. Run configuration: Instantiate  **Estimator**  and pass the  **RunConfig**  object as the run parameter.
4. Training: Call  **Estimator.train\(\)**  to train your model with a fixed number of steps.

The following guides you through migrating your training script developed with  **Estimator**, which after porting can run on the  AI processor.

## Header File Inclusion

To import NPU-related libraries, add this header file reference in related Python files as follows:

```python
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
```

## Data Preprocessing

The code snippet is ready to use in normal cases. Manual tweaking is required only in the following scenario:

If the original network script relies on  **dataset.batch\(batch_size\)**  to return the dynamic shape, the shape of the last step on the network may be inconsistent with the previous shape because the number of remaining samples in the data flow may be less than the batch size. In this scenario, the dynamic shape compilation process starts. To improve network compilation performance, you are advised to set  **drop_remainder**  to  **True**  to discard the last several samples in the file and ensure that the shape of each step on the network is the same.

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

Note that during inference, if the inference data volume of the last iteration is less than the batch size, you need to pad the inference data with blank data to the batch size. Failure to do so may lead to an assertion in your script that the number of validation results must be equal to the number of validation samples.

```python
 assert num_written_lines == num_actual_predict_examples
```

## Model Building

The code snippet is ready-to-use in normal cases. Manual tweaking is required only in the following scenarios:

- If  **tf.device**  is used in the original network, delete the related code.
- Replace  **gelu**  in the original network with the corresponding CANN API.

    Original TensorFlow code:

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu(x)
    ```

    Code after porting:

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

## Run Configuration Setting

TensorFlow-based run parameters are configured through  **RunConfig**. Modify the configuration as follows:

Original TensorFlow code:

```python
session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)

config=tf.estimator.RunConfig(
  session_config=session_config,
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  ... ...
  )
```

Code after porting:

```python
session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF

npu_config=NPURunConfig(
  session_config=sess_config,
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  ... ...
  )
```

## Estimator Creation

To port  **Estimator**, change its  **config**  parameter to  **npu_config**  and port it to  **NPUEstimator**.

Original TensorFlow code:

```python
mnist_classifier=tf.compat.v1.estimator.Estimator(
  model_fn=cnn_model_fn,
  config=config,
  model_dir="/tmp/mnist_convnet_model")
```

Code after porting:

```python
mnist_classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  config=npu_config,
  model_dir="/tmp/mnist_convnet_model"
  )
```

## Training

When training your model, specify the number of steps. The code snippet is ready-to-use in normal cases.

```python
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=20000,
  hooks=[logging_hook])
```
