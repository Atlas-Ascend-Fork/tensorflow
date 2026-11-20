# Using Single-Server Single-Device Scripts

## Porting Principle

The principle for adapting TF Adapter to TensorFlow 1.15 is as follows:

The  **config**  extension mechanism in TensorFlow is utilized to pass NPU-related function configurations downstream through  **config**. Then, TensorFlow's optimizer registration mechanism is employed to register an optimizer on NPUs for graph processing based on configurations. Finally, the processed graph is delivered to the CANN platform for execution.

The main point of manual porting is to add configurations on NPUs and transfer the extended NPU configurations to  **config**  of  **tf.Session**. In TensorFlow 1.15, the common  **Estimator**,  **sess.run**, and  **Keras**  scripts are implemented by calling  **sess.run**, making the porting approaches for these three scripts fundamentally identical.

## Porting with Estimator

If the original TensorFlow network is constructed based on the  **Estimator**  API, see this section to understand the manual porting process.

### About Estimator

The **Estimator** API is a high-level API in TensorFlow that greatly simplifies machine learning programming. **Estimator** has many advantages, for example, good support for distribution, simplified model creation, and code sharing between model developers.

Develop your training script with the **Estimator** API as follows:

1. Create an input function  **input_fn**  during data preprocessing.
2. Create a model function  **model_fn**  during model construction.
3. Run configuration: Instantiate  **Estimator**  and pass the  **RunConfig**  object as the run parameter.
4. Call  **Estimator.train\(\)**  to train your model with a fixed number of steps that you set.

To perform training on the NPU, the following guides you to port your training script developed with  **Estimator**.

### Header File Inclusion

To import NPU-related libraries, add this header file reference in related Python files as follows:

```python
from npu_bridge.npu_init import *
```

> [!NOTE]NOTE
> After the preceding header file is imported, the training script is executed on the NPU by default.

### Data Preprocessing

The code snippet is ready to use in normal cases. Manual tweaking is required only in the following scenarios:

If the original network script relies on  **dataset.batch\(batch_size\)**  to return the dynamic shape, the shape of the last step on the network may be inconsistent with the previous shape because the number of remaining samples in the data flow may be less than the batch size. In this scenario, the dynamic shape compilation process starts. To improve network compilation performance, you are advised to set  **drop_remainder**  to  **True**  to discard the last several samples in the file and ensure that the shape of each step on the network is the same.

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

Note that during inference, if the inference data volume of the last iteration is less than  **batch_size**, you need to pad blank data to reach  **batch_size**. This is because some scripts use an assertion to verify that the number of outputs is consistent with the number of validation samples.

```python
 assert num_written_lines == num_actual_predict_examples
```

### Model Construction

The code snippet is ready to use in normal cases. Manual tweaking is required only in the following scenarios:

- Replace  **dropout**  in the original network with the corresponding CANN API for better performance. You must also pay attention to the impact on the network accuracy.
  - If  **tf.nn.dropout**  exists, modify it as follows:

     ```python
     layers = npu_ops.dropout()
     ```

  - If  **tf.layers.dropout**,  **tf.layers.Dropout**,  **tf.keras.layers.Dropout**,  **tf.keras.layers.SpatialDropout1D**,  **tf.keras.layers.SpatialDropout2D**, or  **tf.keras.layers.SpatialDropout3D**  exists, add the following header file reference:

     ```python
     from npu_bridge.estimator.npu import npu_convert_dropout
      ```

- Replace  **gelu**  in the original network with the corresponding CANN API to achieve optimal performance.

    Original TensorFlow code:

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    Code after porting:

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

### Run Configuration Setting

TensorFlow uses  **RunConfig**  to configure the run parameters. You need to port  **RunConfig**  to  **NPURunConfig**. The  **NPURunConfig**  class inherits from the  **RunConfig**  class. Therefore, you can directly modify a script during porting according to the following example with most parameters unchanged.

Original TensorFlow code:

```python
config=tf.estimator.RunConfig(
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
```

Code after porting:

```python
npu_config=NPURunConfig(
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  # If tf.device code is used on the original network, add the session configuration allow_soft_placement=True to allow TensorFlow to automatically allocate devices.
  session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  )
```

However, some are not allowed by  **NPURunConfig**, including  **train_distribute**,  **device_fn**,  **protocol**,  **eval_distribute**, and  **experimental_distribute**. Remove them if they are used in the original script.

If  **tf.device**  code is used on the original network, add the session configuration  **allow_soft_placement=True**  to allow TensorFlow to automatically allocate devices.

In addition, some parameters, such as  **iterations_per_loop**  and  **precision_mode**, are added to  **NPURunConfig**  to improve training performance and accuracy. For details about the parameters, see  [NPURunConfig Constructor](../../apiref/npu_config/npurunconfig_constructor/README.md).

### Creating an Estimator Object

You only need to port an  **Estimator**  object of TensorFlow to  **NPUEstimator**  that inherits from the  **Estimator**  class. Change the API by referring to the following example during porting and keep the parameters unchanged.

Original TensorFlow code:

```python
mnist_classifier=tf.estimator.Estimator(
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

### Training

When training your model, specify only the inputs. The code snippet is ready to use in normal cases.

```python
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=20000,
  hooks=[logging_hook])
```

## Porting with sess.run

If the original TensorFlow network is constructed based on the **sess.run** API, see this section to understand the manual porting process.

### About sess.run

As a low-level API of TensorFlow,  **sess.run**  appears more flexible than  **Estimator**. However, using it for model implementation could be complex.

Develop your training script with the  **sess.run**  API as follows:

1. Preprocess data.
2. Construct a model, calculate the loss, and update the gradient.
3. Create a session and initialize resources.
4. Start training.

To perform training on the  AI processor, the following guides you to port your training script developed with  **sess.run**.

### Header File Inclusion

To import NPU-related libraries, add this header file reference in related Python files as follows:

```python
from npu_bridge.npu_init import *
```

> [!NOTE]NOTE
> After the preceding header file is imported, the training script is executed on the NPU by default.

### Data Preprocessing

The code snippet is ready to use in normal cases. Manual tweaking is required only in the following scenarios:

If the original network script relies on  **dataset.batch\(batch_size\)**  to return the dynamic shape, the shape of the last step on the network may be inconsistent with the previous shape because the number of remaining samples in the data flow may be less than the batch size. In this scenario, the dynamic shape compilation process starts. To improve network compilation performance, you are advised to set  **drop_remainder**  to  **True**  to discard the last several samples in the file and ensure that the shape of each step on the network is the same.

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

Note that during inference, if the inference data volume of the last iteration is less than  **batch_size**, you need to pad blank data to reach  **batch_size**. This is because some scripts use an assertion to verify that the number of outputs is consistent with the number of validation samples.

```python
 assert num_written_lines == num_actual_predict_examples
```

### Model Construction, Loss Calculation, and Gradient Update

The code snippet is ready to use in normal cases. Manual tweaking is required only in the following scenarios:

- Replace  **dropout**  in the original network with the corresponding CANN API for better performance. You must also pay attention to the impact on the network accuracy.
  - If  **tf.nn.dropout**  exists, modify it as follows:

    ```python
    layers = npu_ops.dropout()
    ```

  - If  **tf.layers.dropout**,  **tf.layers.Dropout**,  **tf.keras.layers.Dropout**,  **tf.keras.layers.SpatialDropout1D**,  **tf.keras.layers.SpatialDropout2D**, or  **tf.keras.layers.SpatialDropout3D**  exists, add the following header file reference:

     ```python
    from npu_bridge.estimator.npu import npu_convert_dropout
    ```

- Replace  **gelu**  in the original network with the corresponding CANN API to achieve optimal performance.

    Original TensorFlow code:

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    Code after porting:

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

### Session Creation and Resource Initialization

When running your training script on the  AI processor  by using  **sess.run**, note the following configurations:

- The following configuration option is deactivated by default and should remain deactivated:

    rewrite_options.disable_model_pruning

- The following configuration options are activated by default and should remain activated:
  - rewrite_options.function_optimization
  - rewrite_options.constant_folding
  - rewrite_options.shape_optimization
  - rewrite_options.arithmetic_optimization
  - rewrite_options.loop_optimization
  - rewrite_options.dependency_optimization
  - rewrite_options.layout_optimizer

- The following configuration options are enabled by default and should be disabled explicitly:
  - rewrite_options.remapping
  - rewrite_options.memory_optimization

- If  **tf.device**  code is used on the original network, add the session configuration  **allow_soft_placement=True**  to allow TensorFlow to automatically allocate devices.

Original TensorFlow code:

```python
# Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# Obtain the batch data.
next_batch=iterator.get_next()

# Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

# Initialize the variables.
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

# Get the number of training/validation steps per epoch.
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

Code after porting:

```python
# Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# Obtain the batch data.
next_batch=iterator.get_next()

# Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

# Initialize the variables.
init=tf.global_variables_initializer()

# Add allow_soft_placement=True for the session configurations to allow TensorFlow to automatically allocate devices.
config = tf.ConfigProto(allow_soft_placement=True)
# Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# Explicitly disable the remapping and memory_optimization functions of TensorFlow to avoid conflicts with the functions of the NPU.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Explicitly disable the function.
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # Explicitly disable the function.
sess = tf.Session(config=config)
sess.run(init)

# Get the number of training/validation steps per epoch.
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

The Ascend platform supports all native functions of  **tf.Session**.

It also allows you to enable functions such as automatic mixed precision. For details, see  [Session Configuration](../../apiref/session_config/README.md).

### Training

The code snippet is ready to use. See the following example.

```python
# Start epochs.
for epoch in range(num_epochs):
  ##Initialize iterator with the training dataset
  sess.run(training_init_op)
  for step in range(train_batches_per_epoch):
    #get next batch of data
    img_batch,label_batch=sess.run(next_batch)
    #run the training op
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch, y_:label_batch, is_training:True})
```

However, if you define the session object as a class member instead of using the  **with**  statement for session creation, you must explicitly call  **sess.close\(\)**  in the ported script.

```python
sess = tf.Session(config=config)
sess.run(...)
sess.close()
```

That is because the GEOP destructor function is called in the  **close**  method of  **tf.Session**. If you use the  **with**  statement that calls  **__exit__**  to close the session automatically, there is no need to call  **sess.close\(\)**.

```python
with tf.Session(config=config) as sess:
    sess.run(...)
```

In other cases, for example, taking a session object as a user-defined class member, you should explicitly call  **sess.close\(\)**  to exit the session.

## Porting with Keras

If the original TensorFlow network is constructed based on the  **Keras**  API, see this section to understand the manual porting process.

### About Keras

Similar to  **Estimator**,  **Keras**  is another high-level API of TensorFlow. It constructs graphs efficiently, and provides APIs for training, evaluation, validation, and export. Develop your training script with the  **Keras**  API as follows:

1. Preprocess data.
2. Construct your model.
3. Build your model.
4. Train your model.

> [!CAUTION]NOTICE
> Currently, only training scripts compiled using TensorFlow  **Keras**  APIs are supported. Native  **Keras**  APIs are not supported.

The following describes how to port the  **Keras**  training scripts for training on the  AI processor:

### Header File Inclusion

To import NPU-related libraries, add this header file reference in related Python files as follows:

```python
from npu_bridge.npu_init import *
```

> [!NOTE]NOTE
> After the preceding header file is imported, the training script is executed on the NPU by default.

### Porting Configuration

To train your model on the  AI processor, create a TensorFlow session, register  **Keras**, and add related configurations. When the training ends, close the session.

```python
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from npu_bridge.npu_init import *

# Add allow_soft_placement=True for the session configurations to allow TensorFlow to automatically allocate devices.
sess_config = tf.ConfigProto(allow_soft_placement=True)
# Add an NPU optimizer named NpuOptimizer. During network compilation, the NPU traverses only the session configurations under NpuOptimizer.
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# Explicitly disable the remapping and memory_optimization functions of TensorFlow to avoid conflicts with the functions of the NPU.
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

# Preprocess data...
# Construct a model...
# Build the model...
# Train the model...

sess.close()
```

With the ported training script, the number of training iterations performed by the Ascend AI Processor is fixed to 1 in every  **sess.run**  call. To reduce data transfers between the host and device and shorten the training duration, use the  **model_to_npu_estimator**  API to convert the model constructed by  **Keras**  to an  **NPUEstimator**  object and set  **iterations_per_loop**  \(number of training iterations performed by the Ascend AI Processor every  **sess.run**  call\) in  **NPURunConfig**  to a value that suits your needs. For details, see  [Enabling Iteration Offload In Keras Mode](../performance_tuning/iteration_offload.md#enabling-iteration-offload-in-keras-mode).
