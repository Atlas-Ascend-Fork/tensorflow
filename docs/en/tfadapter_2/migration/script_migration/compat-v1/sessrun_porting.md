# Porting with sess.run

## About sess.run

As a low-level API of TensorFlow,  **sess.run**  appears more flexible than  **Estimator**. On the flip side, using it for model implementation could be complex.

This API has been deprecated in TensorFlow 2.6. To use it in TensorFlow 2.6, call it using the  **compat.v1**  module as follows:

```python
tf.compat.v1.Session.run
```

Develop your training script with the  **sess.run**  API as follows:

1. Preprocess data.
2. Construct a model, calculate the loss, and update the gradient.
3. Create a session and initialize resources.
4. Start training.

The following guides you through migrating your training script developed with  **sess.run**, which after porting can run on the  AI processor.

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

## Model Construction, Loss Computation, and Gradient Update

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

## Session Creation and Resource Initialization

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

- The following configuration option is enabled by default and should be disabled explicitly:
  - rewrite_options.remapping
  - rewrite_options.memory_optimization

Original TensorFlow code:

```python
# Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

# Obtain the batch data.
next_batch=iterator.get_next()

# Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

# Initialize the variables.
init=tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(init)

# Get the number of training/validation steps per epoch.
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

Code after porting:

```python
# Construct an iterator.
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

# Obtain the batch data.
next_batch=iterator.get_next()

# Initialize the iterator.
training_init_op=iterator.make_initializer(train_dataset)

# Initialize the variables.
init=tf.compat.v1.global_variables_initializer()

# Create a session.
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # Must be disabled explicitly.
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # Must be disabled explicitly.
sess = tf.compat.v1.Session(config=config)
sess.run(init)

# Get the number of training/validation steps per epoch.
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

The NPU supports all native functions of  **tf.compat.v1.Session**.

It also allows you to enable functions such as automatic mixed precision. For details, see the corresponding API description.

## Training

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
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch,y_:label_batch,is_training:True})
```

However, you need an explicit call to  **sess.close\(\)**  in your ported script if you define a session object as a class member instead of creating a session with a  **with**  block.

```python
sess = tf.compat.v1.Session(config=config)
sess.run(...)
sess.close()
```

That is because the destructor of GEOP is called in the  **close**  method of  **tf.compat.v1.session**. If you use a  **with**  block that calls  ****exit****  to close the session automatically, there is no need to call  **sess.close\(\)**.

```python
with tf.compat.v1.Session(config=config) as sess:
    sess.run(...)
```

In other cases, for example, taking a session object as a user-defined class member, you should explicitly call  **sess.close\(\)**  to exit the session.
