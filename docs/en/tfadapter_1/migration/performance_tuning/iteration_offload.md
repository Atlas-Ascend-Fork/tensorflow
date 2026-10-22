# Iteration Offload

## Restrictions

**iterations_per_loop**  is the number of iterations per training loop performed on the device per  **sess.run**  call. Training is performed according to the specified number of iterations per loop \(**iterations_per_loop**\) on the device and then the result is returned to the host. This parameter can save unnecessary interactions between the host and device and reduce the training time.

**iterations_per_loop**  defaults to  **1**. You can enable the iteration offload feature by setting this parameter to a value greater than  **1**. Note the following restrictions when using this feature:

- The training script must read data in TensorFlow's dataset mode instead of the one-shot iterator for preprocessing initialization. For example, use the  **tf.data.make_initializable_iterator\(\)**  iterator. Reading data using the Dataset method is a prerequisite for data preprocessing offload and training iteration loop offload. For detailed usage of Datasets, see the  [TensorFlow official website](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset).
- Enable data preprocessing offload by setting  **enable_data_pre_proc**  to  **True**. This will generate the GetNext operator that runs on the device, thereby enabling training iteration loop offload.
  - The following is an example of enabling data preprocessing offload in  **sess.run**:

    ```python
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    ```

  - The following is an example of enabling data preprocessing offload in  **NPURunConfig**:

    ```python
    config = NPURunConfig(enable_data_pre_proc=True)
    ```

- The total number of training iterations must be evenly divisible by  **iterations_per_loop**.
- When saving checkpoint data in iteration offload mode, set  **save_checkpoints_steps**  to a positive integer multiple of  **iterations_per_loop**, so that checkpoints can be saved in strict accordance with  **save_checkpoints_steps**. If the value of  **iterations_per_loop**  is greater than 1, data may not be saved as defined by  **save_summary_steps**  and  **log_step_count_steps**. In this case, follow  [Log and Summary Operators](../others/Log-Summary.md)  to resolve this problem.
- In mixed computing mode \(with  **mix_compile_mode**  set to  **True**\), iteration offload must not be enabled. That is,  **iterations_per_loop**  must be set to  **1**.
- During network development, you are advised to set  **iterations_per_loop**  to  **1**  to facilitate log printing every iteration. After the network is set up correctly, you can set the  **iterations_per_loop**  parameter to shorten the training time.

## Enabling Iteration Offload in Estimator Mode

### Automated porting

1. Search for **npu_run_config_init**  in the ported script and find the run configuration parameter \(such as  **run_config**  in the example\). Pass the **session_config** parameter to the run configuration function, and add  **iterations_per_loop** to the  **session_config** parameter.

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["enable_data_pre_proc"].b = True # The GetNext operator offload is a prerequisite for iteration offload.
    custom_op.parameter_map["iterations_per_loop"].i = 10

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,       # Add the session_config configuration to the run configuration parameter.
        save_checkpoints_secs=60*60*24)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
    ```

2. Add **SetIterationsVarHook**.

    ```python
    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)
    train_hooks.append(SetIterationsVarHook(10))
    ```

3. Add **IterationOp** to **train_op**.

    ```python
    train_op = opt.apply_gradients( grad_var_list, global_step=global_step )
    train_op = tf.group(train_op, name="IterationOp")   # Set name to the operator that receives the gradient update.
    ```

### Manual porting

In  **Estimator**  mode, configure  **iterations_per_loop**  in  **NPURunConfig**  as follows.

```python
from npu_bridge.npu_init import *

session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(session_config=session_config, iterations_per_loop=10)
```

In addition, enable the GetNext operator offload, which is a prerequisite for iteration offload. In  **Estimator**  mode, the GetNext operator offload is enabled by default, that is,  **enable_data_pre_proc**  is set to  **True**  by default. Retain the default setting.

### Checking Whether iterations_per_loop Takes Effect

After iteration offload is enabled, you can check whether the keyword "Insert op success" exists in the  **INFO**  log on the host to determine whether  **iterations_per_loop**  takes effect.

You can run the following command to set the log level on the host to  **INFO**. The default output path of  **INFO**  logs is  **$HOME/ascend/log/run/plog/**.

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

## Enabling Iteration Offload in sess.run Mode

### Automated porting

In  **sess.run**  mode, configure the  **iterations_per_loop**  parameter by using  [set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)  and change the  **sess.run()**  call count according to the following formula:  **sess.run()**  call count = Original  **sess.run()**  call count/**iterations_per_loop**.

Find  **npu_config_proto**  in the script, configure  **iterations_per_loop**  in the session configuration, and set the number of iterations per loop using  [set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md), as shown in the following code in bold:

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# Set the model.
# Set the learning rate.
learning_rate = 0.01
# Set the number of training iterations.
training_epochs = 10
# Set the batch size.
batch_size = 100
# Set the number of iterations to display the loss.
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set the model parameters.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Build the model.
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the loss function: cross entropy.
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Update gradients.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables.
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b = False  # Disable mixed computing (disabled by default).
custom_op.parameter_map["enable_data_pre_proc"].b = True # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10 # Determine whether the training iteration is offloaded. Must be equal to iterations_per_loop set in set_iteration_per_loop.
config = npu_config_proto(config_proto=config)

# Train the model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    train_op = util.set_iteration_per_loop(sess, optimizer, 10)  # sess indicates the created TensorFlow session, optimizer indicates the defined gradient update operation, and 10 indicates the number of training iterations on the device.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

The  [set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)  API involves graph modification. If a graph cannot be modified \(for example, the graph is frozen or a session is created using  **tf.train.Supervisor**\), you cannot use the  **set_iteration_per_loop**  API to set the number of iterations per loop. Instead, you can use  [create_iteration_per_loop_var](../../apiref/npu_util/create_iteration_per_loop_var.md)  and  [load_iteration_per_loop_var](../../apiref/npu_util/load_iteration_per_loop_var.md)  to set the number of iterations per loop, as shown in the following code in bold:

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# Set the model.
# Set the learning rate.
learning_rate = 0.01
# Set the number of training iterations.
training_epochs = 10
# Set the batch size.
batch_size = 100
# Set the number of iterations to display the loss.
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set the model parameters.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Build the model.
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the loss function: cross entropy.
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Update gradients.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables.
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b = False  # Disable mixed computing (disabled by default).
custom_op.parameter_map["enable_data_pre_proc"].b = True # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10  # Used for functional validation. Must be equal to iterations_per_loop set in load_iteration_per_loop_var.
config = npu_config_proto(config_proto=config)

# Train the model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    iteration = util.IterationPerLoop()   # Define the IterationPerLoop class object.
    train_op = iteration.create_iteration_per_loop_var(optimizer)    # optimizer indicates the defined operation for updating gradients.
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  # Freeze the graph.
    iteration.load_iteration_per_loop_var(sess, 10)  # Set the number of iterations per loop. sess indicates the created TensorFlow session, and 10 indicates the number of training iterations on the device.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

### Manual porting

In  **sess.run**  mode, configure the  **iterations_per_loop**  parameter by using  [set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)  and change the  **sess.run()**  call count according to the following formula:  **sess.run()**  call count = Original  **sess.run()**  call count/**iterations_per_loop**. See the following code in bold:

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# Set the model.
# Set the learning rate.
learning_rate = 0.01
# Set the number of training iterations.
training_epochs = 10
# Set the batch size.
batch_size = 100
# Set the number of iterations to display the loss.
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set the model parameters.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Build the model.
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the loss function: cross entropy.
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Update gradients.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables.
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True # Perform training on the Ascend AI Processor.
custom_op.parameter_map["mix_compile_mode"].b = False  # Disable mixed computing (disabled by default).
custom_op.parameter_map["enable_data_pre_proc"].b = True # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10 # Determine whether the training iteration is offloaded. Must be equal to iterations_per_loop set in set_iteration_per_loop.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

# Train the model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    train_op = util.set_iteration_per_loop(sess, optimizer, 10) # sess indicates the created TensorFlow session, optimizer indicates the defined gradient update operation, and 10 indicates the number of training iterations on the device.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

The  [set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)  API involves graph modification. If a graph cannot be modified \(for example, the graph is frozen or a session is created using  **tf.train.Supervisor**\), you cannot use the  **set_iteration_per_loop**  API to set the number of iterations per loop. Instead, you can use  [create_iteration_per_loop_var](../../apiref/npu_util/create_iteration_per_loop_var.md)  and  [load_iteration_per_loop_var](../../apiref/npu_util/load_iteration_per_loop_var.md)  to set the number of iterations per loop, as shown in the following code in bold:

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# Set the model.
# Set the learning rate.
learning_rate = 0.01
# Set the number of training iterations.
training_epochs = 10
# Set the batch size.
batch_size = 100
# Set the number of iterations to display the loss.
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# Set the model parameters.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Build the model.
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# Define the loss function: cross entropy.
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Update gradients.
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize all variables.
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # Perform training on the Ascend AI Processor.
custom_op.parameter_map["mix_compile_mode"].b = False  # Disable mixed computing (disabled by default).
custom_op.parameter_map["enable_data_pre_proc"].b = True  # If the GetNext operator exists on the network, offload it. GetNext operator offload is a prerequisite for iteration offload.
custom_op.parameter_map["iterations_per_loop"].i = 10 # Determine whether the training iteration is offloaded. Must be equal to iterations_per_loop set in load_iteration_per_loop_var.
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
# Train the model.
with tf.Session(config=config) as sess:
    sess.run(init)
    # Set the number of iterations per loop to 10 in sess.run mode.
    iteration = util.IterationPerLoop()
    train_op = iteration.create_iteration_per_loop_var(optimizer) # optimizer indicates the defined operation for updating gradients.
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  # Freeze the graph.
    iteration.load_iteration_per_loop_var(sess, 10)  # Set the number of iterations per loop. sess indicates the created TensorFlow session, and 10 indicates the number of training iterations on the device.

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

> [!NOTE]NOTE
>After the number of iterations per loop is changed, you are advised to adjust other loop parameters based on the actual situation, such as the time required for obtaining a single step and updating the number of iterations.

### Checking Whether iterations_per_loop Takes Effect

After iteration offload is enabled, you can check whether the keyword "Insert op success" exists in the  **INFO**  log on the host to determine whether  **iterations_per_loop**  takes effect.

You can run the following command to set the log level on the host to  **INFO**. The default output path of  **INFO**  logs is  **$HOME/ascend/log/run/plog/**.

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

## Enabling Iteration Offload In Keras Mode

On the Ascend platform, you can directly use the native  **Keras**  API for training. However, the number of iterations per training loop on the  AI processor  is fixed at 1 in each  **sess.run**  call. To reduce the number of interactions between the host and devices and shorten the training duration, use the  **model_to_npu_estimator**  API to convert the model constructed by using  **Keras**  into an  **NPUEstimator**  object. Besides, specify the number of iterations per training loop on the  AI processor  per  **sess.run\(\)**  call by using the  **iterations_per_loop**  parameter in  **NPURunConfig**.

Original TensorFlow code:

```python
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications import ResNet50

# This returns a tensor
inputs = Input(shape=(224, 224, 3))

# This creates a model that includes
# the Input layer and three Dense layers
keras_model = ResNet50(input_tensor=inputs, weights=None, include_top=True)
keras_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

keras_model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10)
```

Code after porting:

```python
from npu_bridge.npu_init import *

run_config = NPURunConfig(save_checkpoints_steps=2,
                          model_dir=model_path,
                          iterations_per_loop=10)
# Convert the model constructed by using Keras to an NPUEstimator object.
est_resnet = keras_to_npu.model_to_npu_estimator(keras_model=keras_model, config=run_config)
# Perform training.
est_resnet.train(input_fn=lambda: input_fn(), max_steps=1000)
```

In addition, you need to port the  **Keras**  data preprocessing part to  **input_fn**  in  **NPUEstimator**. The following is an example. In the following example,  **Keras**  reads image data from the folder, automatically labels the data, performs data augmentation operations such as data resize, normalization, and horizontal flip, and finally outputs the data. In  **Estimator**  mode, data is preprocessed in the same way as reading data from the file list. The difference is that the file name list needs to be read in advance and each image needs to be labeled to output the label list. The data is output after the same data augmentation operations such as normalization, resize, and horizontal flip.

Original TensorFlow code:

```python
# Keras reads images from the folder.
train_datagen = ImageDataGenerator(rescale=1./255,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('data/',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='sparse')
```

Code after porting:

```python
# The function is used to read the image files corresponding to the file names and resize the image files to a unified size.
def _parse_function(filename, label):
  image = tf.read_file(filename)
  image = tf.image.decode_image(image)
  image = image / 255.0
  image = tf.image.resize_images(image, [224, 224])
  image = tf.image.random_flip_left_right(image)
  return image, label

def input_fn():
    # List of image files. The image list needs to be generated by yourself.
    filenames = tf.constant(["/data/image1.jpg", "/data/image2.jpg", ...])
    # label[i] is the label of the filenames[i] image. The label list needs to be generated by yourself.
    labels = tf.constant([0, 5, ...])
    # Now an element in the dataset is (filename, label).
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).repeat(10)
    # Now an element in the dataset is (image_resized, label).
    dataset = dataset.map(_parse_function)
    # Now an element in the dataset is (image_resized_batch, label_batch).
    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    return dataset
```

Note that the callback function of Keras cannot be used after being converted to an  **NPUEstimator**  object.

**Checking Whether iterations_per_loop Takes Effect**

After iteration offload is enabled, you can check whether the keyword "Insert op success" exists in the  **INFO**  log on the host to determine whether  **iterations_per_loop**  takes effect.

You can run the following command to set the log level on the host to  **INFO**. The default output path of  **INFO**  logs is  **$HOME/ascend/log/run/plog/**.

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```
