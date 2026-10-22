# 训练迭代循环下沉

## 使用约束

iterations_per_loop是针对一次session.run调用，在Device侧执行训练迭代的次数。Device侧会运行iterations_per_loop指定的迭代次数，然后再返回到Host侧，该参数可以减少Host与Device间的交互次数，缩短训练时长。

iterations_per_loop默认为1，配置该参数大于1即可使能训练迭代循环下沉的特性，但使用该特性时需要注意以下约束：

- 要求训练脚本必须使用TensorFlow的Dataset方式读数据，并且不使用one shot iterator进行预处理初始化，例如使用tf.data.make_initializable_iterator\(\)迭代器。使用Dataset方式读取数据是数据预处理下沉与训练迭代循环下沉的前提条件，关于Datasets的详细使用方法可参见[TensorFlow官网](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/data/Dataset)。
- 要求开启数据预处理下沉，即enable_data_pre_proc开关配置为True，此时才会生成在Device侧执行的GetNext算子，训练迭代循环下沉才会生效。
  - sess.run开启数据预处理下沉配置示例：

    ```python
    custom_op.parameter_map["enable_data_pre_proc"].b = True
    ```

  - NPURunConfig开启数据预处理下沉配置示例：

    ```python
    config = NPURunConfig(enable_data_pre_proc=True)
    ```

- 要求训练迭代总次数必须为iterations_per_loop的整数倍。
- 训练迭代循环下沉场景下保存checkpoint数据时，要求save_checkpoints_steps必须大于或等于iterations_per_loop，且是iterations_per_loop的整数倍，否则不会按照save_checkpoints_steps配置的值保存数据。且iterations_per_loop\>1时，可能无法按照save_summary_steps和log_step_count_steps配置的值保存信息，请参考[Log/Summary](../others/Log-Summary.md)实现信息保存。
- 混合计算模式（mix_compile_mode为True）下，不能开启训练迭代循环下沉，即要求iterations_per_loop必须为1。
- 在网络调测阶段建议设置iterations_per_loop为1，方便打印每轮迭代的日志信息。网络调通后可以设置iterations_per_loop参数用于缩短训练时长。

## Estimator模式下使能训练迭代循环下沉

### 自动迁移场景

1. 在迁移后的脚本中查找“npu_run_config_init”，找到运行配置参数（例如示例中的“run_config”），在运行配置函数中传入session_config参数，并在session_config参数中添加“iterations_per_loop”配置。

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    custom_op.parameter_map["enable_data_pre_proc"].b = True # GetNext算子下沉是迭代循环下沉的必要条件
    custom_op.parameter_map["iterations_per_loop"].i = 10

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,       # 运行配置参数中添加session_config配置
        save_checkpoints_secs=60*60*24)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
    ```

2. 增加“SetIterationsVarHook”：

    ```python
    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks,
        model_dir=flags_obj.model_dir,
        batch_size=flags_obj.batch_size)
    train_hooks.append(SetIterationsVarHook(10))
    ```

3. 在train_op中增加“IterationOp”：

    ```python
    train_op = opt.apply_gradients( grad_var_list, global_step=global_step )
    train_op = tf.group(train_op, name="IterationOp")   #该name设置到梯度更新返回的op
    ```

### 手工迁移场景

Estimator模式下，通过NPURunConfig中的iterations_per_loop参数配置，配置方法如下：

```python
from npu_bridge.npu_init import *

session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(session_config=session_config, iterations_per_loop=10)
```

同时需要使能GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件。Estimator模式下GetNext算子默认下沉，即enable_data_pre_proc默认为True，无需手工配置。

### 检查iterations_per_loop生效

开启“训练迭代循环下沉”功能后，可通过查看Host侧INFO日志中是否存在关键字“Insert op success”来判断iterations_per_loop是否生效。

可通过如下命令设置Host侧日志级别为INFO，INFO日志的默认输出路径为“$HOME/ascend/log/run/plog/”。

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

## session.run模式下使能训练迭代循环下沉

### 自动迁移场景

session.run模式下，通过[set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)设置iterations_per_loop参数，并修改session.run调用次数为原调用次数除以iterations_per_loop。

在脚本中找到“npu_config_proto”，在session配置中进行iterations_per_loop的配置，并通过[set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)接口设置小循环次数，如下述示例代码中的粗体部分。

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# 设置模型
# 学习率
learning_rate = 0.01
# 训练迭代次数
training_epochs = 10
# batch大小
batch_size = 100
# 每多少次迭代显示一次损失
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 建立模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数：交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度更新
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b = False  # 关闭混合计算，根据实际情况配置，默认关闭
custom_op.parameter_map["enable_data_pre_proc"].b = True # 若网络中存在GetNext算子，需要设置GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件
custom_op.parameter_map["iterations_per_loop"].i = 10 # 此处设置的值和set_iteration_per_loop接口设置的iterations_per_loop参数值保持一致，用于判断是否进行训练迭代下沉
config = npu_config_proto(config_proto=config)

# 训练模型
with tf.Session(config=config) as sess:
    sess.run(init)
    # sess.run模式下设置小循环次数为10
    train_op = util.set_iteration_per_loop(sess, optimizer, 10)  # 其中sess为已经创建的TensorFlow会话，optimizer为已定义的更新梯度的操作,10为设置的在Device侧进行训练迭代的次数

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

[set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)接口存在改图的操作，如果图无法修改（例如冻结了图或者使用tf.train.Supervisor创建session等），则无法使用set_iteration_per_loop接口设置小循环次数。此种场景下开发者可使用[create_iteration_per_loop_var](../../apiref/npu_util/create_iteration_per_loop_var.md)和[load_iteration_per_loop_var](../../apiref/npu_util/load_iteration_per_loop_var.md)接口设置小循环次数，如下述示例代码中的粗体部分。

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# 设置模型
# 学习率
learning_rate = 0.01
# 训练迭代次数
training_epochs = 10
# batch大小
batch_size = 100
# 每多少次迭代显示一次损失
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 建立模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数：交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度更新
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["mix_compile_mode"].b = False  # 关闭混合计算，根据实际情况配置，默认关闭
custom_op.parameter_map["enable_data_pre_proc"].b = True # 若网络中存在GetNext算子，需要设置GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件
custom_op.parameter_map["iterations_per_loop"].i = 10  # 此处设置的值和load_iteration_per_loop_var接口设置的iterations_per_loop参数值保持一致，用于功能校验
config = npu_config_proto(config_proto=config)

# 训练模型
with tf.Session(config=config) as sess:
    sess.run(init)
    # sess.run模式下设置小循环次数为10
    iteration = util.IterationPerLoop()   # 定义IterationPerLoop类对象
    train_op = iteration.create_iteration_per_loop_var(optimizer)    # optimizer为已定义的更新梯度的操作
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  # 冻结图
    iteration.load_iteration_per_loop_var(sess, 10)  # 设置小循环次数，其中sess为已经创建的TensorFlow会话，10为设置的在Device侧进行训练迭代的次数

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

### 手工迁移场景

session.run模式下，通过[set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)设置iterations_per_loop参数，并修改session.run调用次数为原调用次数除以iterations_per_loop，如下述示例代码中的粗体部分。

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# 设置模型
# 学习率
learning_rate = 0.01
# 训练迭代次数
training_epochs = 10
# batch大小
batch_size = 100
# 每多少次迭代显示一次损失
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 建立模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数：交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度更新
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True # 在昇腾AI处理器执行训练
custom_op.parameter_map["mix_compile_mode"].b = False  # 关闭混合计算，根据实际情况配置，默认关闭
custom_op.parameter_map["enable_data_pre_proc"].b = True # 若网络中存在GetNext算子，需要设置GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件
custom_op.parameter_map["iterations_per_loop"].i = 10 # 此处设置的值和set_iteration_per_loop接口设置的iterations_per_loop参数值保持一致，用于判断是否进行训练迭代下沉
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

# 训练模型
with tf.Session(config=config) as sess:
    sess.run(init)
    # sess.run模式下设置小循环次数为10
    train_op = util.set_iteration_per_loop(sess, optimizer, 10) # 其中sess为已经创建的TensorFlow会话，optimizer为已定义的更新梯度的操作,10为设置的在Device侧进行训练迭代的次数

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

[set_iteration_per_loop](../../apiref/npu_util/set_iteration_per_loop.md)接口存在改图的操作，如果图无法修改（例如冻结了图或者使用tf.train.Supervisor创建session等），则无法使用set_iteration_per_loop接口设置小循环次数。此种场景下开发者可使用[create_iteration_per_loop_var](../../apiref/npu_util/create_iteration_per_loop_var.md)和[load_iteration_per_loop_var](../../apiref/npu_util/load_iteration_per_loop_var.md)接口设置小循环次数，如下述示例代码中的粗体部分。

```python
from __future__ import print_function
import input_data
from npu_bridge.npu_init import *

mnist = input_data.read_data_sets("/test/", one_hot=True)

import tensorflow as tf

# 设置模型
# 学习率
learning_rate = 0.01
# 训练迭代次数
training_epochs = 10
# batch大小
batch_size = 100
# 每多少次迭代显示一次损失
display_step = 1

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 建立模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数：交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# 梯度更新
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化所有变量
init = tf.global_variables_initializer()

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True  # 在昇腾AI处理器执行训练
custom_op.parameter_map["mix_compile_mode"].b = False  # 关闭混合计算，根据实际情况配置，默认关闭
custom_op.parameter_map["enable_data_pre_proc"].b = True  # 若网络中存在GetNext算子，需要设置GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件
custom_op.parameter_map["iterations_per_loop"].i = 10  # 此处设置的值和load_iteration_per_loop_var接口设置的iterations_per_loop参数值保持一致，用于判断是否进行训练迭代下沉
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
# 训练模型
with tf.Session(config=config) as sess:
    sess.run(init)
    # sess.run模式下设置小循环次数为10
    iteration = util.IterationPerLoop()
    train_op = iteration.create_iteration_per_loop_var(optimizer) # optimizer为已定义的更新梯度的操作
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  # 冻结图
    iteration.load_iteration_per_loop_var(sess, 10)  # 设置小循环次数，其中sess为已经创建的TensorFlow会话，10为设置的在Device侧进行训练迭代的次数

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```

> [!NOTE]说明
> 修改循环次数后，建议根据脚本实际情况调整其他相关loop参数，如获取单步耗时、迭代数的更新等。

### 检查iterations_per_loop生效

开启“训练迭代循环下沉”功能后，可通过查看Host侧INFO日志中是否存在关键字“Insert op success”来判断iterations_per_loop是否生效。

可通过如下命令设置Host侧日志级别为INFO，INFO日志的默认输出路径为“$HOME/ascend/log/run/plog/”。

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```

## Keras模式下使能训练迭代循环下沉

在Ascend平台可以直接使用Keras原生API进行训练时，一次session.run调用在AI处理器执行训练迭代的次数固定为1。如需减少Host与Device间的交互次数，缩短训练时长，需要通过model_to_npu_estimator接口，将通过Keras构建的模型转换为NPUEstimator对象，同时通过NPURunConfig中的iterations_per_loop参数，指定一次session.run调用时在AI处理器执行训练迭代的次数。

TensorFlow原始代码：

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

迁移后的代码：

```python
from npu_bridge.npu_init import *

run_config = NPURunConfig(save_checkpoints_steps=2,
                          model_dir=model_path,
                          iterations_per_loop=10)
# 将通过Keras构建的模型转换为NPUEstimator对象
est_resnet = keras_to_npu.model_to_npu_estimator(keras_model=keras_model, config=run_config)
# 执行训练
est_resnet.train(input_fn=lambda: input_fn(), max_steps=1000)
```

另外，还需要将Keras的数据预处理部分迁移为NPUEstimator中input_fn，请用户自行实现，下面仅给出迁移示例。以下示例中，Keras的数据预处理方式为从文件夹中读取图片数据，并自动打标签，经过数据增强resize、归一化、水平翻转等操作，最终输出数据；而Estimator模式下，数据预处理方式采用相同的从文件list中读取的方式，区别在于需要提前读取好文件名的list，并给每张图片打上标签输出标签的list。同样的经过归一化、resize、水平翻转的数据增强操作，输出数据。

TensorFlow原始代码：

```python
# keras从文件夹中读取图片
train_datagen = ImageDataGenerator(rescale=1./255,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory('data/',
                                                    target_size=(224, 224),
                                                    batch_size=32,
                                                    class_mode='sparse')
```

迁移后的代码：

```python
# 函数的功能是将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
  image = tf.read_file(filename)
  image = tf.image.decode_image(image)
  image = image / 255.0
  image = tf.image.resize_images(image, [224, 224])
  image = tf.image.random_flip_left_right(image)
  return image, label

def input_fn():
    # 图片文件的列表，图片的list需要用户自己生成
    filenames = tf.constant(["/data/image1.jpg", "/data/image2.jpg", ...])
    # label[i]就是图片filenames[i]的label, label的list需要用户自己生成
    labels = tf.constant([0, 5, ...])
    # 此时dataset中的一个元素是(filename, label)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels)).repeat(10)
    # 此时dataset中的一个元素是(image_resized, label)
    dataset = dataset.map(_parse_function)
    # 此时dataset中的一个元素是(image_resized_batch, label_batch)
    dataset = dataset.shuffle(buffer_size=1000).batch(32)
    return dataset
```

其他说明：Keras的callback回调函数在转换为NPUEstimator对象后无法使用。

**检查iterations_per_loop生效**

开启“训练迭代循环下沉”功能后，可通过查看Host侧INFO日志中是否存在关键字“Insert op success”来判断iterations_per_loop是否生效。

可通过如下命令设置Host侧日志级别为INFO，INFO日志的默认输出路径为“$HOME/ascend/log/run/plog/”。

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1
```
