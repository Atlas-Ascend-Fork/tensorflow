# 单机单卡脚本迁移

## 迁移思路

TF Adapter适配TensorFlow 1.15的思路是：

利用TensorFlow提供的config扩展机制，将NPU上的相关功能配置通过config的方式向下传递；然后再利用TensorFlow提供的优化器注册机制注册NPU上的优化器并根据config配置对图进行处理；最终将处理好的图下发到CANN平台执行。

所以手工迁移的主要点是添加NPU上的config配置，将扩展后的NPU config传入tf.Session的config参数。TensorFlow 1.15中常见的Estimator、sess.run、Keras三种脚本最终都是调用sess.run实现的，所以三种脚本的迁移方式本质上是一致的。

## Estimator迁移

若原始TensorFlow网络基于Estimator API构造，可参见本节了解手工迁移全流程。

### Estimator简介

Estimator API属于TensorFlow的高阶API，可极大简化机器学习的编程过程。Estimator有很多优势，例如：对分布式的良好支持、简化了模型的创建工作、有利于模型开发者之间的代码分享等。

使用Estimator进行训练脚本开发的流程为：

1. 数据预处理，创建输入函数input_fn。
2. 模型构建，构建模型函数model_fn。
3. 运行配置，实例化Estimator，并传入RunConfig类对象作为运行参数。
4. 执行训练，在Estimator上调用训练方法Estimator.train\(\)，利用指定输入对模型进行固定步数的训练。

下面介绍如何迁移Estimator训练脚本，以便在NPU上进行训练。

### 头文件增加

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
from npu_bridge.npu_init import *
```

> [!NOTE]说明
> 引入上述头文件后，训练脚本默认在NPU执行。

### 数据预处理

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

当原始网络脚本中使用dataset.batch\(batch_size\)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，导致网络中最后一个step的shape与之前的shape不一致，此种场景下会进入动态shape编译流程。为提升网络编译性能，建议将drop_remainder设置为True，丢弃文件中的最后几个样本，确保网络中每个step的shape一致。

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

但需要注意的是：推理时，当最后一次迭代的推理数据量小于batch_size时，需要补齐空白数据到batch_size，因为有些脚本最后会加个断言，验证结果的数量要和验证数据的数量一致。

```python
 assert num_written_lines == num_actual_predict_examples
```

### 模型构建

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

- 对于原始网络中的dropout，请替换为CANN对应的API实现，以获得更优性能，但需关注对网络精度的影响。
  - 如果存在tf.nn.dropout，请修改为：

    ```python
    layers = npu_ops.dropout()
    ```

  - 如果存在tf.layers.dropout/tf.layers.Dropout/tf.keras.layers.Dropout/tf.keras.layers.SpatialDropout1D/tf.keras.layers.SpatialDropout2D/tf.keras.layers.SpatialDropout3D，请增加头文件引用：

    ```python
    from npu_bridge.estimator.npu import npu_convert_dropout
    ```

- 对于原始网络中的gelu，请替换为CANN对应的API实现，以获得更优性能。

    TensorFlow原始代码：

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    迁移后的代码：

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

### 运行配置

TensorFlow通过RunConfig配置运行参数，用户需要将RunConfig迁移为NPURunConfig。NPURunConfig类继承了RunConfig类，因此我们在迁移时可直接按照如下示例进行脚本修改，大多数参数可不变。

TensorFlow原始代码：

```python
config=tf.estimator.RunConfig(
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
```

迁移后的代码：

```python
npu_config=NPURunConfig(
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  # 如果原始网络中使用了tf.device相关代码，则需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
  session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  )
```

但是，部分参数（包括train_distribute/device_fn/protocol/eval_distribute/experimental_distribute）在NPURunConfig中不支持，如果原始脚本使用到了，用户需要进行删除。

如果原始网络中使用了tf.device相关代码，需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。

同时，我们在NPURunConfig新增了部分参数，从而提升训练性能与精度，例如iterations_per_loop、precision_mode等，详细的参数信息可参见[NPURunConfig构造函数](../../apiref/npu_config/npurunconfig_constructor/README.md)。

### 创建Estimator对象

用户需要将TensorFlow的Estimator对象迁移为NPUEstimator，NPUEstimator类继承了Estimator类，因此我们在迁移时按照如下示例直接更改接口即可，参数可保持不变。

TensorFlow原始代码：

```python
mnist_classifier=tf.estimator.Estimator(
  model_fn=cnn_model_fn,
  config=config,
  model_dir="/tmp/mnist_convnet_model")
```

迁移后的代码：

```python
mnist_classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  config=npu_config,
  model_dir="/tmp/mnist_convnet_model"
  )
```

### 执行训练

利用指定输入对模型进行训练，此部分代码无需改造。

```python
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=20000,
  hooks=[logging_hook])
```

## sess.run迁移

若原始TensorFlow网络基于sess.run API构造，可参见本节了解手工迁移全流程。

### sess.run简介

sess.run API属于TensorFlow的低阶API，相对于Estimator来讲，灵活性较高，但模型的实现较为复杂。

使用sess.run API进行训练脚本开发的流程为：

1. 数据预处理。
2. 模型搭建/计算Loss/梯度更新。
3. 创建session并初始化资源。
4. 执行训练。

下面介绍如何迁移sess.run训练脚本，以便在AI处理器上进行训练。

### 头文件增加

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
from npu_bridge.npu_init import *
```

> [!NOTE]说明
> 引入上述头文件后，训练脚本默认在NPU执行。

### 数据预处理

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

当原始网络脚本中使用dataset.batch\(batch_size\)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，导致网络中最后一个step的shape与之前的shape不一致，此种场景下会进入动态shape编译流程。为提升网络编译性能，建议将drop_remainder设置为True，丢弃文件中的最后几个样本，确保网络中每个step的shape一致。

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

但需要注意的是：推理时，当最后一次迭代的推理数据量小于batch_size时，需要补齐空白数据到batch_size，因为有些脚本最后会加个断言，验证结果的数量要和验证数据的数量一致。

```python
 assert num_written_lines == num_actual_predict_examples
```

### 模型搭建/计算Loss/梯度更新

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

- 对于原始网络中的dropout，请替换为CANN对应的API实现，以获得更优性能，但需关注对网络精度的影响。
  - 如果存在tf.nn.dropout，请修改为：

    ```python
    lars = npu_ops.dropout()
    ```

  - 如果存在tf.layers.dropout/tf.layers.Dropout/tf.keras.layers.Dropout/tf.keras.layers.SpatialDropout1D/tf.keras.layers.SpatialDropout2D/tf.keras.layers.SpatialDropout3D，请增加头文件引用：

    ```python
    from npu_bridge.estimator.npu import npu_convert_dropout
    ```

- 对于原始网络中的gelu，请替换为CANN对应的API实现，以获得更优性能。

    TensorFlow原始代码：

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu()
    ```

    迁移后的代码：

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

### 创建session并初始化资源

在AI处理器上通过sess.run模式执行训练脚本时，相关配置说明：

- 以下配置默认关闭，请勿开启：

    rewrite_options.disable_model_pruning

- 以下配置默认开启，请勿关闭：
  - rewrite_options.function_optimization
  - rewrite_options.constant_folding
  - rewrite_options.shape_optimization
  - rewrite_options.arithmetic_optimization
  - rewrite_options.loop_optimization
  - rewrite_options.dependency_optimization
  - rewrite_options.layout_optimizer

- 以下配置默认开启，需要显示关闭：
  - rewrite_options.remapping
  - rewrite_options.memory_optimization

- 如果原始网络中使用了tf.device相关代码，需要增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。

TensorFlow原始代码：

```python
# 构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# 取batch数据
next_batch=iterator.get_next()

# 迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)

# 变量初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

迁移后的代码：

```python
# 构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

# 取batch数据
next_batch=iterator.get_next()

# 迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)

# 变量初始化
init=tf.global_variables_initializer()

# 增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
config = tf.ConfigProto(allow_soft_placement=True)
# 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# 需要显示关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 显式关闭
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 显式关闭
sess = tf.Session(config=config)
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

tf.Session原生功能在Ascend平台上全部支持。

另外，Ascend平台还支持自动混合精度等功能，如果用户需要进行相关使能，可以参考[session配置](../../apiref/session_config/README.md)。

### 执行训练

此部分代码无需改造，例如：

```python
#开始循环迭代
for epoch in range(num_epochs):
  ##Initialize iterator with the training dataset
  sess.run(training_init_op)
  for step in range(train_batches_per_epoch):
    #get next batch of data
    img_batch,label_batch=sess.run(next_batch)
    #run the training op
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch, y_:label_batch, is_training:True})
```

但是，如果用户训练脚本中没有使用with创建session，例如将session对象作为自己定义的一个类成员，那么需要在迁移后的脚本中显式调用sess.close\(\)。

```python
sess = tf.Session(config=config)
sess.run(...)
sess.close()
```

这是因为，GEOP的析构函数在tf.Session的close方法中会被调用到，如果是with创建的session，with会调用session的__exit__方法，里面会自动调用close：

```python
with tf.Session(config=config) as sess:
    sess.run(...)
```

如果是其他情况，例如是把session对象作为自己定义的一个类成员，那么退出之前需要显式调用sess.close\(\)，这样才可以保证正常退出。

## Keras迁移

若原始TensorFlow网络基于Keras API构造，可参见本节了解手工迁移全流程。

### Keras简介

Keras和Estimator类似，都属于TensorFlow高阶API，提供了方便的构图功能，并为训练、评估、验证、导出提供了方便的接口。使用TensorFlow的Keras API进行训练脚本开发的一般步骤为：

1. 数据预处理。
2. 模型搭建。
3. 模型编译。
4. 模型训练。

> [!NOTE]说明
> 当前仅支持通过TensorFlow的Keras API编写的训练脚本，不支持原生Keras API。

下面介绍如何迁移Keras训练脚本，以便在AI处理器上进行训练。

### 头文件增加

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
from npu_bridge.npu_init import *
```

> [!NOTE]说明
> 引入上述头文件后，训练脚本默认在NPU执行。

### 迁移点说明

创建一个TensorFlow会话并且注册Keras，并增加相关配置项以便在AI处理器执行训练。同时在训练结束时，关闭会话。

```python
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import backend as K
from npu_bridge.npu_init import *

# 增加session配置“allow_soft_placement=True”，允许TensorFlow自动分配设备。
sess_config = tf.ConfigProto(allow_soft_placement=True)
# 添加名称为“NpuOptimizer”的NPU优化器，网络编译时，NPU只会遍历“NpuOptimizer”下的session配置。
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
# 显式关闭TensorFlow的remapping、memory_optimization功能，避免与NPU中的功能冲突。
sess_config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
sess_config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
sess = tf.Session(config=sess_config)
K.set_session(sess)

#数据预处理...
#模型搭建...
#模型编译...
#模型训练...

sess.close()
```

通过以上配置迁移后，一次session.run调用在昇腾AI处理器执行训练迭代的次数固定为1，如需减少Host与Device间的交互次数，缩短训练时长，需要通过model_to_npu_estimator接口，将通过Keras构建的模型转换为NPUEstimator对象，同时通过NPURunConfig中的iterations_per_loop参数，指定一次session.run调用时在昇腾AI处理器执行训练迭代的次数，具体请参考[Keras模式下使能训练迭代循环下沉](../performance_tuning/iteration_offload.md#keras模式下使能训练迭代循环下沉)。
