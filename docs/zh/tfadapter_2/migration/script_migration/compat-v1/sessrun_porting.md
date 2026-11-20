# sess.run迁移

## sess.run简介

sess.run API属于TensorFlow的低阶API，相对于Estimator来讲，灵活性较高，但模型的实现较为复杂。

TensorFlow 2.6版本中已经弃用该API，如果需要在TensorFlow 2.6版本中使用sess.run，需要通过compat.v1模块引用，方式如下：

```python
tf.compat.v1.Session.run
```

使用sess.run API进行训练脚本开发的流程为：

1. 数据预处理。
2. 模型搭建/计算Loss/梯度更新。
3. 创建session并初始化资源。
4. 执行训练。

下面介绍如何迁移此类sess.run训练脚本，以便在AI处理器上进行训练。

## 头文件增加

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
```

## 数据预处理

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

当原始网络脚本中使用dataset.batch\(batch_size\)返回动态形状时，由于数据流中剩余的样本数可能小于batch大小，导致网络中最后一个step的shape与之前的shape不一致，此种场景下会进入动态shape编译流程。为提升网络编译性能，建议将drop_remainder设置为True，丢弃文件中的最后几个样本，确保网络中每个step的shape一致。

```python
  dataset = dataset.batch(batch_size, drop_remainder=True)
```

但需要注意的是：推理时，当最后一次迭代的推理数据量小于batch_size时，需要补齐空白数据到batch_size，因为有些脚本最后会加个断言，验证结果的数量要和验证数据的数量一致。

```python
 assert num_written_lines == num_actual_predict_examples
```

## 模型搭建/计算Loss/梯度更新

一般情况下，此部分代码无需改造。如下情况需要进行适配修改：

- 如果原始网络中使用到了tf.device，需要删除相关代码。
- 对于原始网络中的gelu，建议替换为CANN对应的API实现，以获得更优性能。

    TensorFlow原始代码：

    ```python
    def gelu(x):
      cdf = 0.5 * (1.0 + tf.tanh(
         (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
      return x*cdf
    layers = gelu(x)
    ```

    迁移后的代码：

    ```python
    layers = npu_unary_ops.gelu(x)
    ```

## 创建session并初始化资源

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

- 以下配置默认开启，必须显式关闭：
  - rewrite_options.remapping
  - rewrite_options.memory_optimization

TensorFlow原始代码：

```python
#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)

#变量初始化
init=tf.compat.v1.global_variables_initializer()
sess=tf.compat.v1.Session()
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

迁移后的代码：

```python
#构造迭代器
iterator=Iterator.from_structure(train_dataset.output_types,train_dataset.output_shapes)

#取batch数据
next_batch=iterator.get_next()

#迭代器初始化
training_init_op=iterator.make_initializer(train_dataset)

#变量初始化
init=tf.compat.v1.global_variables_initializer()

#创建session
config = tf.compat.v1.ConfigProto()
custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF  # 必须显式关闭
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF  # 必须显式关闭
sess = tf.compat.v1.Session(config=config)
sess.run(init)

#Get the number of training/validation steps per epoch
train_batches_per_epoch=int(np.floor(train_size/batch_size))
```

tf.compat.v1.Session原生功能在NPU上全部支持。

另外，NPU还支持自动混合精度等功能，如果用户需要进行相关使能，可以参考对应接口说明。

## 执行训练

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
    _,train_loss = sess.run([train_op, loss],feed_dict={x:img_batch,y_:label_batch,is_training:True})
```

但是，如果用户训练脚本中没有使用with创建session，而是将session对象作为自己定义的一个类成员，那么需要在迁移后的脚本中显式调用sess.close\(\)。

```python
sess = tf.compat.v1.Session(config=config)
sess.run(...)
sess.close()
```

这是因为，GEOP的析构函数在tf.compat.v1.session的close方法中会被调用到，如果是with创建的session，with会调用session的__exit__方法，里面会自动调用close：

```python
with tf.compat.v1.Session(config=config) as sess:
    sess.run(...)
```

如果是其他情况，例如把session对象作为自己定义的一个类成员，那么退出之前需要显式调用sess.close\(\)，以确保可以正常退出。
