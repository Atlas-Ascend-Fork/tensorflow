# Estimator迁移

## Estimator简介

Estimator API属于TensorFlow的高阶API，可简化机器学习的编程过程。Estimator有很多优势，例如：对分布式的良好支持、简化了模型的创建工作、有利于模型开发者之间的代码分享等。

TensorFlow 2.6版本中继续支持该高阶API，如果需要沿用在TensorFlow1.x版本中的用法，则可以通过compat.v1模块调用，调用方式如下：

```python
tf.compat.v1.estimator.Estimator
```

使用compat.v1的Estimator进行训练脚本开发的流程为：

1. 数据预处理，创建输入函数input_fn。
2. 模型构建，构建模型函数model_fn。
3. 运行配置，实例化Estimator，并传入RunConfig类对象作为运行参数。
4. 执行训练，在Estimator上调用训练方法Estimator.train\(\)，利用指定输入对模型进行固定步数的训练。

下面介绍如何迁移此类Estimator训练脚本，以便在AI处理器上进行训练。

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

## 模型构建

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

## 运行配置

TensorFlow通过RunConfig配置运行参数，用户需要按照如下示例，更改config相关配置。

TensorFlow原始代码：

```python
session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)

config=tf.estimator.RunConfig(
  session_config=session_config,
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  # ... ...
  )
```

迁移后的代码：

```python
session_config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=False)
custom_op = sess_config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name = "NpuOptimizer"
sess_config.graph_options.rewrite_options.remapping = rewriter_config_pb2.RewriterConfig.OFF

npu_config=NPURunConfig(
  session_config=sess_config,
  model_dir=FLAGS.model_dir,
  save_checkpoints_steps=FLAGS.save_checkpoints_steps,
  # ... ...
  )
```

## 创建Estimator

Estimator的迁移需要更改其config参数为上述npu_config，并将TensorFlow的Estimator迁移为NPUEstimator。

TensorFlow原始代码：

```python
mnist_classifier=tf.compat.v1.estimator.Estimator(
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

## 执行训练

利用指定输入对模型进行固定步数的训练，此部分代码无需改造。

```python
mnist_classifier.train(
  input_fn=train_input_fn,
  steps=20000,
  hooks=[logging_hook])
```
