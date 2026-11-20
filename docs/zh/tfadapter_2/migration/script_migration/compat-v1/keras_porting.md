# Keras迁移

## Keras简介

Keras和Estimator类似，都属于TensorFlow高阶API，提供了方便的构图功能，并对训练、评估、验证、导出提供了方便的接口。

TensorFlow 2.6版本中继续支持Keras API，如果需要沿用在TensorFlow1.x版本中的用法，则可以通过compat.v1模块调用，调用方式如下：

```python
tf.compat.v1.Session
```

使用Keras API进行训练脚本开发的一般步骤为：

1. 数据预处理。
2. 模型搭建。
3. 模型编译。
4. 模型训练。

> [!CAUTION]注意
> 当前仅支持通过TensorFlow的Keras API编写的训练脚本，而不支持原生Keras API。

下面介绍如何迁移此类Keras训练脚本，以便在AI处理器上进行训练。

## 头文件增加

对于以下步骤中涉及修改的Python文件，新增以下头文件引用，用于导入NPU相关库。

```python
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
```

## 迁移点说明

如果您是Keras训练脚本，由于Keras迁移到NPU运行时，某些功能受限支持，例如不支持动态学习率等，因此不建议用户在NPU上迁移Keras开发的网络脚本。如需在NPU运行Keras脚本，您需要对脚本进行如下修改：

创建一个TensorFlow会话并注册Keras，需要增加相关配置项以便在AI处理器执行训练。同时在训练结束时，关闭会话。

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

#数据预处理...
#模型搭建...
#模型编译...
#模型训练...

sess.close()
```
