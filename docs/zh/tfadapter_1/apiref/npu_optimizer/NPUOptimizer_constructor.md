# NPUOptimizer构造函数

## 功能说明

NPUOptimizer类的构造函数，该优化器将[NPUDistributedOptimizer](NPUDistributedOptimizer_constructor.md)和[NPULossScaleOptimizer](../npu_loss_scale_optimizer/NPULossScaleOptimizer_constructor.md)优化器合并。主要提供如下功能：

- Loss Scaling：支持在混合精度训练中使能Loss Scaling，从而解决由于float16表示范围较小导致的下溢出问题。
- 分布式训练：包装用户提供的单机训练优化器，构造NPU分布式训练优化器，支持单机单卡、单机多卡、多机多卡等组网形式下，各个Device之间计算梯度后执行梯度聚合操作。
- 通信拖尾优化：通过计算依赖关系的改变，将不依赖于最后一个AR（梯度聚合分片）的计算操作调度到和最后一个AR并行进行，以达到优化通信拖尾时间的目的。

## 函数原型

```python
class NPUOptimizer(tf.train.Optimizer):
    def __init__(self, opt, loss_scale_manager=None, is_distributed=False, is_loss_scale=False,
                 is_tailing_optimization=False, name=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| opt | 输入 | 用于梯度计算和更新权重的单机版训练优化器。 |
| loss_scale_manager | 输入 | 仅当is_loss_scale为True，开启Loss Scaling功能时需要配置该参数，用于决定Loss Scale的更新方式，包括静态更新和动态更新。<br>  - 用户在创建NPUOptimizer之前，可以实例化一个FixedLossScaleManager类进行静态Loss Scale的配置（Loss Scale值固定，用户需指定Loss Scale值）。FixedLossScaleManager类的构造函数，请参见[FixedLossScaleManager构造函数](../npu_loss_scale_manager/FixedLossScaleManager_constructor.md)。<br>  - 用户在创建NPUOptimizer之前，可以实例化一个ExponentialUpdateLossScaleManager类进行动态Loss Scale的配置。ExponentialUpdateLossScaleManager类的构造函数，请参见[ExponentialUpdateLossScaleManager构造函数](../npu_loss_scale_manager/ExponentialUpdateLossScaleManager_constructor.md)。 |
| is_distributed | 输入 | 是否开启分布式训练，取值：<br>  - True：开启。<br>  - False（默认值）：不开启。 |
| is_loss_scale | 输入 | 是否开启Loss Scaling，取值：<br>  - True：开启Loss Scaling。使用自动混合精度功能后，推荐开启，此时loss_scale_manager的值不能为None。<br>  - False（默认值）：不开启Loss Scaling。 |
| is_tailing_optimization | 输入 | 是否开启通信拖尾优化，用于提升训练性能，该功能仅在is_distributed为True的情况下配置生效。<br>  - True：开启通信拖尾。<br>  - False（默认值）：不开启通信拖尾。<br>必须和[NPURunConfig构造函数](../npu_config/npurunconfig_constructor/README.md)中的配置值保持一致。 |
| name | 输入 | 优化器名称。 |

## 返回值

返回NPUOptimizer类对象。

## 调用示例

```python
import tensorflow as tf
from npu_bridge.npu_init import *

#定义单机优化器
optimizer = LAMBOptimizer(
          learning_rate=learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

#使能Loss Scaling
if tf.flags.FLAGS.npu_bert_loss_scale not in [None, -1]:
  if tf.flags.FLAGS.npu_bert_loss_scale == 0:
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=tf.flags.FLAGS.init_loss_scale_value, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
  elif tf.flags.FLAGS.npu_bert_loss_scale >= 1:
    loss_scale_manager = FixedLossScaleManager(loss_scale=tf.flags.FLAGS.npu_bert_loss_scale)
  else:
    raise ValueError("Invalid loss scale: %d" % tf.flags.FLAGS.npu_bert_loss_scale)
  optimizer = NPUOptimizer(optimizer, loss_scale_manager, is_distributed=tf.flags.FLAGS.distributed, is_loss_scale=True, is_tailing_optimization=True)

#不使能loss_scale
else:
  optimizer = NPUOptimizer(optimizer, is_distributed=tf.flags.FLAGS.distributed)
```
