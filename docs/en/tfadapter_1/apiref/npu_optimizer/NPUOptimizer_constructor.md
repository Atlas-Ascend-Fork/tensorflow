# NPUOptimizer Constructor

## Description

Constructs an object of class  **NPUOptimizer**, which combines the  [NPUDistributedOptimizer](NPUDistributedOptimizer_constructor.md)  and  [NPULossScaleOptimizer](../npu_loss_scale_optimizer/NPULossScaleOptimizer_constructor.md)  optimizers. It provides the following functions:

- Loss scaling: Loss scaling can be enabled during mixed precision training to solve the underflow problem caused by a small float16 representation range.
- Distributed training: With an NPU distributed training optimizer wrapped from a single-server training optimizer, calculated gradients can be aggregated in single-server single-device, single-server multi-device, and multi-server multi-device networking modes.
- Communication tailing optimization: By changing a computation dependency relationship, a computation operation that does not depend on the last AR \(gradient aggregation fragment\) is scheduled to be performed in parallel with the last AR, to optimize communication tailing.

## Prototype

```python
class NPUOptimizer(tf.train.Optimizer):
    def __init__(self, opt, loss_scale_manager=None, is_distributed=False, is_loss_scale=False,
                 is_tailing_optimization=False, name=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| opt | Input | Single-server training optimizer for gradient calculation and weight update. |
| loss_scale_manager | Input | This parameter needs to be configured only when is_loss_scale is set to True and the loss scaling function is enabled. This parameter determines the update mode of loss scaling, including static update and dynamic update.<br><br>  - Before creating NPUOptimizer, you can instantiate a FixedLossScaleManager class to set the loss scaling with a static value. For details about the constructor of the FixedLossScaleManager class, see [FixedLossScaleManager Constructor](../npu_loss_scale_manager/FixedLossScaleManager_constructor.md).<br>  - Before creating NPUOptimizer, you can instantiate an ExponentialUpdateLossScaleManager class to dynamically configure loss scaling. For details about the constructor of the ExponentialUpdateLossScaleManager class, see [ExponentialUpdateLossScaleManager Constructor](../npu_loss_scale_manager/ExponentialUpdateLossScaleManager_constructor.md). |
| is_distributed | Input | Distributed training enable.<br><br>  - True: enabled.<br>  - False (default): disabled. |
| is_loss_scale | Input | Loss scaling enable.<br><br>  - True: enabled. True: enabled (recommended if mixed precision training is enabled). In this case, the value of loss_scale_manager cannot be None.<br>  - False (default): disabled. |
| is_tailing_optimization | Input | Communication hangover optimization enable, for improving training performance. This function takes effect only when is_distributed is set to True.<br><br>  - True: enabled.<br>  - False (default): disabled.<br><br>Argument of this parameter must be the same as that set in [NPURunConfig Constructor](../npu_config/npurunconfig_constructor/README.md). |
| name | Input | Name of the optimizer. |

## Returns

An object of the  **NPUOptimizer**  class

## Example

```python
import tensorflow as tf
from npu_bridge.npu_init import *

# Define a single-server optimizer.
optimizer = LAMBOptimizer(
          learning_rate=learning_rate,
          weight_decay_rate=0.01,
          beta_1=0.9,
          beta_2=0.999,
          epsilon=1e-6,
          exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

# Enable loss scaling.
if tf.flags.FLAGS.npu_bert_loss_scale not in [None, -1]:
  if tf.flags.FLAGS.npu_bert_loss_scale == 0:
    loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=tf.flags.FLAGS.init_loss_scale_value, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
  elif tf.flags.FLAGS.npu_bert_loss_scale >= 1:
    loss_scale_manager = FixedLossScaleManager(loss_scale=tf.flags.FLAGS.npu_bert_loss_scale)
  else:
    raise ValueError("Invalid loss scale: %d" % tf.flags.FLAGS.npu_bert_loss_scale)
  optimizer = NPUOptimizer(optimizer, loss_scale_manager, is_distributed=tf.flags.FLAGS.distributed, is_loss_scale=True, is_tailing_optimization=True)

# Disable loss scaling.
else:
  optimizer = NPUOptimizer(optimizer, is_distributed=tf.flags.FLAGS.distributed)
```
