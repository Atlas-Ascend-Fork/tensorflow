# NPUOutputTensorHook Constructor

## Description

Constructs an object of class  **NPUOutputTensorHook**.  **NPUOutputTensorHook**  is a hook for training, evaluation, and prediction of  **NPUEstimator**, and it can call the user-defined  **output_fn**  every  _N_  steps or at the end to print the output tensors. The  **NPUOutputTensorHook**  class inherits the  **LoggingTensorHook**  class and can call native APIs of the base class.

## Prototype

```python
class NPUOutputTensorHook(basic_session_run_hooks.LoggingTensorHook):
    def __init__(self, tensors,
                 dependencies=None,
                 output_fn=None,
                 output_every_n_steps=0
                 )
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| tensors | Input | Name set of the input tensors, in dictionary or list format. |
| dependencies | Input | Dependencies corresponding to tensors. |
| output_fn | Input | Print function of tensor output. |
| output_every_n_steps | Input | The user-defined output_fn, which is called when the session is executed for N times and the training script is executed. |

## Returns

An object of the  **NPUOutputTensorHook**  class

## Restrictions

When  **iterations_per_loop \> 1**,  **output_fn**  cannot be called as specified by  **output_every_n_steps**.

## Example

```python
from npu_bridge.npu_init import *

# Define output_fn.
def output_fn(inputs):
  device_id = os.environ["ASCEND_DEVICE_ID"]
  output_file = os.path.join("/code", device_id, "test_npu_output_tensor.txt")
  for item in inputs:
    content = "step:{},loss:{}".format(str(item['global_step']), str(item['loss']))
    with open(output_file, 'a') as f:
      f.write(content)
      f.write("\n")

# Define output_hook for calling the user-defined output_fn.
  tensors = {'global_step': global_step, 'loss': loss}
  output_hook = NPUOutputTensorHook(
      tensors,
      dependencies=train_op_list,
      output_fn=output_fn,
      output_every_n_steps=10)
  train_hook.append(output_hook)

# Pass the hook to EstimatorSpec.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      training_chief_hooks=train_hook,
      eval_metric_ops=metrics)
```
