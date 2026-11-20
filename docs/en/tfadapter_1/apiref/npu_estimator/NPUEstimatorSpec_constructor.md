# NPUEstimatorSpec Constructor

## Description

Constructor of the  **NPUEstimatorSpec**  class. The  **NPUEstimatorSpec**  class inherits the  **EstimatorSpec**  class of the  TensorFlow  and can call the native APIs of the base class to define specific model objects.

**EstimatorSpec**  is the return data structure of  **model_fn**, including the  **mode**,  **predictions**,  **loss**,  **train_op**, and  **export_outputs**  fields. If  **EstimatorSpec**  cannot meet the training requirements, define  **NPUEstimatorSpec**  to replace  **EstimatorSpec**.

## Prototype

```python
class NPUEstimatorSpec(model_fn_lib.EstimatorSpec):
    def __new__(cls,
                mode,
                predictions=None,
                loss=None,
                train_op=None,
                eval_metric_ops=None,
                export_outputs=None,
                training_chief_hooks=None,
                training_hooks=None,
                scaffold=None,
                evaluation_hooks=None,
                prediction_hooks=None,
                host_call=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| mode | Input | Mode, indicating whether the current operation is training, validation, or inference. This parameter is inherited from EstimatorSpec.<br><br>  - ModeKeys.TRAIN: training<br>  - ModeKeys.EVAL: validation<br>  - ModeKeys.PREDICT: inference |
| predictions | Input | Inference output tensor, required when mode is set to ModeKeys.PREDICT. It is a parameter inherited from EstimatorSpec. |
| loss | Input | Training loss. It is a parameter inherited from EstimatorSpec. |
| train_op | Input | Training operator. It is a parameter inherited from EstimatorSpec. |
| eval_metric_ops | Input | Dictionary of measurement results (based on tensor names). It is a parameter inherited from EstimatorSpec.<br>The dictionary value can be one of the following:<br><br>  - Metric instance.<br>  - Result of calling the metric function, that is, the (metric_tensor, update_op) tuple. |
| export_outputs | Input | Saves a model and describes the output format of the model exported to SavedModel. This parameter is inherited from EstimatorSpec. |
| training_chief_hooks | Input | SessionRunHooks set of the primary node during training. It is a parameter inherited from EstimatorSpec. |
| training_hooks | Input | SessionRunHooks set during training. It is a parameter inherited from EstimatorSpec. |
| scaffold | Input | Scaffold definition (providing the capabilities of customizing saver, init_op, summary_op, and global_step). It is a parameter inherited from EstimatorSpec. |
| evaluation_hooks | Input | SessionRunHooks set during validation. It is a parameter inherited from EstimatorSpec. |
| prediction_hooks | Input | SessionRunHooks set during inference. It is a parameter inherited from EstimatorSpec. |
| host_call | Input | Captures the summary information and sends the information of each step back to the host side. It is a new parameter in NPUEstimatorSpec.<br>host_call is a tuple consisting of a function and a list or dictionary of tensors. It is used to return a list of tensors. <br>host_call applies to train() and evaluate(). |

## Returns

An object of the  **NPUEstimatorSpec**  class

## Example

```python
from npu_bridge.npu_init import *
...
host_call = (_host_call_fn, [global_step, loss])
return NPUEstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op, host_call=host_call)
```
