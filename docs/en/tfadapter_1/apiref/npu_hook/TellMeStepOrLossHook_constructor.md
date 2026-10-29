# TellMeStepOrLossHook Constructor

## Description

Constructs an object of class  **TellMeStepOrLossHook**, which is used to notify the bottom-layer software of the serial number of the current  **step**  and the total number of  **step**s or the current  **loss**  and the target  **loss**.

## Prototype

```python
class TellMeStepOrLossHook(session_run_hook.SessionRunHook):
    def __init__(self, step=None, total_step=None, loss=None, final_loss=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| step | Input | Tensor name of the current step. |
| total_step | Input | Total number of training steps of the training script. |
| loss | Input | Tensor name of the current loss. |
| final_loss | Input | Target loss of the training script. |

## Returns

An object of the  **TellMeStepOrLossHook**  class

## Restrictions

When  **Iterations_per_loop**  is greater than  **1**, the bottom-layer software is notified of the serial number of the current  **step**  or  **loss**  each time the number of steps increases by  **Iterations_per_loop**. It is impossible to notify the bottom-layer software of such information each time the number of steps increases by 1, as this may affect some functions that depend on the result of this  **hook**  function.

## Example

```python
from npu_bridge.npu_init import *
est = NPUEstimator(
        model_fn=model_fn,
        config=config,
        params=params)
hooks = []
max_steps = 10000
# Splitting by step: In this example, the tensor name of the current step is global_step:0, and the total number of steps is 10000. Set this parameter based on the tensor name of the actual step and total number of steps.
my_hook = TellMeStepOrLossHook(step='global_step:0', total_step=max_steps)
# Splitting by loss: In this example, the tensor name of the loss is loss:0, and the target loss is 7.1. Set this parameter based on the tensor name of the actual loss and the value of the target loss.
# my_hook = TellMeStepOrLossHook(loss='loss:0', final_loss=7.1)
hooks.append(my_hook)
# Start training.
est.train(
          input_fn=imagenet_train.input_fn,
          max_steps=max_steps,
          hooks=hooks)
```
