# NPUCheckpointSaverHook Constructor

## Description

Constructs an object of class  **NPUCheckpointSaverHook**, which is used to save the checkpoint file. The  **NPUCheckpointSaverHook**  class inherits the  **CheckpointSaverHook**  class and can call the native APIs of the base class to record model information during training.

## Prototype

```python
class NPUCheckpointSaverHook(basic_session_run_hooks.CheckpointSaverHook):
    def __init__(self,
                 checkpoint_dir,
                 save_secs=None,
                 save_steps=None,
                 saver=None,
                 checkpoint_basename="model.ckpt",
                 scaffold=None,
                 listeners=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| checkpoint_dir | Input | Checkpoint file directory |
| save_secs | Input | Interval (in seconds) for saving the checkpoint file |
| save_steps | Input | Interval (in steps) for saving the checkpoint file |
| saver | Input | Saver object |
| checkpoint_basename | Input | Basename of the checkpoint file |
| scaffold | Input | Scaffold of the saver object |
| listeners | Input | Example of the CheckpointSaverListener subclass, for saving the checkpoint file |

## Returns

An object of the  **NPUCheckpointSaverHook**  class

## Restrictions

When  **NPUEstimator**  is used and  **iterations_per_loop**  is set to a value greater than 1, the hook may not take effect.

## Example

```python
from npu_bridge.npu_init import *
checkpoint_hook = NPUCheckpointSaverHook(checkpoint_dir='./ckpt', save_steps=2000)
...
mnist_classifier.train(
   input_fn=train_input_fn,
    steps=2000,
    hooks=[checkpoint_hook])
```
