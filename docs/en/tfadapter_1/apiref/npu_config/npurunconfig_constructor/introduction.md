# Overview

## Description

When performing model training or online inference in  **Estimator**  mode on the  AI processor, you can use the constructor of the  **NPURunConfig**  class to specify the running configuration of the Estimator.

The  **NPURunConfig**  class inherits the  **RunConfig**  class of  **tf.estimator**. For details about the support for native APIs of the  **RunConfig**  class, see  [Supported RunConfig Parameters](../runconfig_params_support_info.md).

## Prototype

You can view the  **NPURunConfig**  prototype definition in the  **python/site-packages/npu_bridge/estimator/npu/npu_config.py**  file in the TensorFlow Adapter installation directory. The following is an example:

```python
class NPURunConfig(run_config_lib.RunConfig):
    def __init__(self,
                 iterations_per_loop=1,
                 profiling_config=None,
                 model_dir=None,
                 tf_random_seed=None,
                 save_summary_steps=0,
                 save_checkpoints_steps=None,
                 save_checkpoints_secs=None,
                 ...
                 )
```

For details about the parameters supported by  **NPURunConfig**, see the parameter description in the following sections.

## Restrictions

In multi-device training scenarios, the  **save_checkpoints_secs**  parameter cannot be used to save files by time.

## Returns

An object of the  **NPURunConfig**  class, as the initialization argument passed to the  **NPUEstimator**  call.

## Example

The usage of  **NPURunConfig**  configurations is as follows:

```python
from npu_bridge.npu_init import *
session_config=tf.ConfigProto()
config = NPURunConfig(
    session_config=session_config,
    mix_compile_mode=False,
    iterations_per_loop=1000)
```
