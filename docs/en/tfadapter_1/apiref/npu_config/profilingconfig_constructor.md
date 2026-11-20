# ProfilingConfig Constructor

## Description

Constructs an object of class  **ProfilingConfig**  as the profiling configuration.

## Prototype

```python
def __init__(self,
enable_profiling=False,
profiling_options=None
)
```

## Parameters

- **enable_profiling**: input, whether to enable profiling.
  - **True**: enabled. The profiling options are determined by  **profiling_options**.
  - **False**  \(default\): disabled.

- **profiling_options**: input, profiling configuration options.

  For supported options, refer to the environment variable [PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/en/user_guide/env-vars/PROFILING_OPTIONS.md).

   Example:

    ```python
    profiling_options = '{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
    ```

## Returns

An object of the  **ProfilingConfig**  class, as an argument passed to the  **NPURunConfig**  call.

## Restrictions

None

## Example

```python
from npu_bridge.npu_init import *
...
profiling_options = '{"output":"/home/test/output","task_trace":"on"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```
