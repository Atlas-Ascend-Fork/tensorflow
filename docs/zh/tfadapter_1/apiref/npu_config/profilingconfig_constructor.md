# ProfilingConfig构造函数

## 功能说明

ProfilingConfig类的构造函数，用于配置Profiling功能。

## 函数原型

```python
def __init__(self,
enable_profiling=False,
profiling_options=None
)
```

## 参数说明

- **enable_profiling**：输入，是否开启Profiling功能。
  - True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
  - False：关闭Profiling功能，默认关闭。

- **profiling_options**：输入，Profiling配置选项。

  支持的配置选项可参见环境变量[PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/env-vars/PROFILING_OPTIONS.md)。

    配置示例：

    ```python
    profiling_options = '{"output":"/tmp/profiling","training_trace":"on","task_trace":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
    ```

## 返回值

返回ProfilingConfig类对象，作为NPURunConfig的参数传入。

## 约束说明

无

## 调用示例

```python
from npu_bridge.npu_init import *
...
profiling_options = '{"output":"/home/test/output","task_trace":"on"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```
