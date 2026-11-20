# Profiling

## profiling_config.enable_profiling

是否开启Profiling功能

- True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
- False（默认值）：关闭Profiling功能。

配置示例：

```python
npu.global_options().profiling_config.enable_profiling=True
```

说明：此配置项的优先级高于环境变量[PROFILING_MODE](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/env-vars/PROFILING_MODE.md)。

## profiling_config.profiling_options

Profiling配置选项，支持的配置选项可参见环境变量[PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/env-vars/PROFILING_OPTIONS.md)。

配置示例：

```python
npu.global_options().profiling_config.profiling_options = '{"output":"/tmp/profiling","training_trace":"on","fp_point":"resnet_model/conv2d/Conv2Dresnet_model/batch_normalization/FusedBatchNormV3_Reduce","bp_point":"gradients/AddN_70"}'
```
