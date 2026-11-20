# Profiling

## profiling_mode

是否开启Profiling功能。

- True：开启Profiling功能，从profiling_options读取Profiling的采集选项。
- False（默认值）：关闭Profiling功能。

配置示例：

```python
custom_op.parameter_map["profiling_mode"].b = True
```

说明：此配置项的优先级高于环境变量PROFILING_MODE，关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“性能数据采集”章节。

## profiling_options

Profiling配置选项。支持的配置选项可参见环境变量[PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/user_guide/env-vars/PROFILING_OPTIONS.md)。

配置示例：

```python
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/tmp/profiling","training_trace":"on","fp_point":"","bp_point":""}')
```
