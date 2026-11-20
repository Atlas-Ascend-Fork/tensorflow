# Profiling

## profiling_mode

Enables profiling.

- True: enabled. The profiling options are determined by profiling_options.
- False (default): disabled.

Example:

```python
custom_op.parameter_map["profiling_mode"].b = True
```

Note: The priority of this configuration item is higher than that of the environment variable PROFILING_MODE. For details about the environment variable, see Profile Data Collection in [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

## profiling_options

Profiling configuration options. For supported options, refer to the environment variable [PROFILING_OPTIONS](https://gitcode.com/cann/oam-tools/blob/master/docs/en/user_guide/env-vars/PROFILING_OPTIONS.md).

Example:

```python
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/tmp/profiling","training_trace":"on","fp_point":"","bp_point":""}')
```
