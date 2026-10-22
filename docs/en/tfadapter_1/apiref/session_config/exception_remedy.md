# Exception Remedy

## hccl_timeout

Synchronization timeout for inter-device task execution, in seconds.

You can set the timeout interval if the default value does not meet your requirement (for example, when a communication failure occurs).

- For the Ascend 950PR&950DT products, the value range is [0, 2147483647], in seconds. The default value is 1836. The value 0 indicates that the session never times out.
- For the Atlas A3 products, the value range is [0, 2147483647], in seconds. The default value is 1836. The value 0 indicates that the session never times out.
- For the Atlas A2 products, the value range is [0, 2147483647], in seconds. The default value is 1836. The value 0 indicates that the session never times out.
- For the Atlas training products, the value range is (0, 17340], in seconds. The default value is 1836.

  Note: For the Atlas training products, the actual timeout interval set in the system = (Value of this parameter // 68) × 68 (unit: s). If the parameter value is less than 68, 68s is used by default.

  For example, if hccl_timeout is set to 600, the actual timeout interval set in the system is 544s (600 // 68 × 68 = 8 × 68).

- For the Atlas 300I Duo Inference Card, the value range is (0, 17340], in seconds. The default value is 1836.

  Note: For the Atlas 300I Duo Inference Card, actual timeout interval set in the system = (Value of this parameter // 68) × 68 (unit: s). If the parameter value is less than 68, 68s is used by default.

  For example, if hccl_timeout is set to 600, the actual timeout interval set in the system is 544s (600 // 68 × 68 = 8 × 68).

> [!NOTE]NOTE
> The priority of hccl_timeout supersedes that of the environment variable HCCL_EXEC_TIMEOUT. If both hccl_timeout and HCCL_EXEC_TIMEOUT are configured, hccl_timeout is used. For details about HCCL_EXEC_TIMEOUT, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Example:

```python
custom_op.parameter_map["hccl_timeout"].i = 1800
```

## op_wait_timeout

Operator wait timeout interval (s). Defaults to 120. You can set the timeout interval if the default value does not meet your requirement.

Example:

```python
custom_op.parameter_map["op_wait_timeout"].i = 120
```

## op_execute_timeout

Operator execution timeout interval (s).

Example:

```python
custom_op.parameter_map["op_execute_timeout"].i = 90
```

## stream_sync_timeout

Timeout interval for stream synchronization during graph execution. If the timeout interval exceeds the configured value, a synchronization failure is reported. The unit is ms.

The default value is -1, indicating that there is no waiting time and no error is reported when the synchronization fails.

Note: In cluster scenarios, the value of this parameter (timeout interval for stream synchronization) must be greater than the collective communication timeout interval, that is, the value of hccl_timeout or the environment variable HCCL_EXEC_TIMEOUT.

Example:

```python
custom_op.parameter_map["stream_sync_timeout"].i = 60000
```

## event_sync_timeout

Timeout interval for event synchronization during graph execution. If the timeout interval exceeds the configured value, a synchronization failure is reported. The unit is ms.

The default value is -1, indicating that there is no waiting time and no error is reported when the synchronization fails.

Example:

```python
custom_op.parameter_map["event_sync_timeout"].i = 60000
```
