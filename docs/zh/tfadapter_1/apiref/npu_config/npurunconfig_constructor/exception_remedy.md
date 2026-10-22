# 异常补救

## hccl_timeout

设备间任务执行的同步等待时间，单位为s。

当默认时长不满足需求时（例如出现通信失败的错误），可通过此配置项延长超时时间。

- 针对Ascend 950PR&950DT系列产品，单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，当配置为0时代表永不超时。
- 针对Atlas A3系列产品，单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，当配置为0时代表永不超时。
- 针对Atlas A2系列产品，单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，当配置为0时代表永不超时。
- 针对Atlas训练系列产品，单位为s，取值范围为：\(0, 17340\]，默认值为1836。

  **需要注意：针对Atlas训练系列产品，系统实际设置的超时时间 = 参数取值先整除“68”，再乘以“68”，单位s。如果参数取值小于68，则默认按照68s进行处理。**

  例如，假设“hccl_timeout”配置为600，则系统实际设置的超时时间为：600整除68乘以68 = 8\*68 = 544s。

- 针对Atlas 300I Duo推理卡，单位为s，取值范围为：\(0, 17340\]，默认值为1836。

  **需要注意：针对Atlas 300I Duo推理卡，系统实际设置的超时时间 = 参数取值先整除“68”，再乘以“68”，单位s。如果参数取值小于68，则默认按照68s进行处理。**

  例如，假设“hccl_timeout”配置为600，则系统实际设置的超时时间为：600整除68乘以68 = 8\*68 = 544s。

> [!NOTE]说明
> 参数“hccl_timeout”的优先级大于环境变量“HCCL_EXEC_TIMEOUT”，若同时配置了参数“hccl_timeout”与环境变量“HCCL_EXEC_TIMEOUT”，以参数“hccl_timeout”的配置值为准。关于环境变量“HCCL_EXEC_TIMEOUT”的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

配置示例：

```python
config = NPURunConfig(hccl_timeout=1800)
```

## op_wait_timeout

算子等待超时时间，单位为s，默认值为120s。

配置示例：

```python
config = NPURunConfig(op_wait_timeout=120)
```

## op_execute_timeout

算子执行超时时间，单位为s。

配置示例：

```python
config = NPURunConfig(op_execute_timeout=90)
```

## stream_sync_timeout

图执行时，stream同步等待超时时间，超过配置时间时报同步失败。单位：ms

默认值-1，表示无等待时间，出现同步失败不报错。

**说明：集群场景下，此配置的值（即stream同步等待超时时间）需要大于集合通信超时时间，即“hccl_timeout”配置项的值或者环境变量“HCCL_EXEC_TIMEOUT”的值。**

配置示例：

```python
config = NPURunConfig(stream_sync_timeout=60000)
```

## event_sync_timeout

图执行时，event同步等待超时时间，超过配置时间时报同步失败。单位：ms

默认值-1，表示无等待时间，出现同步失败不报错。

配置示例：

```python
config = NPURunConfig(event_sync_timeout=60000)
```
