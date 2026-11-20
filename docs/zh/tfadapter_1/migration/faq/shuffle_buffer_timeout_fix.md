# 数据集shuffle数量过大导致训练超时

## 问题现象

训练执行报错：

```text
[2020-11-27 11:26:00.510219: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 2169 of 10000
[2020-11-27 11:26:10.454132: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 3252 of 10000
[2020-11-27 11:26:20.375176: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 3915 of 10000
[2020-11-27 11:26:30.543144: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 4672 of 10000
[2020-11-27 11:26:40.479843: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 5439 of 10000
[2020-11-27 11:26:50.496244: I tensorflow/core/kernels/data/shuffle_dataset_op.cc:145] Filling up shuffle buffer (this may take a while): 6232 of 10000
[2020-11-27 11:26:50.638388: W tensorflow/core/framework/op_kernel.cc:1639] Unavailable: Internal errors
[2020-11-27 11:26:53.638664: F tf_adapter/kernels/geop_npu.cc:669] GeOp33_0GEOP::::DoRunAsync Failed
[ERROR] RUNTIME(62299)model execute error, retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(62299)model execute task failed, device_id=0, model stream_id=575, model task_id=1, model_id=522, first_task_id=65535
Fatal Python error: Aborted
```

## 原因分析

当开启GetNext算子下沉时，NPU采用预处理与前后向运算并行的方式工作。此时如果预处理过程对数据进行了shuffle且shuffle数量过大，则可能在前向计算任务下发很长时间后，预处理仍然无法输出有效数据，导致前向计算任务超时。

以shuffle数量设置为10000引发的任务超时为例，通过上述日志信息可以看到，在前向计算任务超时时，buffer仍未获取到足够的数据（仅获取到了6232个数据），从而出现task超时的错误打印。

## 解决方案

开发者可以采取以下策略解决：

可以根据超时时间内实际完成的shuffle数量，适当减少shuffle的数量。例如本例中实际完成的shuffle数量为6232个数据，此处配置为5000。

```python
dataset = dataset.shuffle(buffer_size=5000)
```

也可以将enable_data_pre_proc设置为False，关闭GetNext算子下沉开关，从而保证预处理与前后向运算串行，但是可能会有性能上的损失。

```python
run_config = NPURunConfig(enable_data_pre_proc=False)
```
