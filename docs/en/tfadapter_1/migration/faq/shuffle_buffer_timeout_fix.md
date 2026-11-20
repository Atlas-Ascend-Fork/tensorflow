# What Do I Do If Training Times Out Due to Too Large Dataset Shuffle Buffer Size?

## Symptom

An error is reported during the training process.

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

## Possible Cause

When the GetNext operator is offloaded to the NPU, preprocessing is performed in parallel with forward and backward propagation. If the configured preprocessing shuffle buffer size is too large, preprocessing output will be unavailable in a long time after the forward propagation task is delivered. As a result, the forward propagation task times out.

Assume that the shuffle buffer size is set to  **10000**. According to the preceding log, when the forward propagation task times out, the buffer receives only 6232/10000 records. As a result, the task timeout error is printed.

## Solution

You can resolve this problem in either of the following ways:

Reduce the number of shuffle operations based on the actual capacity. In this example, only 6232 shuffle operations are complete. Therefore,  **buffer_size**  can be set to  **5000**.

```python
dataset = dataset.shuffle(buffer_size=5000)
```

Disable GetNext operator offloading by setting  **enable_data_pre_proc**  to  **False**  so that preprocessing and forward and backward propagation can be executed in serial. However, the performance may be compromised.

```python
run_config = NPURunConfig(enable_data_pre_proc=False)
```
