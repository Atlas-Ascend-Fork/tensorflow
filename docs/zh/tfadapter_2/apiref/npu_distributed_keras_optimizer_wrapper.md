# npu.distribute.npu_distributed_keras_optimizer_wrapper

## 功能说明

在更新梯度前，添加NPU的allreduce操作对梯度进行聚合，然后再更新梯度。该接口仅在分布式场景下使用。

## 函数原型

```python
def npu_distributed_keras_optimizer_wrapper(optimizer, reduce_reduction="mean", fusion=1, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| optimizer | 输入 | TensorFlow梯度训练优化器。 |
| reduce_reduction | 输入 | String类型。<br>聚合运算的类型，可以为"mean","max","min","prod"或"sum"。 |
| fusion | 输入 | int类型。<br>allreduce算子融合标识，支持以下取值：<br>  - 0：网络编译时，不会对该算子进行融合，即该allreduce算子不和其他allreduce算子融合。<br>  - 1：网络编译时，对该算子按照梯度切分策略进行融合。<br>  - 2：网络编译时，对allreduce算子按照相同的fusion_id进行融合，即“fusion_id”相同的allreduce算子之间会进行融合。 |
| fusion_id | 输入 | int类型。<br>allreduce算子的融合id。<br>当“fusion”取值为“2”时，网络编译时会对相同fusion_id的allreduce算子进行融合。 |
| group | 输入 | String类型，最大长度为128字节，含结束符。<br>group名称，可以为用户自定义group或者"hccl_world_group"。 |

## 返回值

返回输入的optimizer。

## 调用示例

```python
import npu_device as npu
optimizer = tf.keras.optimizers.SGD()
optimizer = npu.distribute.npu_distributed_keras_optimizer_wrapper(optimizer) # 使用NPU分布式计算，更新梯度
model.compile(optimizer = optimizer)
```
