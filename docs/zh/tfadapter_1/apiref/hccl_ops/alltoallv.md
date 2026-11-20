# alltoallv

## 功能说明

集合通信算子AlltoAllV操作接口，向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。

![](../figures/alltoallv.png)

## 函数原型

```python
def all_to_all_v(send_data, send_counts, send_displacements, recv_counts, recv_displacements, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| send_data | 输入 | 待发送的数据，TensorFlow的tensor类型。<br>针对Ascend 950PR&950DT系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A3系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64 。 |
| send_counts | 输入 | 发送的数据量，send_counts[i]表示本rank发给rank i的数据个数，基本单位为send_data数据类型对应的元素个数。<br>例：send_data的数据类型为int32，send_counts[0]=1,send_counts[1]=2，表示本rank给rank0发送1个int32类型的数据，给rank1发送2个int32类型的数据。<br>TensorFlow的tensor类型，tensor支持的数据类型为int64。 |
| send_displacements | 输入 | 发送数据的偏移量，send_displacements[i]表示本rank发送给rank i的数据块相对于send_data的偏移量，基本单位为send_data数据类型对应的元素个数。<br>例：<br><br>  - send_data的数据类型为int32。<br>  - send_counts[0]=1,send_counts[1]=2<br>  - send_displacements[0]=0,send_displacements[1]=1<br><br>则表示本rank给rank0发送send_data上的第1个int32类型的数据，给rank1发送send_data上第2个与第3个int32类型的数据。<br>TensorFlow的tensor类型，tensor支持的数据类型为int64。 |
| recv_counts | 输入 | 接收的数据量，recv_counts[i]表示本rank从rank i收到的数据量。使用方法与send_counts类似。<br>TensorFlow的tensor类型。tensor支持的数据类型为int64。 |
| recv_displacements | 输入 | 接收数据的偏移量，recv_displacements[i]表示本rank从rank i接收的数据块相对于接收数据起始地址的偏移量，基本单位为接收数据数据类型对应的元素个数。使用方法与send_displacements类似。<br>TensorFlow的tensor类型。tensor支持的数据类型为int64。 |
| group | 输入 | group名称，可以为用户自定义group或者"hccl_world_group"。<br>String类型，最大长度为128字节，含结束符。 |

## 返回值

对输入tensor执行完all_to_all_v操作之后的结果tensor。

## 约束说明

1. 调用该接口的rank必须在当前接口入参group定义的范围内，不在此范围内的rank调用该接口会失败。
2. 针对Atlas训练系列产品，alltoallv的通信域需要满足如下约束：

    集群组网下，单Server 1p、2p通信域要在同一个cluster内（Server内0-3卡和4-7卡各为一个cluster），单Server 4p、8p和多Server通信域中rank要以cluster为基本单位，并且Server间cluster选取要一致。

3. alltoallv操作的性能与NPU之间共享数据的缓存区大小有关，当通信数据量超过缓存区大小时性能将出现明显下降。若业务中alltoallv通信数据量较大，建议通过配置环境变量HCCL_BUFFSIZE适当增大缓存区大小以提升通信性能，关于环境变量HCCL_BUFFSIZE的介绍可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
4. 针对Atlas训练系列产品，如果是单Server场景，要求网卡的状态是“up”，否则此接口会执行失败。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
send_data_tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
send_counts_tensor = tf.constant([3,3],dtype=tf.int64)
send_displacements_tensor = tf.constant([0,0],dtype=tf.int64)
recv_counts_tensor = tf.constant([3,3],dtype=tf.int64)
recv_displacements_tensor = tf.constant([0,0],dtype=tf.int64)
result = hccl_ops.all_to_all_v(send_data_tensor,send_counts_tensor,send_displacements_tensor,recv_counts_tensor,recv_displacements_tensor)
```
