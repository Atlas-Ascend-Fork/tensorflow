# alltoallvc

## 功能说明

集合通信alltoallvc操作接口。向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。

alltoallvc通过输入参数send_count_matrix传入所有rank的收发参数，与[alltoallv](alltoallv.md)相比，性能更优。

![](../figures/alltoallvc.png)

## 函数原型

```python
def all_to_all_v_c(send_data, send_count_matrix, rank, fusion=0, fusion_id=-1, group="hccl_world_group")
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| send_data | 输入 | 待发送的数据，TensorFlow的tensor类型。<br>针对Ascend 950PR&950DT系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A3系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64 。 |
| send_count_matrix | 输入 | 所有rank的收发参数，send_count_matrix[i][j]表示rank i发给rank j的数据量，基本单位为send_data数据类型对应的元素个数。<br>例：send_data的数据类型为int32，send_count_matrix[0][1]=1，表示rank0给rank1发送1个int32。<br>TensorFlow的tensor类型。tensor支持的数据类型为int64。 |
| rank | 输入 | 本节点的rank id，该id是group内的rank id，int类型。 |
| fusion | 输入 | alltoallvc算子融合标识，int类型，支持以下取值：<br><br>  - 0：标识网络编译时不会对该算子进行融合，即该alltoallvc算子不和其他alltoallvc算子融合。<br>  - 2：网络编译时，会对alltoallvc算子按照相同的fusion_id进行融合，即“fusion_id”相同的alltoallvc算子之间会进行融合。说明：“fusion_id”相同的alltoallvc算子之间融合有一定的前提，算子需要在相同的通信域内，并且算子发送数据的数据类型需要相同。 |
| fusion_id | 输入 | 标识alltoallvc算子的融合id，int类型。<br>开启alltoallvc算子融合功能的场景下，需要配置该参数，取值范围[0, 0x7fffffff]。 |
| group | 输入 | group名称，可以为用户自定义group或者"hccl_world_group"。<br>String类型，最大长度为128字节，含结束符。 |

## 返回值

对输入tensor执行完all_to_all_v_c操作之后的结果tensor。

## 约束说明

1. 调用该接口的rank必须在当前接口入参group定义的范围内，输入的rank id有效且不重复，否则调用该接口会失败。
2. 针对Atlas训练系列产品，alltoallvc的通信域需要满足如下约束：

    单Server 1p、2p通信域要在同一个cluster内（Server内0-3卡和4-7卡各为一个cluster），单Server 4p、8p和多Server通信域中rank要以cluster为基本单位，并且Server间cluster选取要一致。

3. 各节点输入的send_count_matrix要保持一致。
4. alltoallvc操作的性能与NPU之间共享数据的缓存区大小有关，当通信数据量超过缓存区大小时性能将出现明显下降。若业务中alltoallvc通信数据量较大，建议通过配置环境变量HCCL_BUFFSIZE适当增大缓存区大小以提升通信性能，关于环境变量HCCL_BUFFSIZE的介绍可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。
5. 针对Atlas训练系列产品，如果是单Server场景，要求网卡的状态是“up”，否则此接口会执行失败。

## 调用示例

```python
from npu_bridge.hccl import hccl_ops
send_data_tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
send_counts_matrix_tensor = tf.Variable( [[3,3],[3,3]], dtype=tf.int64)
all_to_all_v_c = hccl_ops.all_to_all_v_c(send_data_tensor, send_counts_matrix_tensor, 0)
```
