# alltoallvc

## Function Description

Sends data \(with the customized data size\) to all ranks in the collective communicator and receives data from all ranks.

**alltoallvc**  passes the RX and TX parameters of all ranks through the argument  **send_count_matrix**, which outperforms  [alltoallv](alltoallv.md).

![](../figures/alltoallvc.png)

## Function Prototype

```python
def all_to_all_v_c(send_data, send_count_matrix, rank, fusion=0, fusion_id=-1, group="hccl_world_group")
```

## Parameters

| Option | Input/Output | Description |
| --- | --- | --- |
| send_data | Input | Data to be sent.<br>TensorFlow tensor type.<br>For the Ascend 950PR&950DT products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, and bfp16.<br>For the Atlas A3 products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas A2 products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>For the Atlas training products, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64. |
| send_count_matrix | Input | TX and RX parameters of all ranks. send_count_matrix[i][j] indicates the amount of data sent from rank i to rank j. The basic unit is the number of elements of the send_data data type.<br>For example, if the data type of send_data is set to int32 and send_count_matrix[0][1] is set to 1, then rank 0 sends one int32 to rank 1.<br>TensorFlow tensor type, with the data type of int64. |
| rank | Input | Int type.<br>Rank ID in the group. |
| fusion | Input | Int type.<br>alltoallvc operator fusion flag. The values are as follows:<br><br>  - 0: The alltoallvc operator is not fused with other alltoallvc operators during network compilation.<br>  - 2: alltoallvc operators with the same fusion_id are fused during network compilation.Note: The prerequisite for alltoallvc operator fusion is that the alltoallvc operators with the same fusion_id must be in the same communicator and the types of their sent data must be the same. |
| fusion_id | Input | alltoallvc operator fusion ID.<br>Int type.<br>This parameter needs to be configured when alltoallvc operator fusion is enabled. The value range is [0, 0x7fffffff]. |
| group | Input | Group name, which can be a user-defined value or hccl_world_group.<br>A string containing a maximum of 128 bytes, including the end character. |

## Returns

The result tensor

## Constraints

1. The rank that calls this API must be within the range defined by the argument group of the current API. The entered rank ID must be valid and unique. Otherwise, the API call fails.
2. For the  Atlas training products, the  **alltoallvc**  communicators must meet the following requirement:

    The communicators of 1p and 2p in a single server must be in the same cluster \(with devices 0–3 and devices 4–7 each belonging to a separate cluster\). In the communicators of 4p and 8p in a single server and multiple servers, the ranks must be based on the clusters, and the selected clusters in servers must be consistent.

3. The value of  **send_count_matrix**  on each node must be the same.
4. The performance of the  **alltoallvc**  operation is related to the size of the buffer for storing shared data between NPUs. When the communication data size exceeds the buffer size, the performance deteriorates significantly. If the amount of  **alltoallvc**  communication data is large, you are advised to increase the buffer size by setting the environment variable  **HCCL_BUFFSIZE**  to improve the communication performance. For details about  **HCCL_BUFFSIZE**, see  _[Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclenvref_07_0005.html)_.
5. For the  Atlas training products, if a single server is used, the NIC must be in the  **up**  state. Otherwise, this API fails to be executed.

## Example

```python
from npu_bridge.hccl import hccl_ops
send_data_tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
send_counts_matrix_tensor = tf.Variable( [[3,3],[3,3]], dtype=tf.int64)
all_to_all_v_c = hccl_ops.all_to_all_v_c(send_data_tensor, send_counts_matrix_tensor, 0)
```
