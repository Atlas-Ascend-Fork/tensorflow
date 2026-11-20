# alltoallv

## Function

Sends data \(with customizable size\) to all ranks in the communicator and receives data from all ranks.

![](../figures/alltoallv.png)

## Prototype

```python
def all_to_all_v(send_data, send_counts, send_displacements, recv_counts, recv_displacements, group="hccl_world_group")
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| send_data | Input | Data to be transmitted, which is of the TensorFlow tensor type.<br>Ascend 950PR&950DT products: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>Atlas A3 products: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>Atlas A2 products: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.<br>Atlas training products: The supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, and float64. |
| send_counts | Input | Size of sent data. send_counts[i] indicates the number of data pieces sent by the current rank to rank i. The basic unit is the number of elements of the send_data data type.<br>For example, if the data type of send_data is int32 and send_counts[0]=1,send_counts[1]=2, the current rank sends one int32 data segment to rank 0 and two int32 data segments to rank 1.<br>TensorFlow tensor type, with the data type of int64. |
| send_displacements | Input | Offset of the sent data. send_displacements[i] indicates the offset of the data block sent from the current rank to rank i relative to send_data. The basic unit is the number of elements of the send_data data type.<br>For example:<br><br>  - The data type of send_data is int32.<br>  - send_counts[0]=1,send_counts[1]=2<br>  - send_displacements[0]=0,send_displacements[1]=1<br><br>The current rank sends the first int32 data segment in send_data to rank 0, and sends the second and third int32 data segments in send_data to rank 1.<br>TensorFlow tensor type, with the data type of int64. |
| recv_counts | Input | Amount of received data. recv_counts[i] indicates the amount of data received by the current rank from rank i. The usage is similar to that of send_counts.<br>TensorFlow tensor type, with the data type of int64. |
| recv_displacements | Input | Offset of the received data. recv_displacements[i] indicates the offset of the data block received by the current rank from rank i relative to the start address of the received data. The basic unit is the number of elements of the received data data type. The usage is similar to that of send_displacements.<br>TensorFlow tensor type, with the data type of int64. |
| group | Input | Group name, which can be a user-defined value or hccl_world_group.<br>A string containing a maximum of 128 bytes, including the end character. |

## Returns

The result tensor after the  **all_to_all_v**  operation is performed on the input tensor.

## Restrictions

1. The caller rank must be within the range defined by the  **group**  argument passed to this API call. Otherwise, the API call fails.
2. For the  Atlas training products, the AlltoAllV communicators must meet the following requirement:

    In a cluster network, the communicators of 1p and 2p in a single server must be in the same cluster \(with devices 0–3 and devices 4–7 each belonging to a separate cluster\). In the communicators of 4p and 8p in a single server and multiple servers, the ranks must be based on the clusters, and the selected clusters in servers must be consistent.

3. The performance of the AlltoAllV operation is related to the size of the buffer for storing shared data between NPUs. When the communication data size exceeds the buffer size, the performance deteriorates significantly. If the amount of AlltoAllV communication data is large, you are advised to increase the buffer size by setting the environment variable  **HCCL_BUFFSIZE**  to improve the communication performance. For details about  **HCCL_BUFFSIZE**, see  [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclenvref_07_0005.html).
4. Atlas training products: If a single server is used, the NIC must be in the  **up**  state. Otherwise, this API fails to be executed.

## Example

```python
from npu_bridge.hccl import hccl_ops
send_data_tensor = tf.random_uniform((1, 3), minval=1, maxval=10, dtype=tf.float32)
send_counts_tensor = tf.constant([3,3],dtype=tf.int64)
send_displacements_tensor = tf.constant([0,0],dtype=tf.int64)
recv_counts_tensor = tf.constant([3,3],dtype=tf.int64)
recv_displacements_tensor = tf.constant([0,0],dtype=tf.int64)
result = hccl_ops.all_to_all_v(send_data_tensor,send_counts_tensor,send_displacements_tensor,recv_counts_tensor,recv_displacements_tensor)
```
