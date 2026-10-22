# Data Augmentation

## local_rank_id

Rank ID of the current process, used in data parallel processing in recommendation networks. The main process deduplicates the data and distributes the result to the devices of other processes for forward and backward propagation.

In this mode, multiple devices on a host share one main process for data preprocessing, leaving other processes to receive preprocessed data from the main process.

To identify the main process, call the collective communication API get_local_rank_id() to get the rank ID of the current process on its server.

Example:

```python
config = NPURunConfig(local_rank_id=0, local_device_list="0,1")
```

## local_device_list

Devices that the main process sends data to, used in conjunction with local_rank_id.

```python
config = NPURunConfig(local_rank_id=0, local_device_list="0,1")
```
