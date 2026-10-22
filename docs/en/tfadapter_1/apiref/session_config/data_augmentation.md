# Data Augmentation

## local_rank_id

Rank ID of the current process, used in data parallel processing. The main process deduplicates the data and distributes the result to the devices of other processes for forward and backward propagation.

![](../figures/local_rank_id.png)

In this mode, multiple devices on a host share one process for data preprocessing. Although this is still a multi-process scenario, data preprocessing is performed in the main process, and other processes no longer accept datasets on the current process, but only receive preprocessed data from the main process.

To identify the main process, call the collective communication API get_local_rank_id() to get the rank ID of the current process on its server.

Example:

```python
custom_op.parameter_map["local_rank_id"].i = 0
```

## local_device_list

Devices that the main process sends data to, used in conjunction with local_rank_id.

```python
custom_op.parameter_map["local_device_list"].s = tf.compat.as_bytes("0,1")
```
