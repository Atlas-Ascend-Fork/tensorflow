# Memory Management

## memory_config

System memory usage mode. Before creating NPURunConfig, you can instantiate a MemoryConfig class to configure functions. For details about the constructor of the MemoryConfig class, see [MemoryConfig Constructor](../memoryconfig_constructor.md).

## external_weight

When multiple models are loaded in a session, if the weights of these models can be reused, you are advised to use this configuration item to externalize the weights of the Const/Constant nodes on the network to implement weight reuse among multiple models and reduce the memory usage of the weights.

- False (default): The weights are not externalized but are saved in graphs.
- True: The weights are externalized, the weights of all Const/Constant nodes on the network are flushed to the disk, and the node type is converted to FileConstant. The weight file is named in the format of weight_\<hash value\>.

If the environment variable ASCEND_WORK_PATH is not configured in the environment, the weight files are flushed to the current execution directory tmp_weight_<pid\>_<sessionid\>.

If ASCEND_WORK_PATH is configured in the environment, the weight files are flushed to the ${ASCEND_WORK_PATH}/tmp_weight_<pid\>_<sessionid\> directory. For details about ASCEND_WORK_PATH, see Installation and Configuration in [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

When the model is uninstalled, the tmp_weight_<pid\>_<sessionid\> directory is automatically deleted.

Note: This parameter is usually not required. If the model loading environment has limitations on memory, you can flush the weight externally.

Example:

```python
config = NPURunConfig(external_weight=True)
```

## input_fusion_size

Threshold for fusing and copying multiple discrete pieces of user input data during data transfer from the host to the device. The unit is byte. The minimum value is 0 byte, the maximum value is 33554432 bytes (32 MB), and the default value is 131072 bytes (128 KB). If:

- Size of input data ≤ threshold: The data is fused before being transferred from the host to the device.
- Size of input data > threshold or threshold = 0 (function disabled): The data is not fused before being transferred from the host to the device.

Assume there are 10 user inputs, including two 100 KB inputs, two 50 KB inputs, and the other inputs greater than 100 KB:

- input_fusion_size set to 100KB: The preceding four inputs are fused into 300 KB data for transfer. The other six inputs are directly transferred from the host to the device.
- input_fusion_size set to 0KB: This function is disabled. That is, the data is not fused, and the ten inputs are directly transferred from the host to the device.

Note: This parameter takes effect only for static shape graphs.

Example:

```python
config = NPURunConfig(input_fusion_size=25600)
```

## input_batch_cpy

Whether to enable the batch memory copy function when input data is transferred from the host to the device.

- True: The batch memory copy function is enabled. This value takes effect only when the number of user inputs is greater than 1.
- False (default): The batch memory copy function is disabled.

NOTE:

- This parameter is supported only on the following products:
  - Ascend 950PR&950DT products
  - Atlas A3 products
  - Atlas A2 products
- This parameter improves data transfer performance from the host to the device. It applies to scenarios that require frequent data transfer and have low PCIe bandwidth utilization. Enabling the batch copy function using this parameter can improve bandwidth utilization.
- If the network initially has only one input, the batch copy function does not take effect even if it is enabled.
- When both the input_fusion_size parameter (for enabling fusion and copy) and the input_batch_cpy parameter (for enabling batch copy) are configured, the threshold for the fusion and copy function may affect the batch copy function.

For example, if there are five inputs and four of them are smaller than the threshold for fusion and copy and meet the fusion conditions, these four inputs will be processed using fusion and copy. The remaining input does not meet the input quantity requirement for batch copy and therefore will not be batch-copied.

Example:

```python
config = NPURunConfig(input_batch_cpy=True)
```
