# Memory Management

## memory_config.atomic_clean_policy

Whether to clean up the memory occupied by all operators with the memset attribute (memset operators) on the network. The values are as follows:

- 0 (default): enables collective cleanup.
- 1: disables collective cleanup. Memory used by each memset operator is cleaned up separately. When the memset operators on the network occupy too much memory, you can try this method. However, this method may cause performance loss.

Example:

```python
npu.global_options().memory_config.atomic_clean_policy=1
```

## memory_config.static_memory_policy

Memory allocation mode used during network running.

- 0 (default): dynamic memory allocation. Memory is dynamically allocated based on the actual size.
- 2: dynamic memory expansion supported by only static shape. During network running, this option can be used to implement memory reuse between multiple graphs in a session. That is, the memory required by the maximum graph is allocated. For example, if the memory required by the current graph exceeds the memory of the previous graph, the memory of the previous graph is directly released. The memory is reallocated based on the memory required by the current graph.
- 3: dynamic memory expansion supported by only dynamic shape, which solves the fragment problem during dynamic memory allocation and reduces the memory usage of the dynamic-shape network.
- 4: dynamic memory expansion supported by both static and dynamic shapes.

Configuration example:

```python
npu.global_options().memory_config.static_memory_policy=0
```

> [!NOTE]NOTE:
>
> - This option cannot be set to 2 or 4 when multiple graphs are executed concurrently.
> - To be compatible with earlier versions, the system adopts the method of mode 2 even if this option is set to 1.
> - If this option is set to 3 or 4, memory gains are generated, but performance may deteriorate.

## memory_config.variable_use_1g_huge_page

In recommendation models, the embedding layer in TensorFlow uses variables. When embedding layers serve as input or output addresses for index-based operators (such as Gather and ScatterNd), large memory footprints may lead to extensive scattered access, potentially causing performance degradation. In such cases, you can try configuring this parameter to allocate memory for variables and constants using 1 GB huge pages, thereby improving memory access performance.

The values are as follows:

- 0 (default): uses the system default page size (4 KB or 2 MB) for memory allocation.
- 1: allocates memory using 1 GB huge pages. If the allocation fails, an error log is printed and the service terminates.
- 2: allocates memory using 1 GB huge pages. If the allocation fails, an error log is printed, but the service does not terminate; instead, it falls back to 2 MB pages. If the fallback allocation succeeds, the service continues; if it also fails, the service terminates.

Using 1 GB huge pages can effectively reduce the number of page table entries and expand the address range covered by the translation lookaside buffer (TLB) cache, thereby improving performance for scattered access patterns. The TLB is a hardware module on the Ascend AI processor that caches recently used virtual-to-physical address mappings.

Configuration example:

```python
npu.global_options().memory_config.variable_use_1g_huge_page=1
```

> [!NOTE]NOTE:
> This parameter is supported only on the following products:
>
> - Ascend 950PR&950DT products
> - Atlas A3 products
> - Atlas A2 products

## external_weight

When multiple models are loaded in a session, if the weights of these models can be reused, you are advised to use this configuration item to externalize the weights of the Const/Constant nodes on the network to implement weight reuse among multiple models and reduce the memory usage of the weights.

- False (default): The weights are not externalized but are saved in graphs.
- True: The weights are externalized, the weights of all Const/Constant nodes on the network are flushed to the disk, and the node type is converted to FileConstant. The weight file is named in the format of weight_\<hash value\>.

  If the environment variable ASCEND_WORK_PATH is not configured in the environment, the weight files are flushed to the current execution directory tmp_weight_<pid\>_<sessionid\>.
  
  If ASCEND_WORK_PATH is configured in the environment, the weight files are flushed to the ${ASCEND_WORK_PATH}/tmp_weight_<pid\>_<sessionid\> directory.
  
  When the model is uninstalled, the tmp_weight_<pid\>_<sessionid\> directory is automatically deleted.
  
**Note**: This option is usually not required. If the model loading environment has limitations on memory, you can flush the weight externally.
  
Example:
  
```python
npu.global_options().external_weight=True
```

## input_fusion_size

Threshold for fusing and copying multiple discrete pieces of user input data during data transfer from the host to the device. The unit is byte. The minimum value is 0 byte, the maximum value is 33554432 bytes (32 MB), and the default value is 131072 bytes (128 KB). If:

- Size of input data ≤ threshold: The data is fused before being transferred from the host to the device.
- Size of input data > threshold or threshold = 0 (function disabled): The data is not fused before being transferred from the host to the device.

Assume there are 10 user inputs, including two 100 KB inputs, two 50 KB inputs, and the other inputs greater than 100 KB:

- input_fusion_size set to 100KB: The preceding four inputs are fused into 300 KB data for transfer. The other six inputs are directly transferred from the host to the device.
- input_fusion_size set to 0KB: This function is disabled. That is, the data is not fused, and the ten inputs are directly transferred from the host to the device.

**Note: This parameter takes effect only for static shape graphs.**

Example:

```python
npu.global_options().input_fusion_size=25600
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

Configuration example:

```python
npu.global_options().input_batch_cpy=True
```
