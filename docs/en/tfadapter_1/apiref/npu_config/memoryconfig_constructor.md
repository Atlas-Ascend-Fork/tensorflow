# MemoryConfig Constructor

## Description

Constructs an object of class  **MemoryConfig**  for configuring system memory handling.

## Prototype

```python
class MemoryConfig():
    def __init__(self,
                 atomic_clean_policy=0,
                 static_memory_policy=0,
                 memory_optimization_policy=None,
                 variable_use_1g_huge_page=0):
        ...
```

## Parameters

- **atomic_clean_policy**: input, whether to clean up the memory occupied by all operators with the  **memset**  attribute \(memset operators\) on the network.
  - **0**  \(default\): enables collective cleanup.
  - **1**: disables collective cleanup. Memory used by each memset operator is cleaned up separately. When the memset operators on the network occupy too much memory, you can try this method. However, this method may cause performance loss.

- **static_memory_policy**: input, memory allocation mode used during network running.

  - **0**  \(default\): dynamic memory allocation. Memory is dynamically allocated based on the actual size.
  - **2**: dynamic memory expansion supported by only static shape. During network running, this option can be used to implement memory reuse between multiple graphs in a session. That is, the memory required by the maximum graph is allocated. For example, if the memory required by the current graph exceeds the memory of the previous graph, the memory of the previous graph is directly released. The memory is reallocated based on the memory required by the current graph.
  - **3**: dynamic memory expansion supported by only dynamic shape, which solves the fragment problem during dynamic memory allocation and reduces the memory usage of the dynamic-shape network.
  - **4**: dynamic memory expansion supported by both static and dynamic shapes.

    > [!NOTE]NOTE
    >
    > - This option cannot be set to  **2**  or  **4**  when multiple graphs are executed concurrently.
    > - To be compatible with earlier versions, the system adopts the method of mode 2 even if this option is set to  **1**.
    > - If this option is set to  **3**  or  **4**, memory gains are generated, but performance may deteriorate.

- **memory_optimization_policy**: input, optional memory optimization policy. The default is None, which leaves this option unset in NpuOptimizer. This is not the hugepage option; 1 GB huge pages are controlled by `variable_use_1g_huge_page`, whose values are described below. Prefer keyword arguments, for example `MemoryConfig(variable_use_1g_huge_page=1)`, to avoid positional argument confusion.

- **variable_use_1g_huge_page**: input. In recommendation models, the embedding layer in TensorFlow uses variables. When embedding layers serve as input or output addresses for index-based operators \(such as Gather and ScatterNd\), large memory footprints may lead to extensive scattered access, potentially causing performance degradation. In this case, you can set this parameter to allocate 1 GB hugepage memory for variables and constants to improve memory access performance.

    The values are as follows:

  - **0**  \(default\): uses the system default page size \(4 KB or 2 MB\) for memory allocation.
  - **1**: allocates memory using 1 GB huge pages. If the allocation fails, an error log is printed and the service terminates.
  - **2**: allocates memory using 1 GB huge pages. If the allocation fails, an error log is printed, but the service does not terminate; instead, it falls back to 2 MB pages. If the fallback allocation succeeds, the service continues; if it also fails, the service terminates.

    Using 1 GB huge pages can effectively reduce the number of page table entries and expand the address range covered by the translation lookaside buffer \(TLB\) cache, thereby improving performance for scattered access patterns. The TLB is a hardware module on the Ascend AI processor that caches recently used virtual-to-physical address mappings.

    > [!NOTE]NOTE
    >This parameter is supported only on the following products:
    >
    >- Ascend 950PR&950DT products
    >- Atlas A3 products
    >- Atlas A2 products

## Returns

An object of the  **MemoryConfig**  class, as an argument passed to the  **NPURunConfig**  call.

## Restrictions

None

## Example

```python
from npu_bridge.npu_init import *
...
# Use 1 GB huge pages; terminate the service if allocation fails.
mem_config = MemoryConfig(atomic_clean_policy=0,
                          static_memory_policy=0,
                          variable_use_1g_huge_page=1)
session_config=tf.ConfigProto(allow_soft_placement=True)
config = NPURunConfig(memory_config=mem_config, session_config=session_config)
```
