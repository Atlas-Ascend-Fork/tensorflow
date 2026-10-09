# Experiment Options

The experiment options are extended options for debugging and may be changed in later versions. Therefore, they cannot be used in production environments.

## graph_compiler_cache_dir

Drive cache directory for graph compilation. If this parameter is not empty, the drive cache function for graph compilation takes effect.

The graph compilation cache function supports drive persistence of graph compilation results. When graph compilation is performed again, the compilation results cached on the drive can be directly loaded to reduce the graph compilation duration.

Note:

- The configured cache directory must exist. Otherwise, the compilation fails.
- During graph compilation, the cache file is determined based on the value of this parameter. If the cache file does not exist, the cache is saved. If the cache file exists, the existing cache is directly loaded.
- After a graph is changed, the original cache file is unavailable. You need to manually delete the cache file from the cache directory or rebuild and generate a cache file.
- The cache does not ensure cross-version compatibility. If the version is upgraded, clear the cache directory and rebuild and generate the cache.
- This function does not support models with resource operators.

Example:

```python
npu.global_options().graph_compiler_cache_dir="/rootbuild_cache_dir"
```

## jit_compile

Whether to preferentially perform online model compilation.

- auto (default): For a static shape network, compile the operator online. For a dynamic shape network, search for the compiled operator binary file in the system first. If the corresponding binary file is not available, compile the operator.
- true: Operators are compiled online. The system performs fusion and tuning based on the obtained graph information to get better performing operators.
- false: The compiled operator binary file in the system is preferentially searched. If the file can be found, operators are not compiled anymore, which produces better compilation performance. If the file cannot be found, operators will be compiled.

> [!NOTE]NOTE:
> This option is used only for networks of large recommendation models.

Example:

```python
npu.global_options().jit_compile = "auto"
```

## shape_generalization_mode

When jit_compile is set to true (online operator compilation), use this parameter to configure the shape generalization mode.

- STRICT (default): Uses the shape of the current iteration as is, without any generalization.
- FULL: Generalizes all axes to -1 if the shape changes between iterations.
- ADAPTIVE: Generalizes only the shape of the changed axis to -1 if the shape changes between iterations. The newly generalized axis triggers model recompilation, which may cause the model to be compiled multiple times under this configuration.

> [!NOTE]NOTE
> When [compile_dynamic_mode](./dynamic_shape.md#compile_dynamic_mode) is set to True, the first iteration generalizes all input shapes to -1, and the shape_generalization_mode setting does not take effect.

Configuration example:

```python
npu.global_options().shape_generalization_mode = "FULL"
```

## auto_multistream_parallel_mode

This option applies to static and dynamic shape graph scenarios. You can enable parallel execution of Cube and Vector operators to improve graph execution performance.

- **cv**, Parallel execution of Cube and Vector operators is enabled.
- **LoadBalance:n**, Load balancing algorithm that distributes all operators evenly across n streams for execution. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- **MainStream:n**, Main stream algorithm that executes serial operators on the main stream, while other parallelizable operators are distributed across other streams. Here, n represents the maximum number of streams, which must be a positive integer within the range [1, 64]. If n exceeds the number of available cores, performance may degrade.
- The default value is empty, meaning Cube and Vector operators are executed serially.

> [!NOTE]NOTE
>
> - This option is used only for recommendation networks.
> - To use this function in dynamic shape multi-stream mode, you need to first enable dynamic shape multi-stream by setting the environment variable ENABLE_DYNAMIC_SHAPE_MULTI_STREAM, and then configure this option. For details about environment variables, see [Environment Variables](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/envvar/envref_07_0001.html).

Configuration example:

```python
npu.global_options().auto_multistream_parallel_mode = "cv"
```
