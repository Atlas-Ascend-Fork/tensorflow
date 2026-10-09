# 试验参数

试验参数为调试功能扩展参数，后续版本可能会存在变更，不支持应用于生产环境。

## graph_compiler_cache_dir

该参数用于配置图编译磁盘缓存目录，当该参数配置为非空时，图编译磁盘缓存功能生效。

图编译缓存功能支持将图编译结果进行磁盘持久化，当再次执行图编译运行时，直接加载磁盘上缓存的编译结果，从而减少图编译时长。

需要注意：

- 配置的缓存目录必须存在，否则会导致编译失败。
- 图编译时，会根据此参数的值确定缓存文件，若缓存文件不存在则保存缓存若缓存文件存在则直接加载缓存。
- 图发生变化后，原来的缓存文件不可用，用户需要手动删除缓存目录中的缓存文件，然后重新编译生成缓存文件。
- 缓存不保证跨版本的兼容性，如果版本升级，需要清理缓存目录重新编译生成缓存。
- 该功能当前不支持带资源类算子的模型。

配置示例：

```python
npu.global_options().graph_compiler_cache_dir="/rootbuild_cache_dir"
```

## jit_compile

模型编译时是否优先在线编译。

- auto（默认值）：针对静态shape网络，在线编译算子；针对动态shape网，优先查找系统中已编译好的算子二进制，如果查找不到对应的二进制，再编译子。
- true：在线编译算子，系统根据得到的图信息进行融合及优化，从而编译出运性能更优的算子。
- false：优先查找系统中的已编译好的算子二进制文件，如果能查找到，则不编译算子，编译性能更优；如果查找不到，则再编译算子。

> [!NOTE]说明
> 该参数仅限于大型推荐类型网络使用。

配置示例：

```python
npu.global_options().jit_compile = "auto"
```

## shape_generalization_mode

当“jit_compile”参数配置为“true”（即在线编译算子的场景）时，可通过此参数配置输入shape的泛化模式。

- STRICT（默认值）：直接使用当前迭代的shape，不进行泛化。
- FULL：若两次迭代之间的shape发生变化，则将所有轴的shape泛化为-1。
- ADAPTIVE：若两次迭代之间的shape发生变化，仅将发生变化的轴的shape化为-1。新增泛化的轴会触发模型重新编译，因此该配置下模型可能需要多次译。

> [!NOTE]说明
> 当[compile_dynamic_mode](./dynamic_shape.md#compile_dynamic_mode)配置为“True”时，首次代会将所有输入shape泛化为“-1”，此时shape_generalization_mode的配置将生效。

配置示例：

```python
npu.global_options().shape_generalization_mode = "FULL"
```

## auto_multistream_parallel_mode

该参数适用于静态/动态shape图场景，开发者可通过配置此参数配置多流并行算法，以提升图执行性能。当前支持以下取值：

- **cv**，代表开启Cube算子与Vector算子的并行执行功能。
- **LoadBalance:n**，负载均衡算法，将所有算子均匀分布在n条流上执行。n为最大流数量，正整数，取值范围[1,64]。若n的取值超过了实际可用核数，性能可能会降低。
- **MainStream:n**，主流算法，串行算子分布在主流上执行，其他可并行算子分布在其他流上执行。n为最大流数量，正整数，取值范围[1,64]。若n取值超过了实际可用核数，性能可能会降低。
- 默认值为空，Cube算子与Vector算子串行执行。

> [!NOTE]说明
>
> - 该参数仅限于推荐类型网络使用。
> - 动态shape多流下使用此功能需要先通过环境变量ENABLE_DYNAMIC_SHAPE_MULTI_STREAM使能动态shape多流，再配置此参数。关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

配置示例：

```python
npu.global_options().auto_multistream_parallel_mode = "cv"
```
