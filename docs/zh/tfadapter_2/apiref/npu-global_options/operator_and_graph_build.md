# 算子编译与图编译

## op_compiler_cache_mode

用于配置算子编译磁盘缓存模式。

- enable（默认值）：启用算子编译缓存功能。启用后，算子编译信息缓存至磁，相同编译参数的算子无需重复编译，直接使用缓存内容，从而提升编译速度。
- force：启用算子编译缓存功能，区别于enable模式，force模式下会先删除有缓存，再重新编译并加入缓存。例如当用户的Python变更、依赖库变更、算子优后知识库变更时，需要先指定为force用于先清理已有的缓存，后续再修改为nable模式，避免每次编译时都强制刷新缓存。**需要注意**，force选项不建议程序并行编译时设置，否则可能会导致其他模型因使用的缓存内容被清除而编译败。
- disable：禁用算子编译缓存功能。

使用说明：

- 启动算子编译缓存功能时，可通过op_compiler_cache_dir配置算子编译缓存件存储路径。
- 建议模型最终发布时设置编译缓存选项为disable或者force。
- 若[op_debug_level](./deprecated_options.md#op_debug_level)配置非0值，会忽略p_compiler_cache_mode的配置，不启用算子编译缓存功能，算子全部重新编。
- 若[op_debug_config](./debugging.md#op_debug_config)配置非空，且配置文件中未配置p_debug_list字段，会忽略op_compiler_cache_mode的配置，不启用算子编译存功能，算子全部重新编译。
- 若[op_debug_config](./debugging.md#op_debug_config)配置非空，且配置文件中配置了p_debug_list字段，当op_compiler_cache_mode配置为enable或force时，列中的算子会重新编译，列表外的算子会启用算子编译缓存，不再重新编译。
- 启用算子编译缓存功能时，默认使用的缓存文件磁盘空间大小是500MB，磁盘间不足时，会删除缓存文件，默认保留50%的缓存空间。开发者也可以通过以下方自定义存储缓存文件的磁盘空间大小与保留缓存空间比例：
    1. 通过配置文件op_cache.ini设置。
        
        算子编译完成后，会在op_compiler_cache_dir指定的目录下自动生成op_cache.ini文件，开发者可通过该文件设置缓存磁盘空间大小与保留缓存空间比例。若op_cache.ini文件不存在，可手动创建。

        在“op_cache.ini”文件中，增加如下信息：

        ```text
        #配置文件格式，必须包含，自动生成的文件中默认包括如下信息，手动创建时，需要填写
        [op_compiler_cache]
        #限制某个AI处理器下缓存文件的磁盘空间的大小，整数，单位为MB
        max_op_cache_size=500
        #当磁盘空间不足时，设置需要保留的缓存文件比例，取值范围：[1,100]，单位为百分比；例如80表示磁盘空间不足时，会保留80%的缓存空间中的文件，其余删除
        remain_cache_size_ratio=80
        ```

        - 上述文件中的max_op_cache_size和remain_cache_size_ratio参数取值都有效时，op_cache.ini文件才会生效。
        - 当编译缓存文件大小超过“max_op_cache_size”的设置值，且超过半小时缓存文件未被访问时，缓存文件就会老化（算子编译时，不会因为编译缓存文件大小超过设置值而中断，所以当“max_op_cache_size”设置过小时，会出现实际编译缓存文件大小超过此设置值的情况）。
        - 若需要关闭编译缓存老化功能，可将“max_op_cache_size”设置为“-1”，此时访问算子缓存时不会更新访问时间，算子编译缓存不会老化，磁盘空间使用默认大小500MB。
        - 若多个使用者使用相同的缓存路径，该配置文件会影响所有使用者。

    2. 通过环境变量ASCEND_MAX_OP_CACHE_SIZE设置。
        
        开发者可以通过环境变量ASCEND_MAX_OP_CACHE_SIZE来限制某个AI处理器下缓存文件的磁盘空间的大小，当编译缓存空间大小达到ASCEND_MAX_OP_CACHE_SIZE设置的取值，且超过半个小时缓存文件未被访问时，缓存文件就会老化。可通过环境变量ASCEND_REMAIN_CACHE_SIZE_RATIO设置需要保留缓存的空间大小比例。关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“算子编译”章节。

        若需要关闭编译缓存老化功能，可将环境变量“ASCEND_MAX_OP_CACHE_SIZE”设置为-1。

    若同时配置了op_cache.ini文件和环境变量，则优先读取op_cache.ini文件中的配置项，若op_cache.ini文件和环境变量都未设置，则读取系统默认值：默认磁盘空间大小500MB，默认保留缓存的空间50%。

配置示例：

```python
npu.global_options().op_compiler_cache_mode="enable"
```

## op_compiler_cache_dir

用于配置算子编译磁盘缓存的目录。

路径支持大小写字母（a-z，A-Z）、数字（0-9）、下划线（_）、中划线（-）、点（.）、中文字符。

如果参数指定的路径存在且有效，则在指定的路径下自动创建子目录kernel_cache；如果指定的路径不存在但路径有效，则先自动创建目录，然后在该径下自动创建子目录kernel_cache。

算子编译缓存文件存储优先级为：

配置参数“op_compiler_cache_dir” \> $\{ASCEND_CACHE_PATH\}/kernel_cache \> 默认路径（$HOME/atc_data）。

关于环境变量ASCEND_CACHE_PATH的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

配置示例：

```python
npu.global_options().op_compiler_cache_dir="/home/test/kernel_cache"
```

## aicore_num

用于配置算子编译时使用的最大Cube Core数量和Vector Core数量。

配置格式：“整数1|整数2”，中间使用“|”分割，整数1表示算子编译时使用的最大ube Core数量，整数2表示算子编译时使用的最大Vector Core数量，整数1与整2都需要大于0，小于等于AI处理器包含的Cube Core数量和Vector Core数量。

- 该参数支持如下产品：
  - Atlas A3系列产品
  - Atlas A2系列产品
- 不同型号的AI处理器包含的最大Cube Core与Vector Core的数量可通过CANN软件安装目录/<arch\>-linux/data/platform_config/<soc_version\>_.ini”文件查看，如下所示，说明AI处理器上存在24个Cube ore，48个Vector Core。
  
  ```text
  [SoCInfo]
  ai_core_cnt=24
  cube_core_cnt=24
  vector_core_cnt=48
  ```

- 静态shape场景下，如果模型编译时复用了已有算子二进制文件（即jit_compile”参数设为“false”），“aicore_num”参数不生效。

配置示例：

```python
npu.global_options().aicore_num="2|4"
```

## oo_constant_folding

常量折叠功能开启与关闭开关。

常量折叠，是指在图编译阶段直接计算并替换常量表达式的值，从而降低内存用。一般情况下，建议保持默认值启用常量折叠功能。但有些网络编译运行程中需要较多的内存，而常量内存在图的整个生命周期会一直占用，若存在常量叠后会增加总内存的场景，可考虑通过此参数关闭常量折叠功能。

- True（默认值）：启用常量折叠功能。
- False：关闭常量折叠功能。

```python
npu.global_options().oo_constant_folding=True
```

关闭常量折叠功能后，若网络编译运行出错，如下所示：

- 示例1：debug日志中报错信息如下所示：

    ```text
    [ERROR] GE(3469659,python3.7):2025-02-25-05:** [ge_deleted_opcc:21]3470503 Run: ErrorNo: 4294967295(failed) [Delete][Node]Node:HcomAllReduce/input type is ExpandDims, should be deleted byge.
    ```

    如上错误信息说明网络中存在图编译时需要被常量折叠的算“ExpandDims”，所以不支持关闭常量折叠功能。

- 示例2：返回错误码“EZ3003”，打屏信息如下所示：

    ```text
    Error Message is :
    EZ3003: [PID: 3482331] 2025-02-25-14:07:19.774.362 Nosupported Ops kernel and engine are found for [import/conv2d_1convolutionimport/batch_normalization_1FusedBatchNorm_1_filter_host], optype [ConvBnFilterHost].
    Possible Cause: The operator is not supported by the system.Therefore, no hit is found in any operator information library.
    ```

    如上错误信息说明网络中存在图编译时需要被常量折叠的算“ConvBnFilterHost”，所以不支持关闭常量折叠功能。
