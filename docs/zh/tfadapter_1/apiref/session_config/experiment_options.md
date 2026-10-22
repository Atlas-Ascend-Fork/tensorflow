# 试验参数

试验参数为调试功能扩展参数，后续版本可能会存在变更，不支持应用于生产环境。

## graph_compiler_cache_dir

该参数用于配置图编译磁盘缓存目录，当该参数配置为非空时，图编译磁盘缓存功能生效。

图编译缓存功能支持将图编译结果进行磁盘持久化，当再次执行图编译运行时，可直接加载磁盘上缓存的编译结果，从而减少图编译时长。

需要注意：

- 配置的缓存目录必须存在，否则会导致编译失败。
- 图编译时，会根据此参数的值确定缓存文件，若缓存文件不存在则保存缓存，若缓存文件存在则直接加载缓存。
- 图发生变化后，原来的缓存文件不可用，用户需要手动删除缓存目录中的缓存文件，然后重新编译生成缓存文件。
- 缓存不保证跨版本的兼容性，如果版本升级，需要清理缓存目录重新编译生成缓存。
- 该功能当前不支持带资源类算子的模型。

配置示例：

```python
custom_op.parameter_map["graph_compiler_cache_dir"].s = tf.compat.as_bytes("/root/build_cache_dir")
```

## jit_compile

模型编译时，选择是优先在线编译算子，还是优先使用已编译好的算子二进制文件。

- auto（默认值）：针对静态shape网络，在线编译算子；针对动态shape网络，优先查找系统中已编译好的算子二进制，如果查找不到对应的二进制，再编译算子。
- true：在线编译算子，系统根据得到的图信息进行融合及优化，从而编译出运行性能更优的算子。
- false：优先查找系统中已编译好的算子二进制文件，如果能查找到，则不再编译算子，编译性能更优；如果查找不到，则再编译算子。

> [!NOTE]说明
> 该参数仅限于大型推荐类型网络使用。

配置示例：

```python
custom_op.parameter_map["jit_compile"].s = tf.compat.as_bytes( "auto")
```

## shape_generalization_mode

当“jit_compile”参数配置为“true”（即在线编译算子的场景）时，可通过此参数配置输入shape的泛化模式。

- STRICT（默认值）：直接使用当前迭代的shape，不进行泛化。
- FULL：若两次迭代之间的shape发生变化，则将所有轴的shape泛化为-1。
- ADAPTIVE：若两次迭代之间的shape发生变化，仅将发生变化的轴的shape泛化为-1。新增泛化的轴会触发模型重新编译，因此该配置下模型可能需要多次编译。

> [!NOTE]说明
> 当[compile_dynamic_mode](../../apiref/session_config/dynamic_shape.md#compile_dynamic_mode)配置为“True”时，首次迭代会将所有输入shape泛化为“-1”，此时shape_generalization_mode的配置将不生效。

配置示例：

```python
custom_op.parameter_map["shape_generalization_mode"].s = tf.compat.as_bytes( "FULL")
```

## experimental_accelerate_train_mode

针对超过1小时以上的训练场景，开发者可以通过此配置触发训练加速，提升训练性能。

软件内部会根据开发者配置的加速类型、加速触发模式以及低精度训练流程占比，对相应比例的训练流程降精度编译运行，剩余的训练流程仍按照原始精度编译运行。

该配置项取值类型为字符串，由“|”符号分割为三个字段，例如：fast|step|0.9。

- 第一个字段代表加速类型，支持“fast”与“fast1”两种取值。
  - “fast”表示降精度时，按照float16的数据类型编译执行。
  - “fast1”表示降精度时，按照bf16的数据类型编译执行。

- 第二个字段支持“step”和“loss”两种取值，分别表示根据step值或者loss值来分割整个训练流程为低精度训练和高精度训练。
- 第三个字段代表低精度训练流程在总step或总loss中的占比。
  - 当第二个字段取值为“step”时，此字段的合法取值范围为“0.2 \~ 0.9”，默认值为“0.9”。
  - 当第二个字段的取值为“loss”时，此字段的合法取值范围为“1.01 \~ 1.5”，默认值为“1.05”。

配置示例：

- 按照step触发加速

  ```python
  custom_op.parameter_map["experimental_accelerate_train_mode"].s =
  tf.compat.as_bytes("fast|step|0.9")
  ```

- 按照loss触发加速

  ```python
  custom_op.parameter_map["experimental_accelerate_train_mode"].s =
  tf.compat.as_bytes("fast|loss|1.05")
  ```

**需要注意：**

1. 若需要通过此配置项触发训练加速，需要确保网络脚本能够正常收敛。
2. 针对网络脚本训练耗时较短的场景，若开启此配置项，端到端性能耗时不一定能够产生正向收益。
3. 此配置项的功能与网络脚本中配置的精度模式有关：
   - 当通过“precision_mode”参数配置精度模式时，仅支持取值为“allow_fp32_to_fp16”， “must_keep_origin_dtype”或者“空”的场景下开启此配置。
   - 当通过“precision_mode_v2”参数配置精度模式时，仅支持取值为“origin”或者“空”的场景下开启此配置。

4. 此配置项功能与小循环次数有关，开启小循环的场景下可能不能准确按照指定的步数或者loss值分割整个训练流程，最终可能对loss和精度产生影响。
5. 开启此配置项时，开发者需要按照如下规则适配修改网络脚本。

   - 如果是step分割，需要设置"STEP_NOW"和"TOTAL_STEP"的环境变量，告知底层每次run的step值和总共的step数。
   - 如果是loss分割，则需要设置"LOSS_NOW"和"TARGET_LOSS"环境变量，告知底层每次run的loss值和目标loss值。

   step分割方式的网络脚本修改示例如下：

   ```python
   # 设置环境变量STEP_NOW的初始值为“0”
   os.environ['STEP_NOW'] =  "0"
   # 设置环境变量TOTAL_STEP为总step数
   os.environ['TOTAL_STEP'] =  str(epoch)
   for i in range(epoch):
       # 执行训练操作
       _, step = sess.run([train_op, global_step])
       # 更新环境变量STEP_NOW的值为当前执行的step
       os.environ['STEP_NOW'] =  str(step)
   ```

   loss分割方式的网络脚本修改示例如下：

   ```python
   # 设置环境变量LOSS_NOW的初始值为网络初始loss值，以下配置值仅为示例
   os.environ['LOSS_NOW'] =  "7.0"
   # 设置环境变量TARGET_LOSS的值为目标loss值，以下配置值仅为示例
   os.environ['TARGET_LOSS'] =  "3.0"
   for i in range(epoch):
       # 执行训练操作
       _, step = sess.run([train_op, global_step])
       # 更新环境变量LOSS_NOW的值为当前执行的loss值
       os.environ['LOSS_NOW'] =  str(loss)
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
> - 动态shape多流下使用此功能需要先通过环境变量ENABLE_DYNAMIC_SHAPE_MULTI_STREAM使能动态shape多流，再配置此参数。
> 关于环境变量的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

配置示例：

```python
custom_op.parameter_map["auto_multistream_parallel_mode"].s =
tf.compat.as_bytes("cv")
```
