# 精度调优

## precision_mode_v2

算子精度模式，配置要求为string类型。

- fp16：表示原图中算子精度为float16、bfloat16或float32时，强制选择float16。

- origin：保持原图精度。

  - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持float32和bfloat16，则系统内部会自动采用高精度float32。
  - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持bfloat16，则会使用float16的AI CPU算子；如果AI CPU算子也不支持，则执行报错。
  - 如果原图中某算子精度为float32，AI Core中该算子的实现不支持float32类型、仅支持float16类型，则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

- cube_fp16in_fp32out：

  AI Core中该算子既支持float32又支持float16数据类型时，系统内部根据算子类型不同，选择不同的处理方式。

  - 对于矩阵计算类算子，系统内部会按算子实现的支持情况处理：
    1. 优先选择输入数据类型为float16且输出数据类型为float32；
    2. 如果1中的场景不支持，则选择输入数据类型为float32且输出数据类型为float32；
    3. 如果2中的场景不支持，则选择输入数据类型为float16且输出数据类型为float16；
    4. 如果3中的场景不支持，则报错。

  - 对于矢量计算类算子，表示原图中算子精度为float16或bfloat16，强制选择float32。

    如果原图中存在部分算子，在AI Core中该算子的实现不支持float32，比如某算子仅支持float16类型，则该参数不生效，仍然使用支持的float16；如果在AI Core中该算子的实现不支持float32，且又配置了黑名单（precision_reduce = false），则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

- mixed_float16：

  表示使用混合精度float16、bfloat16和float32数据类型来处理神经网络。针对原图中float32和bfloat16数据类型的算子，按照内置的优化策略，自动将部分float32和bfloat16的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

  开启该功能开关后，用户可以同时使能Loss Scaling，从而补偿降低精度带来的精度损失。

- mixed_bfloat16：

  表示使用混合精度bfloat16和float32数据类型来处理神经网络。针对原图中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到bfloat16，从而在精度损失很小的情况下提升系统性能并减少内存使用；如果算子不支持bfloat16和float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  说明：仅Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置。

- mixed_hif8：

  开启自动混合精度功能，表示混合使用hifloat8（此数据类型介绍可参见[HiFloat8](https://arxiv.org/abs/2409.16626?context=cs.AR)）、float16、bfloat16和float32数据类型来处理神经网络。针对原图中float16、bfloat16和float32数据类型的算子，按照内置的优化策略，自动将部分float16、bfloat16和float32的算子降低精度到hifloat8，从而在精度损失很小的情况下提升系统性能并减少内存使用。**当前版本不支持该选项。**

  说明：仅Ascend 950PR&950DT系列产品支持此配置。

- cube_hif8：

  表示若原图中的cube算子既支持hifloat8，又支持float16、bfloat16或float32数据类型时，强制选择hifloat8数据类型。****当前版本不支持该选项。****

  说明：仅Ascend 950PR&950DT系列产品支持此配置。

训练场景下：

- 针对Ascend 950PR&950DT系列产品，该配置项默认值为“origin”。
- 针对Atlas A3系列产品，该配置项默认值为“origin”。
- 针对Atlas A2系列产品，该配置项默认值为“origin”。
- 针对Atlas训练系列产品，该配置项无默认取值，以“precision_mode”参数的默认值为准，即“allow_fp32_to_fp16”。

在线推理场景下：该配置项默认值为“fp16”。

配置示例：

```python
custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("origin")
```

说明：

- 该参数不能与“precision_mode”参数同时使用，建议使用“precision_mode_v2”参数。
- 在使用此参数设置整个网络的精度模式时，可能会存在个别算子存在精度问题，此种场景下，建议通过[keep_dtype_scope](../npu_scope/keep_dtype_scope.md)接口设置某些算子保持原图精度。
- 混合精度场景下算子的内置优化策略可参见“modify_mixlist”参数的详细说明。
- bfloat16数据类型不支持以下产品：
  - Atlas训练系列产品
  - Atlas推理系列产品

## precision_mode

算子精度模式，配置要求为string类型。

- allow_fp32_to_fp16：
  - 对于矩阵类算子：
    - 如果原图中算子精度为float32，优先降低精度到float16，如果AI Core中算子不支持float16，则继续选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  - 对于矢量类算子，优先保持原图精度：
    - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
    - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则直接降低精度到float16；如果AI Core中算子不支持float16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

- force_fp16：

  算子同时支持float16、bfloat16和float32数据类型时，强制选择float16数据类型。**此参数仅适用于在线推理场景。**

- force_fp32/cube_fp16in_fp32out：

  配置为force_fp32或cube_fp16in_fp32out，效果等同，该选项用来表示AI Core中该算子既支持float32又支持float16数据类型时，系统内部都会根据算子类型不同，选择不同的处理方式。cube_fp16in_fp32out为新版本中新增的，对于矩阵计算类算子，该选项语义更清晰。

  - 对于矩阵计算类算子，系统内部会按算子实现的支持情况处理：
    1. 优先选择输入数据类型为float16且输出数据类型为float32；
    2. 如果1中的场景不支持，则选择输入数据类型为float32且输出数据类型为float32；
    3. 如果2中的场景不支持，则选择输入数据类型为float16且输出数据类型为float16；
    4. 如果3中的场景不支持，则报错。

  - 对于矢量计算类算子，表示原图中算子精度为float16或bfloat16，强制选择float32。

    如果原图中存在部分算子，在AI Core中该算子的实现不支持float32，比如某算子仅支持float16类型，则该参数不生效，仍然使用支持的float16；如果在AI Core中该算子的实现不支持float32，且又配置了黑名单（precision_reduce = false），则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

- must_keep_origin_dtype：

  保持原图精度。

  - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持float32和bfloat16，则系统内部会自动采用高精度float32。
  - 如果原图中某算子精度为float16，AI Core中该算子的实现不支持float16、仅支持bfloat16，则会使用float16的AI CPU算子；如果AI CPU算子也不支持，则执行报错。
  - 如果原图中某算子精度为float32，AI Core中该算子的实现不支持float32类型、仅支持float16类型，则会使用float32的AI CPU算子；如果AI CPU算子也不支持，则执行报错。

- allow_mix_precision_fp16/allow_mix_precision：

  配置为allow_mix_precision或allow_mix_precision_fp16，效果等同，均表示使用混合精度float16、bfloat16和float32数据类型来处理神经网络的过程。allow_mix_precision_fp16为新版本中新增的，语义更清晰，便于理解。

  针对原始模型中float32和bfloat16数据类型的算子，按照内置的优化策略，自动将部分float32和bfloat16的算子降低精度到float16，从而在精度损失很小的情况下提升系统性能并减少内存使用。

- allow_mix_precision_bf16：

  表示使用混合精度bfloat16和float32数据类型来处理神经网络的过程。针对原始模型中float32数据类型的算子，按照内置的优化策略，自动将部分float32的算子降低精度到bfloat16，从而在精度损失很小的情况下提升系统性能并减少内存使用；如果AI Core中算子不支持bfloat16和float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  说明：仅Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置。

- allow_fp32_to_bf16：

  - 如果原图中算子精度为float32，则优先使用原图精度float32，如果AI Core中算子不支持float32，则降低精度到bfloat16；如果AI Core中算子不支持bfloat16，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。
  - 如果原图中算子精度为bfloat16，则优先使用原图精度bfloat16，如果AI Core中算子不支持bfloat16，则选择float32，如果AI Core中算子不支持float32，则使用AI CPU算子进行计算；如果AI CPU算子也不支持，则执行报错。

  说明：Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置。

**训练场景下**：

- 针对Ascend 950PR&950DT系列产品，默认配置项为“must_keep_origin_dtype”。
- 针对Atlas A3系列产品，默认配置项为“must_keep_origin_dtype”。
- 针对Atlas A2系列产品，默认值为“must_keep_origin_dtype”。
- 针对Atlas训练系列产品，默认值为“allow_fp32_to_fp16”。

**在线推理场景下**，默认值为“force_fp16”。

配置示例：

```python
custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
```

说明：

- 该参数不能与“precision_mode_v2”参数同时使用，建议使用“precision_mode_v2”参数。
- 在使用此参数设置整个网络的精度模式时，可能会存在个别算子存在精度问题。
  
  训练场景下，建议通过[keep_dtype_scope](../npu_scope/keep_dtype_scope.md)接口设置某些算子保持原图精度。
  
  推理场景下，建议通过[keep_tensors_dtypes](../npu_util/keep_tensors_dtypes.md)接口设置某些算子保持原图精度。
- 混合精度场景下算子的内置优化策略可参见“modify_mixlist”参数的详细说明。
- bfloat16数据类型不支持以下产品：
  - Atlas训练系列产品
  - Atlas推理系列产品

## modify_mixlist

开启混合精度的场景下，开发者可通过此参数指定混合精度黑白灰名单的路径以及文件名，自行指定哪些算子允许降精度，哪些算子不允许降精度。

用户可以在脚本中通过配置“precision_mode_v2”（推荐）参数或者“precision_mode”参数开启混合精度。

黑白灰名单存储文件为JSON格式，配置示例如下：

```python
custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
```

ops_info.json中可以指定算子类型，多个算子使用英文逗号分隔，样例如下：

```json
{
  "black-list": {                  // 黑名单
     "to-remove": [                // 黑名单算子转换为灰名单算子
     "Xlog1py"
     ],
     "to-add": [                   // 白名单或灰名单算子转换为黑名单算子
     "MatMul",
     "Cast"
     ]
  },
  "white-list": {                  // 白名单
     "to-remove": [                // 白名单算子转换为灰名单算子 
     "Conv2D"
     ],
     "to-add": [                   // 黑名单或灰名单算子转换为白名单算子
     "Bias"
     ]
  }
}
```

**说明：上述配置文件样例中展示的算子仅作为参考，请基于实际硬件环境和具体的算子内置优化策略进行配置。**

混合精度场景下算子的**内置优化策略**可在“CANN软件安装目录/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json”文件中查询，例如：

```json
"Conv2D":{
    "precision_reduce":{
        "flag":"true"
},
...
}
```

- true（白名单）：表示混合精度模式下，**允许**当前算子降低精度。
- false（黑名单）：表示混合精度模式下，**不允许**当前算子降低精度。
- 不配置（灰名单）：表示当前算子的混合精度处理机制和前一个算子保持一致，即如果前一个算子支持降精度处理，当前算子也支持降精度；如果前一个算子不允许降精度，当前算子也不支持降精度。

## customize_dtypes

使用precision_mode_v2或precision_mode参数设置整个网络的精度模式时，可能会存在个别算子存在精度问题，此种场景下，可以使用customize_dtypes参数配置个别算子的精度模式，而模型中的其他算子仍以precision_mode_v2或precision_mode指定的精度模式进行编译。需要注意，当precision_mode_v2取值为“origin”或precision_mode取值为“must_keep_origin_dtype”时，customize_dtypes参数不生效。

该参数需要配置为配置文件路径及文件名，例如：/home/test/customize_dtypes.cfg。

配置示例：

```python
custom_op.parameter_map["customize_dtypes"].s = tf.compat.as_bytes("/home/test/customize_dtypes.cfg")
```

配置文件中列举需要自定义计算精度的算子名称或算子类型，每个算子单独一行，且算子类型必须为基于Ascend IR定义的算子的类型。对于同一个算子，如果同时配置了算子名称和算子类型，编译时以算子名称为准。

配置文件格式要求：

```text
# 按照算子名称配置
Opname1::InputDtype:dtype1,dtype2,…OutputDtype:dtype1,…
Opname2::InputDtype:dtype1,dtype2,…OutputDtype:dtype1,…
# 按照算子类型配置
OpType::TypeName1:InputDtype:dtype1,dtype2,…OutputDtype:dtype1,…
OpType::TypeName2:InputDtype:dtype1,dtype2,…OutputDtype:dtype1,…
```

配置文件配置示例：

```text
# 按照算子名称配置
resnet_v1_50/block1/unit_3/bottleneck_v1/Relu::InputDtype:float16,int8,OutputDtype:float16,int8
# 按照算子类型配置
OpType::Relu:InputDtype:float16,int8,OutputDtype:float16,int8
```

**说明**：

- 算子具体支持的计算精度可以从算子信息库中查看，默认存储路径为CANN软件安装后文件存储路径的：opp/built-in/op_impl/ai_core/tbe/config/_<soc_version\>_/aic-_<soc_version\>_-ops-info-<opType\>.json。
- 通过该参数指定的优先级高，因此可能会导致精度/性能的下降；如果指定的dtype不支持，会导致编译失败。
- 若通过算子名称进行配置，由于模型编译过程中会进行融合、拆分等优化操作，可能会导致算子名称发生变化，进而导致配置不生效，未达到精度提升的目的。此种场景下，可进一步通过获取日志进行问题定位，关于日志的详细说明请参见《[日志参考](https://hiascend.com/document/redirect/CannCommunitylogref)》。
