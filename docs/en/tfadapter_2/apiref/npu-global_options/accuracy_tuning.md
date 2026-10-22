# Accuracy Tuning

## precision_mode_v2

A string for the operator precision mode.

- fp16: Indicates that float16 is forcibly selected if the operator precision in the original graph is float16, bfloat16, or float32.
- origin: Retains the original precision.
  - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only float32 and bfloat16, the system automatically uses high-precision float32.
  - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only bfloat16, the AI CPU operator of float16 is used. If the AI CPU operator is not supported, an error is reported.
  - If the precision of an operator in the original graph is float32, and the implementation of the operator in the AI Core does not support float32 but supports only float16, the AI CPU operator of float32 is used. If the AI CPU operator is not supported, an error is reported.
- cube_fp16in_fp32out: The system selects a processing mode based on the operator type for AI Core operators supporting both float32 and float16.
  - For cube operators, the system processes the computation based on the operator implementation.The preferred input data type is float16 and the output data type is float32.If the float16 input data and float32 output data types are not supported, set both the input and output data types to float32.If the float32 input and output data types are not supported, set both the input and output data types to float16.If the float16 input and output data types are not supported, an error is reported.
  1. The preferred input data type is float16 and the output data type is float32.
  2. If the float16 input data and float32 output data types are not supported, set both the input and output data types to float32.
  3. If the float32 input and output data types are not supported, set both the input and output data types to float16.
  4. If the float16 input and output data types are not supported, an error is reported.
- For vector compute operators, the operator precision in the original graph is float16 or bfloat16, and float32 is forcibly selected.

  This option is invalid if the original graph contains operators not supporting float32 in the AI Core, for example, an operator that supports only float16. In this case, float16 is retained. If the operator in the AI Core does not support float32 and it is configured to the blocklist of precision reduction (by setting precision_reduce to false), the counterpart AI CPU operator supporting float32 is used. If the AI CPU operator does not support float32, an error is reported.
- mixed_float16: Mixed precision of float16, bfloat16, and float32 is used for neural network processing. For float32 and bfloat16 operators in the original graph, float16 is automatically used for certain float32 and bfloat16 operators based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation.

  Use the mixed precision mode in conjunction with loss scaling to compensate for the accuracy degradation caused by precision reduction.
- mixed_bfloat16: Mixed precision of bfloat16 and float32 is used for neural network processing. In this mode, bfloat16 is automatically used for certain float32 operators in the original graph based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation. If the operators do not support bfloat16 and float32, the AI CPU operators are used for computation. If AI CPU operators also do not support float16 and float32, an error is reported during execution.

    Note: This configuration is supported only by the Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products.
- mixed_hif8: Enables automatic mixed precision, indicating that hifloat8 (for details about this data type, see [Link](https://arxiv.org/abs/2409.16626?context=cs.AR)), float16, bfloat16, and float32 are used together for neural network processing. In this mode, hifloat8 is automatically used for certain float16, bfloat16, and float32 operators in the original graph based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation. The current version does not support this argument.

     Note: This configuration is supported only by the Ascend 950PR&950DT products.

- cube_hif8: The hifloat8 data type is forcibly used if the Cube operator in the original graph supports both hifloat8 and float16, bfloat16, or float32. The current version does not support this argument.

  Note: This configuration is supported only by the Ascend 950PR&950DT products.

Default value:

- For the Ascend 950PR&950DT products, the default value is origin.
- For the Atlas A3 products, the default value is origin.
- For the Atlas A2 products, the default value is origin.
- For the Atlas training products, this parameter does not have a default value. The default value of the precision_mode parameter is used, that is, allow_fp32_to_fp16.

Example:

```python
npu.global_options().precision_mode_v2="origin"
```

> [!NOTE]NOTE
>
> - This parameter cannot be used together with precision_mode. Use precision_mode_v2 instead.
> - This parameter can be used to set the global precision mode of a network, but it may result in precision issues on particular operators. In this case, you are advised to call [npu.keep_dtype_scope](../npu-keep_dtype_scope.md) to keep the precision of some operators unchanged.
> - For details about the built-in tuning policy for operators in mixed precision mode, see the description of modify_mixlist.
> - The Atlas training products do not support the bfloat16 data type.

## precision_mode

A string for the operator precision mode.

- allow_fp32_to_fp16:
  - For matrix operators:
    - If the operator precision in the original graph is float32, the precision is preferably reduced to float16. If the operator in the AI Core does not support float16, float32 is used. If the operator in the AI Core does not support float32, the AI CPU operator is used for computation. If the AI CPU operator also does not support float32, an error is reported during execution.
    - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.
  - For vector operators, the precision of the original graph is retained preferably.
    - If the operator precision in the original graph is float32, the precision of the original graph is preferably used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.
    - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the precision is directly reduced to float16. If the operator in the AI Core does not support float16, the AI CPU operator is used for computation. If the AI CPU operator also does not support float16, an error is reported during execution.
- force_fp16:

  Forces float16 for operators supporting float16, bfloat16, and float32. This parameter applies only to online inference scenarios.
- force_fp32/cube_fp16in_fp32out:

  force_fp32 and cube_fp16in_fp32out have the same effect. This option indicates that the system selects different processing modes based on the operator type when the operator in the AI Core supports both the float32 and float16 data types. cube_fp16in_fp32out is newly added to the new version. For cube operators, this option has clearer semantics.
  - For cube operators, the system processes the computation based on the operator implementation.
    1. The preferred input data type is float16 and the output data type is float32.
    2. If the float16 input data and float32 output data types are not supported, set both the input and output data types to float32.
    3. If the float32 input and output data types are not supported, set both the input and output data types to float16.
    4. If the float16 input and output data types are not supported, an error is reported.
  - For vector compute operators, the operator precision in the original graph is float16 or bfloat16, and float32 is forcibly selected.
  
    This option is invalid if the original graph contains operators not supporting float32 in the AI Core, for example, an operator that supports only float16. In this case, float16 is retained. If the operator in the AI Core does not support float32 and it is configured to the blocklist of precision reduction (by setting precision_reduce to false), the counterpart AI CPU operator supporting float32 is used. If the AI CPU operator does not support float32, an error is reported.
- must_keep_origin_dtype:

  Retains the original precision.
  - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only float32 and bfloat16, the system automatically uses high-precision float32.
  - If the precision of an operator in the original graph is float16, and the implementation of the operator in the AI Core does not support float16 but supports only bfloat16, the AI CPU operator of float16 is used. If the AI CPU operator is not supported, an error is reported.
  - If the precision of an operator in the original graph is float32, and the implementation of the operator in the AI Core does not support float32 but supports only float16, the AI CPU operator of float32 is used. If the AI CPU operator is not supported, an error is reported.
- allow_mix_precision_fp16/allow_mix_precision:

  allow_mix_precision has the same effect as that of allow_mix_precision_fp16, indicating that mixed precision of float16, bfloat16, and float32 is used for neural network processing. allow_mix_precision_fp16 is newly added to the new version, which has clearer semantics for easy understanding.

  For float32 and bfloat16 operators in the original model, float16 is automatically used for certain float32 and bfloat16 operators based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation.
- allow_mix_precision_bf16:

  Mixed precision of bfloat16 and float32 is used for neural network processing. In this mode, bfloat16 is automatically used for certain float32 operators on the original model based on the built-in tuning policy. This will improve system performance and reduce memory usage with minimal precision degradation. If the operator in the AI Core does not support bfloat16 and float32, the AI CPU operator is used for computation. If AI CPU operator also does not support bfloat16 and float32, an error is reported during execution.

  Note: This configuration is supported only by the Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products.

- allow_fp32_to_bf16:
  - If the operator precision in the original graph is float32, the precision of the original graph is preferably used. If the operator in the AI Core does not support float32, the precision is reduced to bfloat16. If the operator in the AI Core does not support bfloat16, the AI CPU operator is used for computation. If the AI CPU operator also does not support bfloat16, an error is reported during execution.
  - If the operator precision in the original graph is bfloat16, the precision of the original graph is preferably used. If the operator in the AI Core does not support bfloat16, float32 is used. If the operator in the AI Core does not support float32, the AI CPU operator is used for computation. If the AI CPU operator also does not support float32, an error is reported during execution.

  Note: This configuration is supported by the Ascend 950PR&950DT products, Atlas A3 products, and Atlas A2 products.

Default value:
- For the Ascend 950PR&950DT products, the default value is must_keep_origin_dtype.
- For the Atlas A3 products, the default value is must_keep_origin_dtype.
- For the Atlas A2 products, the default value is must_keep_origin_dtype.
- For the Atlas training products, the default value is allow_fp32_to_fp16.

Example:

```python
npu.global_options().precision_mode="allow_mix_precision"
```

> [!NOTE]NOTE
>
> - This option cannot be used together with precision_mode_v2. precision_mode_v2 is recommended.
> - This option can be used to set the global precision mode of a network, but it may result in performance or precision problems on particular operators. In this case, you are advised to call [npu.keep_dtype_scope](../npu-keep_dtype_scope.md) to keep the precision of some operators unchanged.
> - For details about the built-in tuning policy of each operator in mixed precision mode, see the description of the modify_mixlist option.
> - The Atlas training products do not support the bfloat16 data type.

## modify_mixlist

When mixed precision is enabled, you can use this parameter to specify the path and file name of the blocklist, trustlist, and graylist, and specify the operators that allow precision reduction and those that do not allow precision reduction.

You can enable the mixed precision by configuring precision_mode_v2 (recommended) or precision_mode in the script.

The blocklist, trustlist, and graylist storage files are in JSON format. A configuration example is as follows:

```python
npu.global_options().modify_mixlist="/home/test/ops_info.json"
```

Specify the operator type (or types separated by commas) in ops_info.json as follows.

```json
{
  "black-list": { // Blocklist
    "to-remove": [ // Move an operator from the blocklist to the graylist.
    "Xlog1py"
    ],
    "to-add": [ // Move an operator from the trustlist or graylist to the blocklist.
    "MatMul",
    "Cast"
    ]
  },
  "white-list": { // Trustlist
    "to-remove": [ // Move an operator from the trustlist to the graylist.
    "Conv2D"
    ],
    "to-add": [ // Move an operator from the blocklist or graylist to the trustlist.
    "Bias"
    ]
  }
}
```

Note: The operators in the preceding example configuration file are for reference only. The configuration should be based on the actual hardware environment and the built-in tuning policies of the operators.

You can query the built-in tuning policy of an operator in the OPP installation directory/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json file. For example:

```json
"Conv2D":{
    "precision_reduce":{
        "flag":"true"
    },
    ...
}
```

- true (trustlist): The precision of operators on the trustlist can be reduced in mixed precision mode.
- false (blocklist): The precision of operators on the blocklist cannot be reduced in mixed precision mode.
- Not specified (graylist): Follows the same mixed precision processing as the upstream operator.

The reduced precision support of the current operator follows that of the preceding operator: if the preceding operator supports reduced precision, so does the current one; otherwise, the current one does not support it either.

## customize_dtypes

If precision_mode_v2 or precision_mode is used to set the global precision mode of a network, precision issues may occur on particular operators. In this case, you can use customize_dtypes to configure the precision mode of these operators, and still compile other operators using the precision mode specified by precision_mode_v2 or precision_mode. Note if precision_mode_v2 is set to origin or precision_mode is set to must_keep_origin_dtype, customize_dtypes does not take effect.

Set it to the path (including the name of the configuration file), for example, /home/test/customize_dtypes.cfg.

Example:

```python
npu.global_options().customize_dtypes = "/home/testcustomize_dtypes.cfg"
```

List the names or types of operators whose precision needs customization in the configuration file. Each operator occupies a line, and the operator type must be defined based on Ascend IR. If both operator name and type are configured for an operator, the operator name applies during building.

The structure of the configuration file is as follows:

```python
# By operator name
Opname1::InputDtype:dtype1,dtype2,...OutputDtype:dtype1,...
Opname2::InputDtype:dtype1,dtype2,...OutputDtype:dtype1,...
# By operator type
OpType::TypeName1:InputDtype:dtype1,dtype2,...OutputDtype:dtype1,...
OpType::TypeName2:InputDtype:dtype1,dtype2,...OutputDtype:dtype1,...
```

Example:

```text
# By operator name
resnet_v1_50/block1/unit_3/bottleneck_v1/Relu::InputDtype:float16,int8,OutputDtype:float16,int8
# By operator type
OpType::Relu:InputDtype:float16,int8,OutputDtype:float16,int8
```

> [!NOTE]NOTE
>
> - You can view the supported precision modes for operators in the operator information library, which is saved at opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json under the CANN installation directory by default.
> - The data type specified by this option takes high priority, which may invite accuracy or performance degradation. If the specified data type is not supported, the building will fail.
> - If the configuration is performed based on the operator name, the operator name may change due to operations such as fusion and splitting during model compilation. As a result, the configuration does not take effect and the precision is not improved. In this case, you need to obtain logs to locate the fault. For details about the logs, see [Log Reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/maintenref/logreference/logreference_0001.html).
