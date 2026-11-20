# ENABLE_HF32_EXECUTION

## 功能描述

针对TensorFlow 1.15网络，是否启用HF32自动代替FP32数据类型的功能，当前版本此环境变量仅针对Conv类算子与Matmul类算子生效。

- "1"：启用FP32数据类型自动转换为HF32数据类型的功能。
- "0"：关闭FP32数据类型自动转换为HF32数据类型的功能。
- 若不配置此环境变量，针对Conv类算子，默认使能FP32转换为HF32，针对Matmul类算子，默认不使能FP32转换为HF32。

HF32是昇腾推出的专门用于算子内部计算的单精度浮点类型，与其他常用数据类型的比较如下图所示。可见，HF32与FP32支持相同的数值范围，但尾数位精度（11位）却接近FP16（10位）。通过降低精度让HF32单精度数据类型代替原有的FP32单精度数据类型，可大大降低数据所占空间大小，实现性能的提升。

**图 1**  HF32与其他数据类型比较
![HF32与其他数据类型比较](figures/hf32_vs_others_datatype.png)

## 配置示例

```bash
export ENABLE_HF32_EXECUTION=1
```

## 使用约束

- 该环境变量仅适用于TensorFlow  1.15网络在昇腾平台执行训练或在线推理的场景。
- 针对同一个算子，如果通过参数op_precision_mode配置了enable_float_32_execution或enable_hi_float_32_execution（当前版本暂不支持此配置），则不能再与此环境变量同时使用，若同时使用，优先级如下：

    op_precision_mode（ByNodeName，按节点名称设置精度模式） \> ENABLE_HF32_EXECUTION  \> op_precision_mode（ByOpType，按算子类型设置精度模式）

- 启用FP32数据类型自动转换为HF32数据类型的功能时，需要确保算子输入或者输出类型为float32。在线推理场景下，由于TF Adapter精度模式配置参数“precision_mode”与“precision_mode_v2”的默认值分别为“force_fp16”与“fp16”，即网络模型中的算子如果既支持float16又支持float32数据类型，会强制使用float16，这种场景下环境变量“ENABLE_HF32_EXECUTION”无法生效，所以建议修改精度配置参数“precision_mode”与“precision_mode_v2”的值分别为“must_keep_origin_dtype”与“origin”。

## 支持的型号

Ascend 950PR&950DT系列产品

Atlas A3系列产品

Atlas A2系列产品
