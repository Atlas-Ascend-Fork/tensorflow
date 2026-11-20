# ENABLE_HF32_EXECUTION

## Description

Enables or disables the function of automatically replacing the FP32 data type with the HF32 data type for the TensorFlow 1.15 network. In the current version, this environment variable takes effect only for Conv and Matmul operators.

- **1**: enabled.
- **0**: disabled.
- If this environment variable is not specified, FP32-to-HF32 conversion is enabled for Conv operators by default, and FP32-to-HF32 conversion is disabled for Matmul operators by default.

HF32 is a single-precision floating-point type developed by Ascend for internal computation of operators. The following figure shows the comparison of HF32 with other common data types. HF32 shares the same value range with FP32, but its mantissa precision \(11 bits\) is close to FP16 \(10 bits\). Replacing the original FP32 single-precision data type with the HF32 single-precision data type by precision reduction can greatly reduce the space occupied by data and achieve performance improvement.

**Figure  1**  Comparison of HF32 with other data types
![](figures/hf32_vs_others_datatype.png)

## Example

```bash
export ENABLE_HF32_EXECUTION=1
```

## Constraints

- This environment variable applies only to the scenario where the  TensorFlow  1.15 network training or online inference is performed on the Ascend platform.
- For the same operator, if  **enable_float_32_execution**  or  **enable_hi_float_32_execution**  \(not supported in the current version\) is configured using  **op_precision_mode**, this environment variable cannot be used. If they are used together, the priority is as follows:

    **op_precision_mode\(ByNodeName\)**  \>  **ENABLE_HF32_EXECUTION**  \>  **op_precision_mode\(ByOpType\)**

- Before enabling the function of automatically converting the FP32 data type to the HF32 data type, ensure that the input or output type of the operator is float32. In the online inference scenario, the default values of  **precision_mode**  and  **precision_mode_v2**  are  **force_fp16**  and  **fp16**, respectively. If operators in the network model support both float16 and float32 data types, float16 is used. In this case, the environment variable  **ENABLE_HF32_EXECUTION**  does not take effect. Therefore, you are advised to change  **precision_mode**  and  **precision_mode_v2**  to  **must_keep_origin_dtype**  and  **origin**, respectively.

## Applicability

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products
