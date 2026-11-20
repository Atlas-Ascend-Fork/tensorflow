# Training with Mixed Precision

## Introduction to Mixed Precision

Mixed precision is a common way to improve performance in the industry. It increases the data computing parallelism by reducing some computing precisions. Mixed precision is the combined use of the float16 and float32 data types in training deep neural networks, which reduces memory usages and accesses. Training with mixed precision presents itself as a better choice for training large networks without compromising the network accuracy produced by float32.

You can enable the mixed precision by configuring  **precision_mode_v2**  \(recommended\) or  **precision_mode**  in the script.

For details about  **precision_mode_v2**  and  **precision_mode**, see  [Accuracy Tuning](../accuracy_debugging/accuracy_debugging.md).

If automatic mixed precision is enabled, you are advised to enable the LossScaleOptimizer to compensate for the accuracy loss caused by precision reduction. For details about how to port the LossScaleOptimizer, see  [Replacing LossScaleOptimizer](../script_migration/manual_porting.md#replacing-lossscaleoptimizer). To analyze profile data, manually modify the operator precision mode. You can refer to  [Modifying the Blocklist and Trustlist for Mixed Precision](#modifying-the-blocklist-and-trustlist-for-mixed-precision)  to specify operators to reduce or preserve the precision.

## Setting the Precision Mode

This section uses setting  **precision_mode_v2**  to  **mixed_float16**  as an example to describe how to set the mixed precision mode.

Before initializing the NPU, set  [precision_mode_v2](../../apiref/npu-global_options/accuracy_tuning.md#precision_mode_v2)  in your training script.

```python
import npu_device as npu
npu.global_options().precision_mode_v2 = 'mixed_float16'  # Enables automatic mixed precision, indicating that both float16 and float32 are used for neural network processing.
npu.open().as_default()
```

## Modifying the Blocklist and Trustlist for Mixed Precision

When automatic mixed precision is enabled, the system automatically reduces the precisions of some data types on a network based on the built-in tiling policy. This improves the system performance while reducing the memory usage at low accuracy loss.

Find the built-in tiling policy in  **/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json**  under the CANN installation directory.

```json
{
    "Conv2D":{
        "precision_reduce":{
            "flag":"true"
        }
    },
    ...
}
```

- Scenarios where  **precision_mode_v2**  is set to  **mixed_float16**  and  **precision_mode**  is set to  **allow_mix_precision_fp16/allow_mix_precision**:
  - If the field value is  **true**, the operator is on the mixed precision trustlist and its precision will be reduced from float32 to float16.
  - If the field value is  **false**, the operator is on the mixed precision blocklist and its precision will not be reduced from float32 to float16.
  - If an operator does not have the  **precision_reduce**  option configured, the operator is on the graylist and will follow the same precision processing as the upstream operator.

- Scenarios where  **precision_mode**  is set to  **allow_mix_precision_bf16**  (only on  Atlas A3 products and Atlas A2 products):
  - If the field value is  **true**, the operator is on the mixed precision trustlist and its precision will be reduced from float32 to bfloat16.
  - If the field value is  **false**, the operator is on the mixed precision blocklist and its precision will not be reduced from float32 to bfloat16.
  - If an operator does not have the  **precision_reduce**  option configured, the operator is on the graylist and will follow the same precision processing as the upstream operator.

You can specify operators to reduce or preserve the precision based on the built-in tuning policy.

- \(Recommended\) Use  **modify_mixlist**  to modify the blocklist, trustlist, and graylist of mixed precision.

    Before initializing the NPU, set [modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist) in your training script to modify the blocklist, trustlist, and graylist of mixed precision. The following is an example:

    ```python
    import npu_device as npu
    npu.global_options().modify_mixlist = "/home/test/ops_info.json"
    npu.open().as_default()
    ```

    **ops_info.json**  is the configuration file of the blocklist, trustlist, and graylist for mixed precision. Multiple operators are separated by commas \(,\). An example is as follows:

    ```json
    {
      "black-list": {                  // Blocklist
         "to-remove": [                // Move an operator from the blocklist to the graylist.
         "Xlog1py"
         ],
         "to-add": [                   // Move an operator from the trustlist or graylist to the blocklist.
         "MatMul",
         "Cast"
         ]
      },
      "white-list": {                  // Trustlist
         "to-remove": [                // Move an operator from the trustlist to the graylist.
         "Conv2D"
         ],
         "to-add": [                   // Move an operator from the blocklist or graylist to the trustlist.
         "Bias"
         ]
      }
    }
    ```

    Assume that operator A is in the trustlist by default. If you want to move it to the blocklist, follow any of the positive examples below:

    1. \(Positive example\) Directly add the operator to the blocklist.

        ```json
        {
          "black-list": {
             "to-add": ["A"]
          }
        }
        ```

        The operator will be deleted from the trustlist and added to the blocklist.

    2. \(Positive example\) Delete the operator from the trustlist and add it to the blocklist.

        ```json
        {
          "black-list": {
             "to-add": ["A"]
          },
          "white-list": {
             "to-remove": ["A"]
          }
        }
        ```

        The operator will be deleted from the trustlist and added to the blocklist.

    3. \(Negative example\) Simply delete the operator from the trustlist. In this case, the operator will be moved to the graylist instead of the blocklist.

        ```json
        {
          "white-list": {
             "to-remove": ["A"]
          }
        }
        ```

        The operator will be deleted from the trustlist and added to the graylist.

        > [!NOTE]NOTE
        > If an operator is simply removed from the blocklist or trustlist, it will be added to the graylist.

- Modify the operator information library.

    > [!CAUTION]NOTICE
    > the built-in operator information library may affect other networks. Proceed with caution.

    1. Go to  **/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>**  under the CANN installation directory.
    2. Grant the write permission on the  **aic-<soc_version\>-ops-info-<opType\>.json**  file.

        ```bash
        chmod u+w aic-<soc_version>-ops-info-<opType>.json
        ```

        All .json files in the current directory will be loaded to the operator information library. If you need to back up the original .json files, back them up to another directory.

    3. Modify or add the  **precision_reduce**  field of the corresponding operator in the  **aic-<soc_version\>-ops-info-<opType\>.json**  file in the operator information library.
