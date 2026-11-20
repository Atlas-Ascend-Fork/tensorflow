# 混合精度训练

## 混合精度简介

混合精度为业内通用的性能提升方式，通过降低部分计算精度提升数据计算的并行度。混合精度训练方法是通过混合使用float16和float32数据类型来加速深度神经网络训练的过程，并减少内存使用和存取，从而可以训练更大的神经网络，同时又能基本保持使用float32训练所能达到的网络精度。

用户可以在脚本中通过配置“precision_mode_v2”（推荐）参数或者“precision_mode”参数开启混合精度。

关于“precision_mode_v2”与“precision_mode”参数的详细说明可参见[精度调优](../accuracy_debugging/accuracy_debugging.md)。

开启“自动混合精度”的场景下，推荐使用LossScale优化器（LossScale优化器的迁移请参见[替换LossScaleOptimizer](../script_migration/manual_porting.md#替换lossscaleoptimizer)），从而补偿降低精度带来的精度损失；若后续进行Profiling数据进行分析时，发现需要手工调整某些算子的精度模式，可以参考[修改混合精度黑白名单](#修改混合精度黑白名单)自行指定哪些算子允许降精度，哪些算子不允许降精度。

## 精度模式设置

下面以将“precision_mode_v2”参数配置为“mixed_float16”为例，说明如何设置混合精度模式。

修改训练脚本，在初始化NPU设备前通过添加[precision_mode_v2](../../apiref/npu-global_options/accuracy_tuning.md#precision_mode_v2)参数设置精度模式。

```python
import npu_device as npu
npu.global_options().precision_mode_v2 = 'mixed_float16'  # 开启自动混合精度功能，表示混合使用float16和float32数据类型来处理神经网络的过程
npu.open().as_default()
```

## 修改混合精度黑白名单

开启自动混合精度的场景下，系统会自动根据内置的优化策略，对网络中的某些数据类型进行降精度处理，从而在精度损失很小的情况下提升系统性能并减少内存使用。

内置优化策略在“CANN软件安装目录/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json“，例如：

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

- precision_mode_v2配置为mixed_float16，precision_mode配置为allow_mix_precision_fp16/allow_mix_precision的场景：
  - 若取值为true（白名单），则表示允许将当前float32类型的算子，降低精度到float16。
  - 若取值为false（黑名单），则不允许将当前float32类型的算子降低精度到float16，相应算子仍使用float32精度。
  - 若网络模型中算子没有配置该参数（灰名单），当前算子的混合精度处理机制和前一个算子保持一致，即如果前一个算子支持降精度处理，当前算子也支持降精度；如果前一个算子不允许降精度，当前算子也不支持降精度。

- precision_mode配置为allow_mix_precision_bf16的场景（仅Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置）：
  - 若取值为true（白名单），则表示允许将当前float32类型的算子，降低精度到bfloat16。
  - 若取值为false（黑名单），则不允许将当前float32类型的算子降低精度到bfloat16，相应算子仍旧使用float32精度。
  - 若网络模型中算子没有配置该参数（灰名单），当前算子的混合精度处理机制和前一个算子保持一致，即如果前一个算子支持降精度处理，当前算子也支持降精度；如果前一个算子不允许降精度，当前算子也不支持降精度。

用户可以在内置优化策略基础上进行调整，自行指定哪些算子允许降精度，哪些算子不允许降精度。

- （推荐）通过modify_mixlist参数指定混合精度黑白灰算子名单

    修改训练脚本，在初始化NPU设备前通过添加[modify_mixlist](../../apiref/npu-global_options/accuracy_tuning.md#modify_mixlist)参数指定混合精度黑白灰算子名单配置文件，配置示例如下：

    ```python
    import npu_device as npu
    npu.global_options().modify_mixlist = "/home/test/ops_info.json"
    npu.open().as_default()
    ```

    其中ops_info.json为混合精度黑白灰算子名单配置文件，多个算子使用英文逗号分隔，样例如下：

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

    假设算子A默认在白名单中，如果您希望将该算子配置为黑名单算子，可以参考如下方法：

    1. （正确示例）用户将该算子添加到黑名单中：

        ```json
        {
          "black-list": {
             "to-add": ["A"]
          }
        }
        ```

        则系统会将该算子从白名单中删除，并添加到黑名单中。

    2. （正确示例）用户将该算子从白名单中删除，同时添加到黑名单中：

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

        则系统会将该算子从白名单中删除，并添加到黑名单中，最终该算子在黑名单中。

    3. （错误示例）用户将该算子从白名单中删除，此时算子最终是在灰名单中，而不是黑名单。

        ```json
        {
          "white-list": {
             "to-remove": ["A"]
          }
        }
        ```

        此时，系统会将该算子从白名单中删除，然后添加到灰名单中，最终该算子在灰名单中。

        > [!NOTE]说明
        > 对于只从黑/白名单中删除，而不添加到白/黑名单的情况，系统会将该算子添加到灰名单中。

- 修改算子信息库。

    > [!CAUTION]注意
    > 对内置算子信息库进行修改，可能会对其他网络造成影响，请谨慎修改。

    1. 切换到“CANN软件安装目录/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>“目录下。
    2. 对aic-<soc_version\>-ops-info-<opType\>.json文件增加写权限。

        ```bash
        chmod u+w aic-<soc_version>-ops-info-<opType>.json
        ```

        当前目录下的所有json文件都会被加载到算子信息库中，如果您需要备份原来的json文件，建议备份到其他目录下。

    3. 修改或增加算子信息库aic-<soc_version\>-ops-info-<opType\>.json文件中对应算子的precision_reduce字段。
