# 网络调测时ReduceSum算子性能差

## 现象描述

网络调测时，网络整体性能较慢。通过Profiling工具获取网络的Profiling数据，并进行算子的性能数据分析，发现ReduceSum算子的性能很差。

查看Profiling性能数据中ReduceSum算子的详细信息，关键字段如下表粗体部分所示：

| op_type | block_dim | input_shape | input_data_type | input_formats |
|---------|-----------|-------------|-----------------|---------------|
| ReduceSum | 1 | 1,256,256,3 | DT_FLOAT16 | NHWC |

其中，ReduceSum算子的输入数据类型（input_data_type）为“DT_FLOAT16”，block_dim字段的值为“1”，说明该算子未开启多核并行计算。

## 解决方案

对于AI处理器的ReduceSum算子，若输入的数据类型为float16，由于硬件限制，某些场景下无法开启多核计算。

以ReduceSum算子为例，输入数据是float16的情况可能有如下两种场景：

- 网络调测时未开启混合精度，ReduceSum算子的输入数据本身就是float16类型，此种情况下，若ReduceSum算子的性能未满足预期，可尝试在ReduceSum算子前插入一个Cast算子，将算子的输入数据类型从float16转换为float32。

    ReduceSum算子在输入类型为float32的场景下，会使能多核并发计算，从而达到提升该算子性能的效果。

- 网络调测时开启了混合精度，将ReduceSum算子的输入数据类型从float32转换成了float16，此种情况下，可将ReduceSum算子加入混合精度黑名单，这样网络调测时ReduceSum算子就不会被转换成float16类型，从而避免该算子性能的劣化。

    将ReduceSum算子加入混合精度黑名单的方法如下：

    1. 修改网络脚本，通过modify_mixlist指定需要修改的混合精度算子黑名单。

        例如：

        ```python
        # Estimator模式修改方法
        npu_config=NPURunConfig(
          ...
          precision_mode="allow_mix_precision",
          modify_mixlist="/home/test/ops_info.json"
          )

        # sess.run模式修改方法
        config = tf.ConfigProto()
        custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name =  "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
        ...
        ```

    2. 在ops_info.json文件中进行算子黑名单的配置，配置示例如下：

        ```json
        {
            "black-list": {
                "to-add": ["ReduceSumD"]
            }
        }
        ```

        详细配置方法可参见[修改混合精度黑白灰名单](../performance_tuning/mixed_precision_training.md#修改混合精度黑白灰名单)。

> [!NOTE]说明
> 仅在ReduceSum算子性能较差时，且符合[现象描述](#现象描述)时，可尝试使用此方法进行性能提升。
