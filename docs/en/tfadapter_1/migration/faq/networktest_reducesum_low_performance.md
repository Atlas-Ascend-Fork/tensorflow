# What Do I Do If Operator ReduceSum Has Poor Performance on a Network?

## Symptom

During network debugging, the overall performance is low. The ReduceSum operator shows low performance according to the network's profiling result \(generated after profile data collection and analysis\).

View details about the ReduceSum operator in the profile data. The following table lists the key fields in bold.

| op_type | block_dim | input_shape | input_data_type | input_formats |
| --- | --- | --- | --- | --- |
| ReduceSum | 1 | 1,256,256,3 | DT_FLOAT16 | NHWC |

The data type of ReduceSum's input is  **DT_FLOAT16**  and the value of  **block_dim**  is  **1**, which indicates that multiple blocks are not enabled for the operator.

## Solution

For a ReduceSum operator of the  AI processor, multiple blocks are unavailable for float16 inputs due to hardware restrictions.

Take the ReduceSum operator as an example. If the input data is float16, there are two solutions as follows:

- Mixed precision is not enabled during network debugging and ReduceSum's input is of type float16. In this case, if ReduceSum's performance is not as expected, you can insert a Cast operator before the ReduceSum operator to cast the data type from float16 to float32.

    Multiple blocks can be enabled when the input data is of the float32 type. As such, the operator performance can be improved.

- The mixed precision is enabled during network debugging and ReduceSum's input data type is cast from float32 to float16. In this case, you can add the ReduceSum operator to the blocklist for mixed precision to avoid the data type being cast to float16 during network debugging, preventing the ReduceSum operator from performance deterioration.

    To add the ReduceSum operator to the blocklist for mixed precision, perform the following steps:

    1. Modify the network script and specify the operator blocklist for mixed precision to be modified by using  **modify_mixlist**.

        Example:

        ```python
        # In Estimator mode
        npu_config=NPURunConfig(
          ...
          precision_mode="allow_mix_precision",
          modify_mixlist="/home/test/ops_info.json"
          )

        # In sess.run mode
        config = tf.ConfigProto()
        custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name =  "NpuOptimizer"
        custom_op.parameter_map["use_off_line"].b = True
        custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
        custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
        ...
        ```

    2. Configure the operator graylist in the  **ops_info.json**  file. The following is a configuration example.

        ```json
        {
            "black-list": {
                "to-add": ["ReduceSumD"]
            }
        }
        ```

        For details, see  [Modifying the Blocklist, Trustlist, and Graylist for Mixed Precision](../performance_tuning/mixed_precision_training.md#modifying-the-blocklist-trustlist-and-graylist-for-mixed-precision).

> [!CAUTION]NOTICE
>This solution is dedicated to improving operator ReduceSum's performance under conditions described in  [Symptom](#symptom).
