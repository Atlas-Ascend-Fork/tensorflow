# Adjusting the Gradient Splitting Strategy

## Background

In the distributed training scenario, gradient aggregation is performed after gradients between devices are calculated. Gradient data is generated in order and does not change after being generated. To improve training performance, the gradient parameter data may be segmented. Gradient aggregation may be immediately started after gradient data of a segment is generated, so that some gradient parameter data is aggregated and forward and backward time is executed in parallel.

The default splitting strategy is two segments with the first taking up 96.54% of the data volume, and the second segment taking up 3.46% of the data volume \(in some cases, the data is not split\). This splitting strategy may not be applicable to other networks due to the data volume and calculation time differences of different network gradients. You can adjust the distributed gradient splitting strategy by referring to this section to improve the training performance in distributed scenarios.

## Determining the Gradient Splitting Strategy

You need to use the Profiling tool to analyze the iteration traces of the training process to determine the gradient splitting strategy and improve the training performance in distributed scenarios.

> [!NOTE]NOTE
>For details, see  [Performance Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0001.html).

Iteration tracing is to trace the software status of a training job and the AI software stack, which can be used to analyze the performance of a training job. If the default two-segment gradient splitting strategy is applied, the following iteration traces of a training job are printed to describe the job execution status in an iteration:  **fp_start**,  **bp_end**,  **allreduce1_start**,  **allreduce1_end**,  **allreduce2_start**,  **allreduce2_end**, and  **Iteration_end**  in the training job.

![](../figures/gradient_split_strategy_1.png)

An optimal gradient data splitting strategy meets the following rules:

- Make AR1 hidden within the FPBP period. By default, Allreduce and forward and backward propagation proceed in serial. You can enable the parallel execution of Allreduce and forward and backward propagation by setting  **hcom_parallel**  to  **True**. For details, see  [Adjusting the Gradient Splitting Strategy](#adjusting-the-gradient-splitting-strategy-1).
- Keep AR2 as short as possible to reduce the collective communication hangover after computation.

Based on the preceding splitting strategies, you can adjust the gradient splitting strategy to improve the training performance in distributed scenarios. The following uses two-segment gradient splitting as an example to describe how to determine a gradient splitting strategy with three optimization scenarios.

\[Scenario 1\] When AR1 starts early and AR2 is long, move backward the splitting point to shorten AR2.

For example, in the original setting, the two gradient segments each account for 50% of the total data volume.

![](../figures/gradient_split_strategy_2.png)

It can be modified so that the first gradient segment accounts for 80% of the total data volume and the second gradient segment accounts for 20%.

![](../figures/gradient_split_strategy_3.png)

\[Scenario 2\] When AR1 starts late and ends later than FPBP, move forward the splitting point to hide AR1 within FPBP.

For example, in the original setting, the first gradient segment accounts for 90% of the total data volume and the second gradient segment accounts for 10%.

![](../figures/gradient_split_strategy_4.png)

It can be modified so that the first gradient segment accounts for 80% of the total data volume and the second gradient segment accounts for 20%.

![](../figures/gradient_split_strategy_5.png)

\[Scenario 3\] You may get a long hangover when AR1 has most gradient data, especially when FPBP is time-consuming and data-hungry in two-segment gradient splitting. In this case, refer to scenario 2. If AR2 is long because it takes most gradient data, see scenario 1. However, there is still a relatively large amount of time in the FPBP to be utilized. In this case, more splitting points may be added, to improve the parallelism.

![](../figures/gradient_split_strategy_6.png)

## Adjusting the Gradient Splitting Strategy

You can use the gradient splitting API in the training script to set the AllReduce splitting and fusion policy in the backward propagation phase.

**set_split_strategy_by_idx**: sets the backward gradient splitting strategy in the collective communication group based on the gradient index ID.

```python
from hccl.split.api import set_split_strategy_by_idx
set_split_strategy_by_idx([20, 100, 159])
```

**set_split_strategy_by_size**: sets the backward gradient splitting strategy in the collective communication group based on the gradient data volume percentage.

```python
from hccl.split.api import set_split_strategy_by_size
set_split_strategy_by_size([60, 20, 20])
```

For details about the detailed API description of  **set_split_strategy_by_idx**  and  **set_split_strategy_by_size**, see "API Reference" in  [Huawei Collective Communication Library \(HCCL\)](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html).

Call either of the preceding APIs before the Allreduce call \(the collective communication API must be initialized\).

```python
import tensorflow as tf
from npu_bridge.npu_init import *

npu_init = npu_ops.initialize_system()
npu_shutdown = npu_ops.shutdown_system()

config = tf.ConfigProto()
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
# Enable iteration tracing.
custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","fp_point":"","bp_point":""}')
# Enable the parallel execution of AllReduce and forward and backward propagation.
custom_op.parameter_map["hcom_parallel"].b = True
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
    # Initialize collective communication.
    sess.run(npu_init)
    # Set the backward gradient splitting strategy.
    set_split_strategy_by_size([80, 20])
    # Perform AllReduce...
    # Perform training...
    sess.run(npu_shutdown)
```
