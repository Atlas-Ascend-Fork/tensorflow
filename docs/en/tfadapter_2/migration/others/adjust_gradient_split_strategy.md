# Adjusting Gradient Splitting Strategy

## Background

In distributed training, gradient aggregation is performed after gradients are computed across devices. Since gradient data is generated sequentially and remains unchanged after generation, the gradient parameter data can be split to improve training performance. Gradient aggregation for a segment can start immediately after its gradient data is generated, enabling parallel execution of gradient aggregation for some parameters and the forward and backward propagation.

The default splitting strategy is two segments with the first taking up 96.54% of the data volume, and the second segment taking up 3.46% of the data volume \(in some cases, the data is not split\). This splitting strategy may not be applicable to other networks due to the data volume and calculation time differences of different network gradients. You can adjust the distributed gradient splitting strategy by referring to this section to improve the training performance in distributed scenarios.

## Determining Gradient Splitting Strategy

You need to use the Profiling tool to analyze the iteration traces of the training process to determine the gradient splitting strategy and improve the training performance in distributed scenarios.

Iteration tracing is to trace the software status of a training job and the Ascend AI Software Stack, which can be used to analyze the performance of a training job. If the default two-segment gradient segmentation policy is applied, the following iteration traces of a training job are printed to describe the job execution status in an iteration:  **fp_start**,  **bp_end**,  **allreduce1_start**,  **allreduce1_end**,  **allreduce2_start**,  **allreduce2_end**, and  **Iteration_end**  in the training job.

![](../figures/gradient_split_strategy_1.png)

An optimal gradient data splitting strategy meets the following rules:

- Make AR1 hidden within the FPBP period. By default, AllReduce and forward and backward propagation proceed in serial. You can enable the parallel execution of AllReduce and forward and backward propagation by setting  **hcom_parallel**  to  **True**. For details about the configuration method, see  [Adjusting Gradient Splitting Strategy](#adjusting-gradient-splitting-strategy-1).
- Keep AR2 as short as possible to reduce the collective communication hangover after computation.

Based on the preceding splitting strategies, you can adjust the gradient splitting strategy to improve the training performance in distributed scenarios. The following uses two-segment gradient splitting as an example to describe how to determine a gradient splitting strategy with three optimization scenarios.

\[Scenario 1\] When AR1 starts early and AR2 is long, move backward the splitting point to shorten AR2.

Assume the first and second gradient segments have the data size of 50% each.

![](../figures/gradient_split_strategy_2.png)

If the data size of the first gradient segment is decreased to 80%, then that of the second gradient segment is 20%.

![](../figures/gradient_split_strategy_3.png)

\[Scenario 2\] When AR1 starts late and ends later than FPBP, move forward the splitting point to hide AR1 within FPBP.

Assume the first gradient segment has the data size of 90%, and the second gradient segment has the data size of 10%.

![](../figures/gradient_split_strategy_4.png)

If the data size of the first gradient segment is decreased to 80%, then that of the second gradient segment is 20%.

![](../figures/gradient_split_strategy_5.png)

\[Scenario 3\] You may get a long hangover when AR1 has most gradient data, especially when FPBP is time-consuming and data-hungry in two-segment gradient splitting. In this case, refer to scenario 2. If AR2 is long because it takes most gradient data, see scenario 1. You may choose to add more segments within FPBP, which is quite long, for better parallelism.

![](../figures/gradient_split_strategy_6.png)

## Adjusting Gradient Splitting Strategy

You can call the gradient splitting API in the training script to set the AllReduce splitting and fusion strategy in the backward propagation phase. Select either of the following APIs:

**set_split_strategy_by_idx**: sets the gradient splitting strategy in the collective communication group based on the gradient index.

```python
from hccl.split.api import set_split_strategy_by_idx
set_split_strategy_by_idx([20, 100, 159])
```

**set_split_strategy_by_size**: sets the gradient splitting strategy in the collective communication group by percent.

```python
from hccl.split.api import set_split_strategy_by_size
set_split_strategy_by_size([60, 20, 20])
```
