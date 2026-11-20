# 调整梯度切分策略

## 背景介绍

分布式训练场景下，各个Device之间计算梯度后执行梯度聚合操作。由于梯度数据会按顺序产生，且产生后不会再变化，为了提高训练性能，我们可以对梯度参数数据进行切分，同一分段中的梯度数据在产生后可以立即开始梯度聚合，使得一部分梯度参数数据聚合和前后向时间并行执行。

系统默认切分策略是：按照梯度数据量切分为两段，第一段梯度数据量为96.54%，第二段梯度数据量为3.46%（可能出现一段的情况）。因不同网络梯度数据量、梯度计算时间差异，该切分方式可能不适用于其他的网络，用户可以参考本节内容调整分布式梯度切分策略，从而提升分布式场景下的训练性能。

## 梯度切分策略确定

用户需要结合Profiling工具分析训练过程的迭代轨迹数据（Training Trace），确定梯度切分策略需要调整到什么值以达到分布式场景下训练性能提升的目标。

迭代轨迹数据即训练任务及AI软件栈的软件信息，实现对训练任务的性能分析。以默认的两段式梯度切分为例，通过打印出训练任务中关键节点fp_start/bp_end/allreduce1_start/allreduce1_end/allreduce2_start/allreduce2_end/Iteration_end的时间戳，达到把一个迭代的执行情况描述清楚的目的。

![确定梯度切分策略](../figures/gradient_split_strategy_1.png)

一个较优的梯度数据切分原则为：

- AR1隐藏在FPBP之间。默认情况下Allreduce和前后向串行执行，此时需要配置**hcom_parallel**为True开启Allreduce和前后向并行执行。具体的配置方法将在代码示例“[调整梯度切分策略](#梯度切分策略调整)”中介绍。
- AR2的时间尽可能短，从而减少计算后因为集合通信而带来的拖尾时间的消耗。

以上述切分原则为依据，用户可以调整梯度切分策略，从而提升分布式场景下的训练性能。下面以两段式梯度切分为例，结合三种优化场景，帮助用户理解如何确定梯度切分策略。

【优化场景1】AR1开始时间较早，AR2时间较长，这种情况下可以将切分点往后设置，从而尽可能缩短AR2的时间。

例如原始设置为第一段梯度数据量为50%，第二段梯度数据量为50%：

![](../figures/gradient_split_strategy_2.png)

可以修改为第一段梯度数据量为80%，第二段梯度数据量为20%：

![](../figures/gradient_split_strategy_3.png)

【优化场景2】AR1开始时间较晚，AR1时间超出了FPBP的时间，这种情况下，可以将切分点往前设置，从而将AR1隐藏在FPBP之间。

例如原始设置为第一段梯度数据量为90%，第二段梯度数据量为10%：

![](../figures/gradient_split_strategy_4.png)

可以修改为第一段梯度数据量为80%，第二段梯度数据量为20%：

![](../figures/gradient_split_strategy_5.png)

【优化场景3】FPBP数据量较大，计算时间较长，两段式梯度切分的情况下，如果把大部分梯度数据放到AR1中会导致拖尾时间很长，参见优化场景2；如果把大部分的梯度数据放到AR2中会导致AR2时间很长，参见优化场景1。但FPBP中还有较多的时间可以利用，此时可以新增切分段数，使更多的集合通信时间和FPBP并行起来。

![](../figures/gradient_split_strategy_6.png)

## 梯度切分策略调整

用户可以在训练脚本中调用梯度切分类接口来设置反向计算阶段的allreduce切分融合策略，以下接口二选一使用。

set_split_strategy_by_idx：基于梯度的索引id，在集合通信group内设置反向梯度切分策略。

```python
from hccl.split.api import set_split_strategy_by_idx
set_split_strategy_by_idx([20, 100, 159])
```

set_split_strategy_by_size：基于梯度数据量百分比，在集合通信group内设置反向梯度切分策略。

```python
from hccl.split.api import set_split_strategy_by_size
set_split_strategy_by_size([60, 20, 20])
```
