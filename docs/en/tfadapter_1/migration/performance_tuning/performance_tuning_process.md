# Performance Tuning Process

If the performance of the network ported to the  AI processor  for training is not satisfactory, you can perform the following steps to tune the performance:

**Figure  1**  Performance tuning process
![](../figures/performance_tune_process.png)

1. If the performance is not satisfactory, you are advised to perform the following common operations to improve it:

    1. Enable the automatic mixed precision mode.
    2. Check whether the affinity interfaces are replaced.
    3. Enable iteration offload.
    4. Use the AOE tool to tune subgraphs, operators, and gradient segmentation policies.

        > [!CAUTION]NOTICE
        > The  Ascend 950PR&950DT products  does not support the AOE tool.

    For details, see  [Basic Tuning](basic_tuning.md).

2. Perform model training again and evaluate whether the training performance is satisfactory.
    - If the performance is satisfactory, the tuning is complete.
    - If the performance is not satisfactory, go to step3.

3. Use the Profiling tool to collect and analyze profile data.

    Refer to [Profile Data Collection and Analysis](Profile_data_collection_and_analysis.md) to collect, parse, export, and analyze profile data.

4. Refer to [Advanced Tuning](advanced_tuning.md)  to further improve the performance based on the identified performance bottleneck.
5. Perform model training again, conduct a regression test, and evaluate whether the training performance is satisfactory.
    - If the performance is satisfactory, the tuning is complete.
    - If the performance does not meet the requirements for the following  Product, perform the operations in  [Automatic AOE Tuning](automatic_aoe_tuning.md)  again.

        Atlas A3 products

        Atlas A2 products

        Atlas training products

    - If the performance does not meet the requirements for the  Ascend 950PR&950DT products, perform the operations in  [Profile Data Collection and Analysis](Profile_data_collection_and_analysis.md) again.
