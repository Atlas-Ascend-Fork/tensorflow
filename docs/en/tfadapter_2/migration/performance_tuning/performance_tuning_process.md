# Performance Tuning Process

If the performance of the network ported to the  Ascend AI Processor  for training is not satisfactory, you can perform the following steps to tune the performance.

**Figure  1**  Performance tuning process of TensorFlow network
![](../figures/performance_tuning_process.png)

1. If the performance is not satisfactory, you are advised to perform the following common operations to improve it:

    1. Enable the automatic mixed precision mode.
    2. Replace the GELU activation function.
    3. Use the AOE tool to tune subgraphs, operators, and gradient segmentation policies.

        > [!CAUTION]NOTICE
        > Ascend 950PR&950DT products do not support the AOE tool.

    For details, see  [Basic Tuning](basic_tuning.md).

2. Perform model training again and evaluate whether the training performance is satisfactory.
    - If the performance is satisfactory, the tuning is complete.
    - If the performance is not satisfactory, go to step3.

3. Use the Profiling tool to collect and analyze profile data.

    Refer to  [Profile Data Collection and Analysis](profile_data_collection_and_analysis.md)  to collect, parse, export, and analyze profile data.

4. Refer to  [Advanced Tuning](advanced_tuning.md)  to further improve the performance based on the identified performance bottleneck.
5. Perform model training again, conduct a regression test, and evaluate whether the training performance is satisfactory.
    - If the performance is satisfactory, the tuning is complete.
    - If the performance is not satisfactory for the following  Product, execute operations in  [Automatic AOE Tuning](automatic_aoe_tuning.md)  again.

        Atlas A3 products

        Atlas A2 products

        Atlas training products

    - If the performance is not satisfactory for the  Ascend 950PR&950DT products, execute operations in  [Profile Data Collection and Analysis](profile_data_collection_and_analysis.md)  again.
