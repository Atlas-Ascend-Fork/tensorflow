# Automatic AOE Tuning

The AOE tool continuously iterates tuning policies through a closed-loop feedback mechanism of policy generation, compilation, and verification in the operating environment, and finally obtains the optimal one. This helps fully utilize hardware resources and improve network performance. During model training, the AOE tool can be enabled to tune subgraphs, operators, and gradients. After the tuning is complete, the optimal tuning policy is added to the repository. When the model is trained again, you can directly use the repository for efficient tuning, without enabling the tuning function.

> [!CAUTION]NOTICE
>The AOE tuning feature supports only the following products:
>
>- Atlas A3 products
>- Atlas A2 products
>- Atlas training products

You are advised to use the AOE tool to perform tuning in the following sequence:

![](../figures/aoe_tune.png)

> [!NOTE]NOTE
> Subgraph tuning is not supported for the  Atlas A3 products, Atlas A2 products.

You can enable AOE tuning in training scenarios using either of the following methods:

- Set the environment variable.

    ```bash
     # 1: subgraph tuning; 2: operator tuning; 4: gradient tuning
    export AOE_MODE=2
    ```

- Modify the training script and use  **aoe_mode**  to specify the tuning mode, for example:
  - In  **sess.run**  mode, modify the training script as follows:

    ```python
    custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
    ```

  - In  **Estimator**  mode, modify the training script as follows:

    ```python
    config = NPURunConfig(
        session_config=session_config,
        aoe_mode=2)
    ```

  - In  **keras**  mode, modify the training script as follows:

    ```python
    custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
    ```

For details about the restrictions and functions of the AOE tool, see  [AOE Tuning Tool](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/devaids/aoe/auxiliarydevtool_aoe_0001.html).
