# AOE自动调优

AOE自动调优工具通过生成调优策略、编译、在运行环境上验证的闭环反馈机制，不断迭代出更优的调优策略，最终得到最佳的调优策略，从而可以更充分利用硬件资源，提升网络的性能。模型训练阶段，分别使能AOE工具进行子图/算子与梯度切分的调优，调优完成后，最优调度策略会固化到知识库。模型再次训练时，无需开启调优，即可以享受知识库带来的性能收益。

> [!NOTE]说明
> AOE调优特性仅支持如下产品：
>
> - Atlas A3系列产品
> - Atlas A2系列产品
> - Atlas训练系列产品

建议按照如下调优顺序使用AOE工具进行调优：

![](../figures/aoe_tune.png)

> [!NOTE]说明
> 针对Atlas A3系列产品，Atlas A2系列产品，不支持子图调优。

训练场景下使能AOE调优有两种方式：

- 设置环境变量

    ```bash
     # 1：子图调优  2：算子调优   4：梯度调优
    export AOE_MODE=2
    ```

- 修改训练脚本，通过“aoe_mode”参数指定调优模式，例如：
  - sess.run模式下，训练脚本修改方法如下：

    ```python
    custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
    ```

  - Estimator模式下，训练脚本修改方法如下：

    ```python
    config = NPURunConfig(
        session_config=session_config,
        aoe_mode=2)
    ```

  - Keras模式下，训练脚本的修改方法如下：

    ```python
    custom_op.parameter_map["aoe_mode"].s = tf.compat.as_bytes("2")
    ```

关于AOE工具的使用约束及更多功能介绍，请参见《[AOE调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolAoe)》。
