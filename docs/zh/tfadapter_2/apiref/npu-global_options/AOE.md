# AOE

## aoe_config.aoe_mode

通过AOE工具进行调优的模式

- 1：子图调优。
- 2：算子调优。
- 4：梯度切分调优。

  在数据并行的场景下，使用allreduce对梯度进行聚合，梯度的切分方式与分布式训练性能强相关，切分不合理会导致反向计算结束后存在较长的通信拖尾时间，影响集群训练的性能和线性度。用户可以通过集合通信的梯度切分接口（set_split_strategy_by_idx或set_split_strategy_by_size）进行人工调优，但难度较高。因此，可以通过工具实现自动化搜索切分策略，通过在实际环境预跑采集性能数据，搜索不同的切分策略，理论评估出最优策略输出给用户，用户拿到最优策略后通过set_split_strategy_by_idx接口设置到该网络中。

> [!NOTE]说明
>
> - 通过修改训练脚本和AOE_MODE环境变量都可配置调优模式，同时配置的情况下，通过修改训练脚本方式优先生效。
> - 针对Atlas A2系列产品，不支持子图调优。
> - 针对Atlas A3系列产品，不支持子图调优。

配置示例：

```python
npu.global_options().aoe_config.aoe_mode="1"
```

## aoe_config.work_path

AOE工具调优工作目录，存放调优配置文件和调优结果文件，默认生成在训练当前目录下

该参数类型为字符串，指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径（相对执行命令行时的当前路径）。

- 绝对路径配置以“/”开头，例如：/home/HwHiAiUser/output。
- 相对路径配置直接以目录名开始，例如：output。

配置示例：

```python
npu.global_options().aoe_config.work_path = "/home/HwHiAiUser/output"
```

## aoe_config.aoe_config_file

通过AOE工具进行调优时，若仅针对网络中某些性能较低的算子进行调优，可通过此参数进行设置。该参数配置为包含算子信息的配置文件路径及文件名，例如：/home/test/cfg/tuning_config.cfg

配置示例：

```python
npu.global_options().aoe_config.aoe_config_file="/home/test/cfg/tuning_config.cfg"
```

配置文件中配置的是需要进行调优的算子信息，文件内容格式如下：

```text
{
       "tune_ops_name":["bert/embeddings/addbert/embeddings/add_1","loss/MatMul"],
       "tune_ops_type":["Add", "Mul"],
       "tune_optimization_level":"O1",
       "feature":["deeper_opat"]
}
```

- tune_ops_name：指定的算子名称，当前实现是支持全字匹配，可以指定一个，也可以指定多个，指定多个时需要用英文逗号分隔。此处配置的算子名称需要为经过图编译器处理过的网络模型的节点名称，可从Profiling调优数据中获取，详细可参见《性能调优工具》。
- tune_ops_type：指定的算子类型，当前实现是支持全字匹配，可以指定一个，也可以指定多个，指定多个时需要用英文逗号分隔。如果有融合算子包括了该算子类型，则该融合算子也会被调优。
- tune_optimization_level：调优模式，取值为O1表示高性能调优模式，取值为O2表示正常模式。默认值为O2。
- feature：调优功能特性开关，可以取值为deeper_opat或者nonhomo_split，取值为deeper_opat时，表示开启算子深度调优，aoe_mode需要配置为2；取值为nonhomo_split时，表示开启子图非均匀切分，aoe_mode需要配置为1。

> [!NOTE]说明
> 如上配置文件中，tune_ops_type和tune_ops_name可以同时存在，同时存在时取并集，也可以只存在某一个。
