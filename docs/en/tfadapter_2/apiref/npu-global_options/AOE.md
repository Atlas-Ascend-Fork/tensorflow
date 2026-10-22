# AOE

## aoe_config.aoe_mode

Tuning mode of AOE.

- 1: subgraph tuning.
- 2: operator tuning.
- 4: gradient splitting tuning.

  In the data parallel scenario, AllReduce is used to aggregate gradients. The gradient splitting mode is closely related to the distributed training performance. Improper splitting may result in a long communication tailing from the completion of backward propagation, affecting the cluster training performance and linearity. It is sophisticated to perform manual tuning through the gradient splitting API (set_split_strategy_by_idx or set_split_strategy_by_size) of collective communication. AOE collects profile data in the real-device environment and automatically looks for the optimal splitting strategy. You only need to set the obtained strategy to your network by passing it to the set_split_strategy_by_idx call.

> [!NOTE]NOTE
>
> - The tuning mode can be configured by modifying the training script or the AOE_MODE environment variable. If both configuration methods are used, the configuration by modifying the training script takes precedence.
> - For the Atlas A2 products, subgraph tuning is not supported.
> - For the Atlas A3 products, subgraph tuning is not supported.

Example:

```python
npu.global_options().aoe_config.aoe_mode="1"
```

## aoe_config.work_path

Working directory of AOE, which stores the configuration and result files. By default, the files are generated in the current directory.

The value is a string. Create the specified directory in advance in the environment (either container or host) where training is performed. The running user configured during installation must have the read and write permissions on this directory. The value can be an absolute path or a path relative to the path where the training script is executed.

- An absolute path starts with a slash (/), for example, /home/HwHiAiUser/output.
- A relative path starts with a directory name, for example, output.

Example:

```python
npu.global_options().aoe_config.work_path = "/home/HwHiAiUser/output"
```

## aoe_config.aoe_config_file

Tunes only operators with low performance on the network with AOE. Set this option to the path and name of the configuration file that contains the operator information, for example, /home/test/cfg/tuning_config.cfg.

Example:

```python
npu.global_options().aoe_config.aoe_config_file="/home/test/cfg/tuning_config.cfg"
```

The configuration file contains information about the operators to be tuned. The file content format is as follows:

```text
{
       "tune_ops_name":["bert/embeddings/addbert/embeddings/add_1","loss/MatMul"],
       "tune_ops_type":["Add", "Mul"],
       "tune_optimization_level":"O1",
       "feature":["deeper_opat"]
}
```

- tune_ops_name: name of the specified operator (whole word match). You can specify one or more operator names. If multiple operator names are specified, separate them with commas (,). The operator name must be the node name of the network model processed by Graph Compiler. You can obtain the operator name from profiling tuning data. For details, see `Performance Tuning Tool`.
- tune_ops_type: specified operator type (whole word match). You can specify one or more operator types. If multiple operator types are specified, separate them with commas (,). If a fused operator contains the specified operator type, the fused operator will also be tuned.
- tune_optimization_level: tuning mode. The value O1 indicates the high-performance tuning mode, and the value O2 indicates the normal mode. The default value is O2.
- feature: tuning feature switch. The value can be deeper_opat or nonhomo_split. The value deeper_opat indicates that in-depth operator tuning is enabled. In this case, aoe_mode must be set to 2. The value nonhomo_split indicates that non-uniform subgraph partition tuning is enabled. In this case, aoe_mode must be set to 1.

> [!NOTE]NOTE
> In the preceding configuration file, tune_ops_type and tune_ops_name can exist at the same time or one of them. If they exist at the same time, use the union set.
