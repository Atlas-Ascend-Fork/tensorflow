# Profiling数据采集与分析

## 功能介绍

若基本调优操作不能达到性能要求，您可以借助Profiling工具采集训练过程中的性能数据并进行分析，从而准确定位系统的软、硬件性能瓶颈，提高性能分析的效率，通过针对性的性能优化方法，以最小的代价和成本提升业务性能。

默认训练过程中不采集Profiling性能数据，如需采集、解析性能数据，请参考本节内容。

TensorFlow网络性能数据采集与分析的整体流程如下所示：

![](../figures/profiling_overview.png)

1. 采集Profiling数据。

    TensorFlow网络的性能数据支持全局采集与局部采集两种方式：

    - 全局采集，采集图执行所有行为的性能数据，数据量较大。
    - 局部采集，支持指定采集局部子图或者指定step的性能数据。

    全局采集有两种开启方式，可以通过修改训练脚本、配置“profiling_mode”参数的方式开启（可参见[全局采集Profiling数据（修改训练脚本方式）](#全局采集profiling数据修改训练脚本方式)），也可以通过设置环境变量“PROFILING_MODE”的方式开启（可参见[全局采集Profiling数据（环境变量方式）](#全局采集profiling数据环境变量方式)），其中配置参数“profiling_mode”的优先级高于环境变量“PROFILING_MODE”。

    局部采集，指通过with语句调用Profiler类，并将需要开启性能数据采集的操作放入Profiler类作用域内的方式，详细操作方式可参见[局部采集Profiling数据（调用Profiler类方式）](#局部采集profiling数据调用profiler类方式)。

2. 解析并导出Profiling数据。

    不管是何种方式采集的Profiling数据，您都可以通过msprof命令行方式进行数据解析，并将解析结果导出到指定目录，详细操作可参见[解析并导出Profiling数据](#解析并导出profiling数据)。

3. 分析Profiling数据。

    您可以通过解析导出得到的结果文件（例如Timeline、Summary文件）进行分析，识别性能瓶颈点，典型分析示例可参见[分析Profiling数据](#分析profiling数据)。

## 全局采集Profiling数据（修改训练脚本方式）

### Estimator模式

#### 自动迁移场景

1. 检查迁移后的脚本是否存在“init_resource”。
    - 如果存在，则需要参考下面示例进行修改；修改完后，执行下一步。

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # 开启Profiling采集
          custom_op.parameter_map["profiling_mode"].b = True
          # 仅采集任务轨迹数据
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在npu_run_config_init函数的run_config中进行配置。

        > [!NOTE]说明
        >
        > - profiling_mode：是否开启Profiling采集。
        > - output：Profiling数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径。
        > - task_trace：是否采集任务轨迹数据。
        > - training_trace：是否采集迭代轨迹数据，training_trace配置为“on”的场景下需要同时配置fp_point和bp_point。
        > - aicpu：是否采集AI CPU算子的详细信息，如算子执行时间、数据拷贝时间等。
        > - fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        > - bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        > - aic_metrics：AI Core和AI Vector Core的硬件信息，取值“PipeUtilization”代表记录计算单元和搬运单元的耗时占比。
        > - Profiling配置的详细介绍请参考[Profiling](../../apiref/session_config/Profiling.md)。

    - 如果不存在，则执行下一步。

2. 在迁移后的脚本中找到“npu_run_config_init”，找到运行配置函数，例如示例中的run_config。

    如果运行配置函数中未传入session_config参数，则需要按照下面示例添加；如果已经传入了session_config参数，则进行下一步。

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60*60*24)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
    ```

3. 添加session_config配置，开启Profiling采集。

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # 开启Profiling采集
    custom_op.parameter_map["profiling_mode"].b = True
    # 仅采集任务轨迹数据
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        session_config=session_config,
        save_checkpoints_secs=60*60*24)

    classifier = tf.estimator.Estimator(
        model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
    ```

4. 重新执行训练脚本进行Profiling数据的采集。

#### 手工迁移场景

您可以尝试先开启task_trace任务轨迹数据采集：

```python
from npu_bridge.npu_init import *

# enable_profiling：是否开启Profiling采集
# output：Profiling数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径
# task_trace：是否采集任务轨迹数据
profiling_options = '{"output":"/home/test/output","task_trace":"on"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto()

config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```

（可选）后续如果仍然无法分析到具体问题，可再开启training_trace迭代轨迹数据采集：

```python
from npu_bridge.npu_init import *

# enable_profiling：是否开启Profiling采集
# output：Profiling数据存放路径
# task_trace：是否采集任务轨迹数据
# training_trace：是否采集迭代轨迹数据
# fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳
# bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，fp_point和bp_point可以计算出正反向时间
profiling_options = '{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
profiling_config = ProfilingConfig(enable_profiling=True, profiling_options= profiling_options)
session_config=tf.ConfigProto(allow_soft_placement=True)

config = NPURunConfig(profiling_config=profiling_config, session_config=session_config)
```

需要注意的是，采集迭代轨迹数据需要fp_point（训练网络迭代轨迹正向算子的开始位置）和bp_point（反向算子的结束位置），可直接配置为空，由系统自动获取，采集异常时可参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)进行配置。

相关接口详细介绍请参考[ProfilingConfig构造函数](../../apiref/profiler/Profiler_constructor.md)。

### sess.run模式

#### 自动迁移场景

1. 检查迁移后的脚本是否存在“init_resource”。
    - 如果存在，则需要参考下面示例进行修改；修改完后，执行下一步。

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # 开启Profiling采集
          custom_op.parameter_map["profiling_mode"].b = True
          # 仅采集任务轨迹数据
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在npu_config_proto函数的config_proto中进行配置。

        > [!NOTE]说明
        >
        >- profiling_mode：是否开启Profiling采集。
        >- output：Profiling数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径。
        >- task_trace：是否采集任务轨迹数据。
        >- training_trace：是否采集迭代轨迹数据，training_trace配置为“on”的场景下需要同时配置fp_point和bp_point。
        >- aicpu：是否采集AI CPU算子的详细信息，如算子执行时间、数据拷贝时间等。
        >- fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        >- bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        >- aic_metrics：AI Core和AI Vector Core的硬件信息，取值“PipeUtilization”代表记录计算单元和搬运单元的耗时占比。
        >- Profiling配置的详细介绍请参考[Profiling](../../apiref/session_config/Profiling.md)。

    - 如果不存在，则执行下一步。

2. 在迁移脚本中查找“npu_config_proto”函数，找到运行配置参数（例如下面示例中的“session_config”），在运行配置中配置相关参数，开启task_trace任务轨迹数据采集。

    ```python
    session_config = tf.ConfigProto(allow_soft_placement=True)
    custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # 开启Profiling采集
    custom_op.parameter_map["profiling_mode"].b = True
    # 仅采集任务轨迹数据
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
    config = npu_config_proto(config_proto=session_config)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.init.run()
    ```

#### 手工迁移场景

您可以尝试先开启task_trace任务轨迹数据采集：

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
  sess.run()
```

（可选）后续如果仍然无法分析到具体问题，可再开启training_trace迭代轨迹数据采集：

```python
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["use_off_line"].b = True
custom_op.parameter_map["profiling_mode"].b = True
custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

with tf.Session(config=config) as sess:
  sess.run()
```

需要注意的是，采集迭代轨迹数据需要fp_point（训练网络迭代轨迹正向算子的开始位置）和bp_point（反向算子的结束位置），可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。

相关接口详细介绍请参考[Profiling](../../apiref/session_config/Profiling.md)。

### Keras模式

1. 检查迁移后的脚本是否存在“init_resource”。
    - 如果存在，则需要参考下面示例进行修改；修改完后，执行下一步。

        ```python
        if __name__ == '__main__':

          session_config = tf.ConfigProto(allow_soft_placement=True)
          custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
          custom_op.name = "NpuOptimizer"
          # 开启Profiling采集
          custom_op.parameter_map["profiling_mode"].b = True
          # 仅采集任务轨迹数据
          custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
          # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
          # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')

          (npu_sess, npu_shutdown) = init_resource(config=session_config)
          tf.app.run()
          shutdown_resource(npu_sess, npu_shutdown)
          close_session(npu_sess)
        ```

        需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在set_keras_session_npu_config函数的config中进行配置。

        > [!NOTE]说明
        >
        >- profiling_mode：是否开启Profiling采集。
        >- output：Profiling数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径。
        >- task_trace：是否采集任务轨迹数据。
        >- training_trace：是否采集迭代轨迹数据，training_trace配置为“on”的场景下需要同时配置fp_point和bp_point。
        >- aicpu：是否采集AI CPU算子的详细信息，如算子执行时间、数据拷贝时间等。
        >- fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        >- bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，可直接配置为空，由系统自动获取，或参考[如何获取fp_point与bp_point](../common_operation/fpbp_point_determination.md)。
        >- aic_metrics：AI Core和AI Vector Core的硬件信息，取值“PipeUtilization”代表记录计算单元和搬运单元的耗时占比。
        >- Profiling配置的详细介绍请参考[Profiling](../../apiref/session_config/Profiling.md)。

    - 如果不存在，则执行下一步。

2. 在脚本中找到“set_keras_session_npu_config”，配置Profiling相关参数。

    ```python
    import tensorflow as tf
    import tensorflow.python.keras as keras
    from tensorflow.python.keras import backend as K
    from npu_bridge.npu_init import *

    config_proto = tf.ConfigProto(allow_soft_placement=True)
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = 'NpuOptimizer'
    # 开启Profiling采集
    custom_op.parameter_map["profiling_mode"].b = True
    # 仅采集任务轨迹数据
    custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on"}')
    # 采集任务轨迹数据和迭代轨迹数据。可先仅采集任务轨迹数据，如果仍然无法分析到具体问题，可再采集迭代轨迹数据
    # custom_op.parameter_map["profiling_options"].s = tf.compat.as_bytes('{"output":"/home/test/output","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}')
    npu_keras_sess = set_keras_session_npu_config(config=config_proto)

    #数据预处理...
    #模型搭建...
    #模型编译...
    #模型训练...
    ```

## 全局采集Profiling数据（环境变量方式）

除了通过修改训练脚本的方式采集Profiling数据外，用户还可以通过设置环境变量的方式开启Profiling采集功能。配置示例如下：

```bash
# 开启Profiling功能
export PROFILING_MODE=true
# 配置Profiling配置选项
export PROFILING_OPTIONS='{"output":"/home/test/output","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
```

环境变量PROFILING_OPTIONS的详细配置说明请参考《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》中的“性能数据采集”章节。

**需要注意，环境变量“PROFILING_MODE”的优先级低于训练脚本中的配置项“profiling_mode”。**

## 局部采集Profiling数据（调用Profiler类方式）

开发者可通过调用[npu_bridge.profiler.profiler](../../apiref/profiler/Profiler_constructor.md)类实现局部采集性能数据的功能，即仅Profiler类作用域下的命令才会开启性能数据采集功能。

下面介绍如何通过调用Profiler类实现采集局部性能数据的功能。

1. 引入Profiler类。

    ```python
    from npu_bridge.npu_init import *
    ```

2. 通过with语句调用Profiler类，并将需要做性能数据采集的操作包含在Profiler类的作用域内。

    如下是一段简单的示例代码片段，此代码片段构造了一个包含add算子的图，并在session中执行此图。其中sess.run\(add, ...\)函数在Profiler的作用域内，所以会执行L1级别的性能数据采集，并统计各种计算类指标占比统计，性能采集数据存储在当前脚本执行路径下。

    ```python
    a = tf.placeholder(tf.int32, (None,None))
    b = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    add = tf.add(a, b)

    with tf.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    当前，开发者可以采集指定step的性能数据，只需要将指定的step操作定义在相应的Profiler作用域内即可，如下所示：

    ```python
    a=tf.placeholder(tf.int32, (None,None))
    b=tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    d = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    add1 = tf.add(a, b)
    add2 = tf.add(c, d)
    add3 = tf.add(add1, add2)

    with tf.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="PipeUtilization", output_path = "/home/test/profiling_data"):
        for i in range(2):
          result=sess.run(add1, feed_dict={a: [[-20],[1]]})
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "/home/test/profiling_data"):
        for i in range(4):
          result=sess.run(add3, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    关于Profiler类的使用约束可参见[Profiler构造函数](../../apiref/profiler/Profiler_constructor.md)中的“约束说明”。

## 解析并导出Profiling数据

下面以msprof命令行方式进行数据解析与导出为例进行说明：

1. 切换到解析工具所在路径。

    ```bash
    cd ${INSTALL_DIR}/tools/profiler/bin
    ```

    其中$\{INSTALL_DIR\}请替换为CANN软件安装后文件存储路径。以root用户安装为例，安装后文件默认存储路径为：/usr/local/Ascend/cann。

2. 执行如下命令完成对Profiling数据文件的解析。

    ```bash
    ./msprof --parse=on --output=/home/test/profiling_output
    ```

    其中“--output”为采集Profiling数据时设置的存储Profiling数据文件的路径。

3. 导出Profiling数据。

    ```bash
    ./msprof --export=on --output=/home/test/profiling_output
    ```

    说明：关于Profiling工具的更详细说明，可参见《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》。

## 分析Profiling数据

开发人员可通过分析Profiling工具解析得到的Timeline和Summary文件，识别性能瓶颈点。

下面仅介绍常用的关键性能数据文件以及分析思路，关于更多Profiling数据文件的介绍可参见《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》。

- Timeline文件：step_trace_\*.csv文件

  “step_trace_\*.csv”记录了迭代轨迹数据信息，包含每轮迭代的耗时，主要字段及含义如下：

  - Iteration Time：一轮迭代的计算时间，主要包含FP/BP和Grad Refresh两个阶段的时间。
  - FP to BP Time：网络正向传播和反向传播的计算时间。
  - Iteration Refresh：迭代拖尾时间。
  - Data Aug Bound：两个相邻Iteration Time的间隔时间。

    展示数据如下图所示，开发者进行数据分析时，需要注意选择合适的Iteration ID和Model ID的数据。

     ![](../figures/step_trace_csv_file.png)

    从上面示例可以看出，Model ID=1的数据明显与后续不同，为初始化图，而Model ID=11才为真正的迭代计算图，因此要选择Model ID=11的数据进行分析。另外，可以看到在Model ID=11，Iteration ID=1时，Data Aug Bound的时间很长，因为该阶段存在编译等操作，所以耗时较长，因此需要选择Model ID=11且Iteration ID\>2的数据进行分析。

- Summary文件：op_statistic_\*.csv与op_summary_\*.csv

    “op_statistic_\*.csv”文件记录了AI Core和AI CPU算子调用次数及耗时统计，“op_summary_\*.csv”文件记录了详细的AI Core和AI CPU算子数据。

    开发者可根据“op_statistic_\*.csv”文件初步判断耗时较长的算子，然后再根据耗时长的算子，查找“op_summary_\*.csv”文件中的详细信息，从而定位出最小粒度事件。
