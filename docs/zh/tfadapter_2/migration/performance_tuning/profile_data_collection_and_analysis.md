# Profiling数据采集与分析

## 功能介绍

若基本调优操作不能达到性能要求，您可以借助Profiling工具采集训练过程中的性能数据并进行分析，从而准确定位系统的软、硬件性能瓶颈，提高性能分析的效率，通过针对性的性能优化方法，以最小的代价和成本提升业务性能。

默认训练过程中不采集Profiling性能数据，如需采集、解析性能数据，请参考本节内容。

TensorFlow网络性能数据采集与分析的整体流程如下所示：

![](../figures/profiling_process.png)

1. 采集Profiling数据。

    TensorFlow网络的性能数据支持全局采集与局部采集两种方式：

    - 全局采集：采集图执行所有行为的性能数据，数据量比较庞大。
    - 局部采集：支持指定采集局部子图或者指定steps的性能数据。

    全局采集有两种开启方式，可以通过修改训练脚本、配置“enable_profiling”参数的方式开启（可参见[修改训练脚本的方式](#修改训练脚本的方式)），也可以通过设置环境变量“ PROFILING_MODE”的方式开启（可参见[环境变量方式](#环境变量方式)），其中配置参数“enable_profiling”的优先级高于环境变量“ PROFILING_MODE”。

    局部采集，指通过with语句调用TF Adapter 1.x中的Profiler类，并将需要开启性能数据采集的操作放入Profiler类作用域内的方式，详细操作方式可参见[局部采集Profiling数据](#局部采集profiling数据)。

2. 解析并导出Profiling数据。

    不管是何种方式采集的Profiling数据，您都可以通过msprof命令行方式进行数据解析，并将解析结果导出到指定目录，详细操作可参见[解析并导出Profiling数据](#解析并导出profiling数据)。

3. 分析Profiling数据。

    您可以通过解析导出得到的结果文件（例如Timeline、Summary文件）进行分析，识别性能瓶颈点，典型分析示例可参见[分析Profiling数据](#分析profiling数据)。

## 全局采集Profiling数据

### 修改训练脚本的方式

修改训练脚本，在初始化NPU设备前通过添加“profiling_config”相关参数指定调优模式：

```python
import npu_device as npu
npu.global_options().profiling_config.enable_profiling=True
npu.global_options().profiling_config.profiling_options = '{"output":"/tmp/profiling","task_trace":"on","training_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
npu.open().as_default()
```

其中：

- enable_profiling：是否开启profiling数据采集功能。
- profiling_options：profiling配置选项。
  - output：profiling数据存放路径，该参数指定的目录需要在启动训练的环境上（容器或Host侧）提前创建且确保安装时配置的运行用户具有读写权限，支持配置绝对路径或相对路径
  - task_trace：是否采集任务轨迹数据
  - training_trace：是否采集迭代轨迹数据，training_trace配置为“on”的场景下需要同时配置fp_point和bp_point。
  - aicpu：是否采集AI CPU算子的详细信息，如算子执行时间、数据拷贝时间等。
  - fp_point：指定训练网络迭代轨迹正向算子的开始位置，用于记录前向计算开始时间戳，可直接配置为空，由系统自动获取。
  - bp_point：指定训练网络迭代轨迹反向算子的结束位置，记录后向计算结束时间戳，可直接配置为空，由系统自动获取。
  - aic_metrics：AI Core和AI Vector Core的硬件信息，取值“PipeUtilization”代表记录计算单元和搬运单元的耗时占比。

- Profiling配置的详细介绍请参考[Profiling](../../apiref/npu-global_options/Profiling.md)。

### 环境变量方式

除了通过修改训练脚本的方式采集Profiling数据外，用户还可以修改启动脚本中的环境变量，开启Profiling采集功能。

配置示例如下：

```bash
# 开启Profiling功能
export PROFILING_MODE=true
# 配置Profiling配置选项
export PROFILING_OPTIONS='{"output":"/home/HwHiAiUser/output","training_trace":"on","task_trace":"on","aicpu":"on","fp_point":"","bp_point":"","aic_metrics":"PipeUtilization"}'
```

环境变量PROFILING_OPTIONS的详细配置说明请参考《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》。

**需要注意，环境变量“PROFILING_MODE”的优先级低于训练脚本中的配置项“enable_profiling”。**

## 局部采集Profiling数据

TF Adapter 2.x暂不支持局部采集Profiling数据，但开发者可通过compat.v1模块调用TF Adapter 1.x中的Profiler类，从而实现局部采集性能数据的功能，即仅Profiler类作用域下的命令才会开启性能数据采集功能。

关于Profiler类的详细介绍可参见[npu_bridge.profiler.profiler](../../../tfadapter_1/apiref/profiler/Profiler_constructor.md)。

下面介绍如何通过compat.v1模块调用TF Adapter 1.x的Profiler类实现采集局部性能数据的功能。

1. 引入Profiler类。

    ```python
    import npu_device
    from npu_device.compat.v1.npu_init import *
    npu_device.compat.enable_v1()   # 禁用TF2行为，启用TF1兼容模式
    ```

2. 通过with语句调用Profiler类，并将需要做性能数据采集的操作包含在Profiler类的作用域内。

    如下是一段简单的示例代码片段，此代码片段构造了一个包含add算子的图，并在session中执行此图。其中sess.run\(add, ...\)函数在Profiler的作用域内，所以会执行L1级别的性能数据采集，并统计各种计算类指标占比，性能采集数据存储在当前脚本执行路径下。

    ```python
    a = tf.placeholder(tf.int32, (None,None))
    b = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    add = tf.add(a, b)

    with tf.compat.v1.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
        result=sess.run(add, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    当前，开发者可以采集指定steps的性能数据，只需要将指定的steps操作定义在相应的Profiler作用域内即可，如下所示：

    ```python
    a=tf.placeholder(tf.int32, (None,None))
    b=tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    c = tf.placeholder(tf.int32, (None,None))
    d = tf.constant([[1,2],[2,3]], dtype=tf.int32, shape=(2,2))
    add1 = tf.add(a, b)
    add2 = tf.add(c, d)
    add3 = tf.add(add1, add2)

    with tf.compat.v1.Session(config=session_config, graph=g) as sess:
      with profiler.Profiler(level="L1", aic_metrics="PipeUtilization", output_path = "/home/test/profiling_data"):
        for i in range(2):
          result=sess.run(add1, feed_dict={a: [[-20],[1]]})
      with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "/home/test/profiling_data"):
        for i in range(4):
          result=sess.run(add3, feed_dict={a: [[-20, 2],[1,3]],c: [[1],[-21]]})
    ```

    使用Profiler类时需要注意以下约束：

    - Profiler类需要通过with语句调用，性能数据采集功能会在对应的作用域内生效。
    - Profiler类仅支持session模式调用。
    - Profiler类不能嵌套使用。

        如下所示，是错误的调用方法。

        ```python
        with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
          with profiler.Profiler(level="L1", aic_metrics="ArithmeticUtilization", output_path = "./"):
            sess.run(add)
        ```

    - Profiler类不能与[“global_options \> Profiling”](../../apiref/npu-global_options/Profiling.md)中的配置项“enable_profiling”、“profiling_config”以及环境变量“PROFILING_MODE”、“PROFILING_OPTIONS”同时使用。
    - Profiler类不支持多线程调用。

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

## 分析Profiling数据

开发人员可通过分析Profiling工具解析得到Timeline和Summary文件，识别性能瓶颈点。

下面仅介绍常用的关键性能数据文件，关于更多Profiling数据文件的介绍可参见《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》。

- Timeline文件：step_trace_\*.csv文件

    “step_trace_\*.csv”记录了迭代轨迹数据信息，包含每轮迭代的耗时，主要字段及含义如下：

  - Iteration Time：一轮迭代的计算时间，主要包含FP/BP和Grad Refresh两个阶段的时间。
  - FP to BP Time：网络正向传播和反向传播的计算时间。
  - Iteration Refresh：迭代拖尾时间。
  - Data Aug Bound：两个相邻Iteration Time的间隔时间。

    展示数据如下图所示，开发者进行数据分析时，需要注意选择合适的Iteration ID和Model ID的数据。

    ![step_trace_-csv文件示例](../figures/step_trace_csv_file.png)

    从上面示例可以看出，Model ID=1的数据明显与后续不同，为初始化图，而Model ID=11才为真正的迭代计算图，因此要选择Model ID=11的数据进行分析。另外，可以看到在Model ID=11，Iteration ID=1时，Data Aug Bound的时间很长，因为该阶段存在编译等操作，所以耗时较长，因此需要选择Model ID=11且Iteration ID\>2的数据进行分析。

- Summary文件：op_statistic_\*.csv与op_summary_\*.csv

    “op_statistic_\*.csv”文件记录了AI Core和AI CPU算子调用次数及耗时统计，“op_summary_\*.csv”文件记录了详细的AI Core和AI CPU算子数据。

    开发者可根据“op_statistic_\*.csv”文件初步判断耗时较长的算子，然后再根据耗时长的算子，查找“op_summary_\*.csv”文件中的详细信息，从而定位出最小粒度事件。
