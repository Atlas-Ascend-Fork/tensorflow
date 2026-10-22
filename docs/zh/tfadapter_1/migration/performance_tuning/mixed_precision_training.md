# 混合精度训练

## 概述

混合精度为业内通用的性能提升方式，通过降低部分计算精度提升数据计算的并行度。混合精度训练方法是通过混合使用float16和float32数据类型来加速深度神经网络训练的过程，并减少内存使用和存取，从而可以训练更大的神经网络，同时又能基本保持使用float32训练所能达到的网络精度。

用户可以在脚本中通过配置“precision_mode_v2”（推荐）或者“precision_mode”参数开启混合精度。

开启混合精度的场景下，推荐使用[Loss Scale](#loss-scale)，从而补偿降低精度带来的精度损失；若后续对Profiling数据进行分析时，发现需要手工调整某些算子的精度模式，可以参考[修改混合精度黑白灰名单](#修改混合精度黑白灰名单)自行指定哪些算子允许降精度，哪些算子不允许降精度。

## 设置混合精度模式

本节以将“precision_mode_v2”参数配置为“mixed_float16”为例，说明如何设置混合精度模式。

### Estimator模式下设置精度模式

- 自动迁移场景
    1. 检查迁移后的脚本是否存在“init_resource”。
        - 如果存在，则需要参考下面示例，在init_resource函数中传入session_config的配置，并在session_config配置中添加“precision_mode_v2”参数。

            ```python
            if __name__ == '__main__':
            
              session_config = tf.ConfigProto(allow_soft_placement=True)
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = "NpuOptimizer"
              custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
            
              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在npu_run_config_init函数的run_config中进行配置。

        - 如果不存在，则执行下一步。

    2. 在迁移后的脚本中查找“npu_run_config_init”，找到运行配置函数，例如示例中的“run_config”。

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

    3. 修改session_config配置，添加“precision_mode_v2”。

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
        
        run_config = tf.estimator.RunConfig(
          train_distribute=distribution_strategy,
          session_config=session_config,
          save_checkpoints_secs=60*60*24)
        
        classifier = tf.estimator.Estimator(
          model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
        ```

- 手工迁移场景

    Estimator模式下，通过NPURunConfig中的precision_mode_v2参数设置精度模式：

    ```python
    from npu_bridge.npu_init import *
    
    npu_config=NPURunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False),
      precision_mode_v2="mixed_float16"
      )
    ```

### sess.run模式下设置精度模式

- 自动迁移场景
    1. 检查迁移后的脚本是否存在“init_resource”。
        - 如果存在，则需要参考下面示例，在init_resource函数中传入session_config配置，并在session_config中添加“precision_mode_v2”参数。

            ```python
            if __name__ == '__main__':
              session_config = tf.ConfigProto(allow_soft_placement=True)
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = 'NpuOptimizer'
              custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
            
              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的session_config中进行配置，若需配置其他功能，请在npu_config_proto函数的config_proto中进行配置。

        - 如果不存在，则执行下一步。

    2. 在迁移后的脚本中查找“npu_config_proto”，找到运行配置参数（例如下面示例中的“session_config”），在运行配置参数中添加“precision_mode_v2”，如下所示。

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
        config = npu_config_proto(config_proto=session_config)
        with tf.Session(config=config) as sess:
          sess.run(tf.global_variables_initializer())
          interaction_table.init.run()
        ```

- 手工迁移场景

    sess.run模式下，通过session配置项precision_mode_v2参数设置精度模式：

    ```python
    import tensorflow as tf
    from npu_bridge.npu_init import *
    
    config = tf.ConfigProto(allow_soft_placement=True)
    
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer" 
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    
    with tf.Session(config=config) as sess:
      print(sess.run(cost))
    ```

### Keras模式下设置精度模式

- 自动迁移场景
    1. 检查迁移后的脚本是否存在“init_resource”。
        - 如果存在，则需要参考下面示例，在init_resource函数中传入session_config配置，并在session_config中添加“precision_mode_v2”参数。

            ```python
            if __name__ == '__main__':
            
              session_config = tf.ConfigProto(allow_soft_placement=True )
              custom_op = session_config.graph_options.rewrite_options.custom_optimizers.add()
              custom_op.name = "NpuOptimizer" 
              custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
              ... ...
            
              (npu_sess, npu_shutdown) = init_resource(config=session_config)
              tf.app.run()
              shutdown_resource(npu_sess, npu_shutdown)
              close_session(npu_sess)
            ```

            需要注意，仅[initialize_system](../../apiref/npu_ops/initialize_system.md)中支持的配置项可在init_resource函数的config中进行配置，若需配置其他功能，请在“set_keras_session_npu_config”函数的config中进行配置。

        - 如果不存在，则执行下一步。

    2. 在脚本中查找“set_keras_session_npu_config”函数，找到运行配置，例如config_proto，然后在运行配置中添加“precision_mode_v2”，如下所示。

        ```python
        import tensorflow as tf
        import tensorflow.python.keras as keras
        from tensorflow.python.keras import backend as K
        from npu_bridge.npu_init import *
        
        config_proto = tf.ConfigProto(allow_soft_placement=True)
        custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
        custom_op.name = 'NpuOptimizer'
        custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
        npu_keras_sess = set_keras_session_npu_config(config=config_proto)
        
        #数据预处理...
        #模型搭建...
        #模型编译...
        #模型训练...
        ```

- 手工迁移场景

    与sess.run的手工迁移场景配置方式类似，请参见[sess.run模式下设置精度模式](#sessrun模式下设置精度模式)。

## 修改混合精度黑白灰名单

开启自动混合精度的场景下，系统会自动根据内置的优化策略，对网络中的某些数据类型进行降精度处理，从而在精度损失很小的情况下提升系统性能并减少内存使用。

内置优化策略在“CANN软件安装目录/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json“，例如：

```json
"Conv2D":{
    "precision_reduce":{
        "flag":"true"
    },
    {
    ... ...
    }
}
```

- precision_mode_v2配置为mixed_float16，precision_mode配置为allow_mix_precision_fp16/allow_mix_precision的场景：
  - 若取值为true（白名单），则表示允许将当前float32类型的算子，降低精度到float16。
  - 若取值为false（黑名单），则不允许将当前float32类型的算子降低精度到float16，相应算子仍使用float32精度。
  - 若网络模型中算子没有配置该参数（灰名单），当前算子的混合精度处理机制和前一个算子保持一致，即如果前一个算子支持降精度处理，当前算子也支持降精度；如果前一个算子不允许降精度，当前算子也不支持降精度。

- precision_mode配置为allow_mix_precision_bf16的场景（仅Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，支持此配置）：
  - 若取值为true（白名单），则表示允许将当前float32类型的算子，降低精度到bfloat16。
  - 若取值为false（黑名单），则不允许将当前float32类型的算子降低精度到bfloat16，相应算子仍旧使用float32精度。
  - 若网络模型中算子没有配置该参数（灰名单），当前算子的混合精度处理机制和前一个算子保持一致，即如果前一个算子支持降精度处理，当前算子也支持降精度；如果前一个算子不允许降精度，当前算子也不支持降精度。

用户可以在内置优化策略基础上进行调整，自行指定哪些算子允许降精度，哪些算子不允许降精度。

下面介绍两种配置方法。

### （推荐）通过modify_mixlist参数指定混合精度黑白灰算子名单

在训练脚本中通过modify_mixlist参数指定混合精度黑白灰算子名单配置文件，配置示例如下：

- 自动迁移场景

    ```python
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
    ```

- 手工迁移场景

    ```python
    from npu_bridge.npu_init import *
    # Estimator模式修改方法
    npu_config=NPURunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False),
      precision_mode_v2="mixed_float16",
      modify_mixlist="/home/test/ops_info.json"
      )
    # sess.run模式修改方法
    config = tf.ConfigProto()
    custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name =  "NpuOptimizer" 
    custom_op.parameter_map["use_off_line"].b = True
    custom_op.parameter_map["precision_mode_v2"].s = tf.compat.as_bytes("mixed_float16")
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF
    with tf.Session(config=config) as sess:
      print(sess.run(cost))
    
    # Keras模式修改方法
    与sess.run模式修改方法类似
    ```

其中ops_info.json为混合精度黑白灰算子名单配置文件，多个算子使用英文逗号分隔，样例如下：

```json
{
  "black-list": {                  // 黑名单
     "to-remove": [                // 黑名单算子转换为灰名单算子
     "Xlog1py"
     ],
     "to-add": [                   // 白名单或灰名单算子转换为黑名单算子
     "MatMul",
     "Cast"
     ]
  },
  "white-list": {                  // 白名单
     "to-remove": [                // 白名单算子转换为灰名单算子 
     "Conv2D"
     ],
     "to-add": [                   // 黑名单或灰名单算子转换为白名单算子
     "Bias"
     ]
  }
}
```

假设算子A默认在白名单中，如果您希望将该算子配置为黑名单算子，可以参考如下方法：

1. （正确示例）用户将该算子添加到黑名单中：

    ```json
    {
      "black-list": { 
         "to-add": ["A"]
      }
    }
    ```

    则系统会将该算子从白名单中删除，并添加到黑名单中。

2. （正确示例）用户将该算子从白名单中删除，同时添加到黑名单中：

    ```json
    {
      "black-list": {
         "to-add": ["A"]
      },
      "white-list": {
         "to-remove": ["A"]
      }
    }
    ```

    则系统会将该算子从白名单中删除，并添加到黑名单中，最终该算子在黑名单中。

3. （错误示例）用户将该算子从白名单中删除，此时算子最终是在灰名单中，而不是黑名单。

    ```json
    {
      "white-list": {
         "to-remove": ["A"]
      }
    }
    ```

    此时，系统会将该算子从白名单中删除，然后添加到灰名单中，最终该算子在灰名单中。

    > [!NOTE]说明
    > 对于只从黑/白名单中删除，而不添加到白/黑名单的情况，系统会将该算子添加到灰名单中。

### 修改算子信息库

> [!NOTE]说明
> 对内置算子信息库进行修改，可能会对其他网络造成影响，请谨慎修改。

1. 切换到“CANN软件安装目录/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>“目录下。
2. 对aic-<soc_version\>-ops-info-<opType\>.json文件增加写权限。

    ```bash
    chmod u+w aic-<soc_version>-ops-info-<opType>.json
    ```

    当前目录下的所有json文件都会被加载到算子信息库中，如果您需要备份原来的json文件，建议备份到其他目录下。

3. 修改或增加算子信息库aic-<soc_version\>-ops-info-<opType\>.json文件中对应算子的precision_reduce字段。

## Loss Scale

在混合精度计算中，使用float16数据格式时数据的动态范围会降低，造成梯度计算出现浮点溢出，从而导致部分参数更新失败。为了保证部分模型训练在混合精度训练过程中收敛，需要配置Loss Scale参数。

Loss Scale方法通过在前向计算所得的loss乘以Loss Scale系数S，起到在反向梯度计算过程中达到放大梯度的作用，从而有效规避浮点计算中较小梯度值无法用FP16表达而出现的溢出问题。在参数梯度聚合之后以及优化器更新参数之前，将聚合后的参数梯度值除以Loss Scale系数S还原。

动态Loss Scale通过在训练过程中检查梯度中浮点计算异常状态，自动动态选取Loss Scale系数S以适应训练过程中梯度变化，从而解决人工选取Loss Scale系数S和训练过程中自适应调整的问题。

**在具体实现中：**

针对Ascend 950PR&950DT系列产品，Atlas A3系列产品，Atlas A2系列产品，浮点计算溢出模式支持饱和模式与INF/NaN模式，请保持默认值INF/NaN模式。饱和模式仅用于兼容旧版本，后续不再演进，且此模式下计算精度可能存在误差。

针对Atlas训练系列产品，浮点计算溢出模式默认为“饱和模式”且仅支持“饱和模式”，即计算出现溢出时，计算结果会饱和为浮点数极值（+-MAX）。

- 浮点计算的溢出模式为“饱和模式”的场景下，AI处理器由于浮点计算特性不同，在计算过程中的浮点异常检查等部分与GPU存在差异，此种场景下，开发者需要参考本章节开启Loss Scale功能或者基于原有Loss Scale功能迁移脚本。
- 浮点计算的溢出模式为“INF/NaN模式”的场景下，开发者使用TensorFlow原生的Loss Scale功能即可，无需参考本节做功能迁移。当然，若您已参考本节进行了Loss Scale功能的迁移，您的网络脚本仍可正常运行。

### 实现原理

- 动态Loss Scale的主要计算流程。
    1. 维护一个float32的主参数版本。
    2. 将Loss Scale系数S初始化为一个较大值。
    3. 在每次迭代中：

        1. 从float32的主参数版本中通过精度转换cast出一份float16的参数版本供本次迭代计算使用。
        2. 前向计算获得loss。
        3. 将loss乘以当前Loss Scale系数S。
        4. 反向计算获得梯度。
        5. 分布式训练场景下进行梯度聚合操作。
        6. 检查梯度，当存在inf/nan时，减小Loss Scale系数S，不进行参数更新，并结束本次迭代。
        7. 将梯度乘以1/S还原。
        8. 通过优化器更新参数。
        9. 如果在最近N次迭代未发现inf/nan，则增加Loss Scale系数S。N为可配置项。

        **图 1**  动态Loss Scale的主要计算流程
        ![动态Loss-Scale的主要计算流程](../figures/Loss-Scale_compute_process.png)

### 使用Loss Scale

- 自动迁移场景

    如果原始网络中使用了Loss Scale功能，使用工具自动迁移的场景下，工具会自动将TensorFlow的LossScaleManager迁移为NPU的ExponentialUpdateLossScaleManager或FixedLossScaleManager。如果原始网络中没有使用Loss Scale功能，用户可以根据需要参考本节自行添加。

- 手工迁移场景

    如果原始网络中使用了Loss Scale功能，需要将LossScaleOptimizer迁移为NPULossScaleOptimizer或NPUOptimizer构造函数，下面仅以NPULossScaleOptimizer举例说明。

  - 静态Loss Scale：用户可定义在混合精度训练过程中使用固定的Loss Scale系数。

    具体做法是，在创建NPULossScaleOptimizer之前，实例化一个FixedLossScaleManager类以指定Loss Scale的值。

  - 动态Loss Scale：用户可定义在混合精度训练过程中根据浮点计算异常状态调整Loss Scale系数。

    具体做法是，在创建NPULossScaleOptimizer之前，实例化一个ExponentialUpdateLossScaleManager类进行动态Loss Scale的配置。

    > [!NOTE]说明
    > ExponentialUpdateLossScaleManager类对象的构造不能在tf.control_dependencies\(\)接口的作用域内，否则可能会造成图结构执行顺序与预期不一致，详细可参见[NPULossScaleOptimizer优化器使用常见问题](../faq/npu_loss_scale_optimizer_faq.md)。

    另外，分布式训练场景下，如果使用了NPULossScaleOptimizer，必须将is_distributed配置为True，以支持分布式训练场景下Loss Scale功能。单卡场景下，NPULossScaleOptimizer的is_distributed必须保持默认值False，否则会导致训练异常。

    TensorFlow原始代码：

    ```python
    if FLAGS.use_fp16 and (FLAGS.bert_loss_scale not in [None, -1]):
      opt_tmp = opt
      if FLAGS.bert_loss_scale == 0:
        loss_scale_manager = tf.contrib.mixed_precision.ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
      elif FLAGS.bert_loss_scale >= 1:
        loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(loss_scale=FLAGS.bert_loss_scale)
      else:
        raise ValueError("Invalid loss scale: %d" % FLAGS.bert_loss_scale)
      opt = tf.contrib.mixed_precision.LossScaleOptimizer(opt_tmp, loss_scale_manager)
    ```

    迁移后的代码：

    ```python
    from npu_bridge.npu_init import *
    
    if FLAGS.use_fp16 and (FLAGS.bert_loss_scale not in [None, -1]):
      opt_tmp = opt
      if FLAGS.bert_loss_scale == 0:
        loss_scale_manager = ExponentialUpdateLossScaleManager(init_loss_scale=2**32, incr_every_n_steps=1000, decr_every_n_nan_or_inf=2, decr_ratio=0.5)
      elif FLAGS.bert_loss_scale >= 1:
        loss_scale_manager = FixedLossScaleManager(loss_scale=FLAGS.bert_loss_scale)
      else:
        raise ValueError("Invalid loss scale: %d" % FLAGS.bert_loss_scale)
      #device数是否大于1，如果大于1，进行分布式训练
      if ops_adapter.size() > 1:
        opt_tmp = npu_distributed_optimizer_wrapper(opt_tmp)
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=True)
      else:
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
    ```

    另外，如果原始代码中没有使用Loss Scale，可以找到优化器名称后补充如下代码（以使用静态Loss Scale为例）：

    ```python
    loss_scale_manager = FixedLossScaleManager(loss_scale=1024)
    optimizer=NPULossScaleOptimizer(optimizer,loss_scale_manager)
    optimizer=optimizer.minimize(self.loss)
    ```

> [!NOTE]说明
> 由于NPU计算特性与GPU混合精度计算特性存在差异，LossScaleManager超参也往往需要进行适当的调整以保证精度。当用户模型基于默认Loss Scale参数训练产生溢出的迭代过多，影响最终精度时，需要对Loss Scale参数进行适当调整，减少发生浮点异常的次数。
>具体方法为：参考[打印Loss Scale值](#打印loss-scale值)打印Loss Scale，根据Loss Scale值观察溢出次数，调整LossScaleManager参数。

### 更新global step

在开启Loss Scale后，需要丢弃Loss Scale溢出的step，具体需要看使用的优化器的更新step逻辑：

- 大多数情况下，例如resnet50HC网络中本来用的tf.train.MomentumOptimizer优化器，它更新global step就是在apply_gradients中处理的，此时能保证溢出时不更新step，因此不需要进行脚本改造。
- 针对某些网络（例如BERT网络），更新global step是在create_optimizer里面实现的，包括判断逻辑，此时需要将更新global step放在优化器进行。具体迁移示例如下：

TensorFlow原始代码中，更新global step是在create_optimizer里面实现的，包括判断逻辑：

```python
def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, hvd=None, manual_fp16=False, use_fp16=False, num_accumulation_steps=1,
                     optimizer_type="adam", allreduce_post_accumulation=False):
  ...
      if tf.flags.FLAGS.npu_bert_clip_by_global_norm:
        new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
      else:
        new_global_step = global_step + 1
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op
```

迁移到Ascend平台时，需要将更新global step放在优化器进行：

1. 将脚本中create_optimizer里面实现的global step更新逻辑注释掉：

    ```python
    def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, hvd=None, manual_fp16=False, use_fp16=False, num_accumulation_steps=1,
                         optimizer_type="adam", allreduce_post_accumulation=False):
      ...
          #if tf.flags.FLAGS.npu_bert_clip_by_global_norm:
          #  new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
          #else:
          #  new_global_step = global_step + 1
          #new_global_step = tf.identity(new_global_step, name='step_update')
          #train_op = tf.group(train_op, [global_step.assign(new_global_step)])
      return train_op
    ```

2. 分别在AdamWeightDecayOptimizer和LAMBOptimizer类的apply_gradients函数最后return之前增加更新global step的逻辑，LossScale只有状态检查未溢出时才会调用apply_gradients：

    ```python
      def apply_gradients(self, grads_and_vars, global_step=None, name=None,
          manual_fp16=False):
        assignments = []
        for (grad, param) in grads_and_vars:
            ...
        new_global_step = global_step + 1
        new_global_step = tf.identity(new_global_step, name='step_update')
        assignments.extend([global_step.assign(new_global_step)])
        return tf.group(*assignments, name=name)
    ```

### 打印Loss Scale值

Estimator模式下，可以通过添加hook的方式实现对Loss Scale值进行打印：

```python
class _LogSessionRunHook(tf.train.SessionRunHook):
   def before_run(self, run_context):
       return tf.estimator.SessionRunArgs(
               fetches=['loss_scale:0'])
 
   def after_run(self, run_context, run_values):
       print('loss scale value=%d' % run_values.results[0], flush=True)
  
...

if 'train' in params.exec_mode:
    training_hooks = get_hooks(params, logger)
    training_hooks.append(_LogSessionRunHook())
    estimator.train(
        input_fn = dataset.train_fn,
        steps = max_steps,
        hooks = training_hooks)
```

需要注意的是，以上hook无法适用所有网络，原因是Loss Scale值是根据算子名称打印的，如果用户使用了scope等指定网络中部分算子的名称，则该hook需要相应更改为需要获取的算子名称。

sess.run模式下，可以通过调用get_loss_scale接口从NPU的Loss Scale优化器获取Loss Scale的值。

```python
# 原始代码示例
for step in range(restore_step, FLAGS.max_steps):
    data = next(data_generator)
    inputs_padded = data[0]
    bbox_padded = pad_bbox(data[1],FLAGS.num_bbox)
    input_image_np = inputs_padded
    input_bbox_np = bbox_padded

    ml, tl,ce_loss, bbox_loss, _, summary_str = sess.run([
                                       model_loss,
                                       total_loss, 
                                       rpn_cross_entropy,
                                       rpn_loss_box,
                                       train_op, summary_op],
                                       feed_dict={input_image: input_image_np,input_bbox: input_bbox_np})
    summary_writer.add_summary(summary_str, global_step=step)

# 修改后的代码示例
for step in range(restore_step, FLAGS.max_steps):
    data = next(data_generator)
    inputs_padded = data[0]
    bbox_padded = pad_bbox(data[1],FLAGS.num_bbox)
    input_image_np = inputs_padded
    input_bbox_np = bbox_padded
    lossScale = loss_scale_manager.get_loss_scale()
    l_s, global_step, ml, tl,ce_loss, bbox_loss, _, summary_str = sess.run(
                                      [lossScale,
                                       global_step,
                                       model_loss,
                                       total_loss,
                                       rpn_cross_entropy,
                                       rpn_loss_box,
                                       train_op, summary_op],
                                       feed_dict={input_image: input_image_np, input_bbox: input_bbox_np})
    summary_writer.add_summary(summary_str, global_step=step)
    print('loss_scale is: ', l_s)
    print("global_step:", global_step)
```
