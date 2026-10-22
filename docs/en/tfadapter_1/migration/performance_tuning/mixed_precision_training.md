# Training with Mixed Precision

## Overview

Mixed precision is a common way to improve performance in the industry. It increases the data computing parallelism by reducing some computing precisions. Mixed precision is the combined use of the float16 and float32 data types in training deep neural networks, which reduces memory usages and accesses. Training with mixed precision presents itself as a better choice for training large networks without compromising the network accuracy produced by float32.

You can enable the mixed precision by configuring  **precision_mode_v2**  \(recommended\) or  **precision_mode**  in the script.

If automatic mixed precision is enabled, you are advised to use  [Loss Scaling](#loss-scaling)  to compensate for the accuracy loss caused by precision reduction. To analyze profile data, you need to manually modify the precision mode of some operators. You can refer to  [Modifying the Blocklist, Trustlist, and Graylist for Mixed Precision](#modifying-the-blocklist-trustlist-and-graylist-for-mixed-precision)  to specify operators to reduce or preserve the precision.

## Setting the Mixed Precision Mode

This section uses setting  **precision_mode_v2**  to  **mixed_float16**  as an example to describe how to set the mixed precision mode.

### In Estimator Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **session_config**  configuration to the  **init_resource**  function and add the  **precision_mode_v2**  parameter to  **session_config**.

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

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **run_config**  of the  **npu_run_config_init**  function.

        - If it does not exist, go to the next step.

    2. Search for  **npu_run_config_init**  in the ported script and find the run configuration function, such as  **run_config**  in the example.

        If the  **session_config**  parameter does not exist in the run configuration function, add the parameter according to the following example. If the  **session_config**  parameter exists, go to the next step.

        ```python
        session_config = tf.ConfigProto(allow_soft_placement=True)
        
        run_config = tf.estimator.RunConfig(
          train_distribute=distribution_strategy,
          session_config=session_config,
          save_checkpoints_secs=60*60*24)
        
        classifier = tf.estimator.Estimator(
          model_fn=model_function, model_dir=flags_obj.model_dir, config=npu_run_config_init(run_config=run_config))
        ```

    3. Modify the  **session_config**  configuration and add  **precision_mode_v2**.

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

- Manual porting

    In  **Estimator**  mode, set the precision mode by using the  **precision_mode_v2**  parameter in  **NPURunConfig**.

    ```python
    from npu_bridge.npu_init import *
    
    npu_config=NPURunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False),
      precision_mode_v2="mixed_float16"
      )
    ```

## In sess.run Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **session_config**  configuration to the  **init_resource**  function and add the  **precision_mode_v2**  parameter to  **session_config**.

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

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **session_config**  of the  **init_resource**  function. For other functions, configure them in  **config_proto**  of the  **npu_config_proto**  function.

        - If it does not exist, go to the next step.

    2. Search for  **npu_config_proto**  in the ported script, find the run configuration parameter \(such as  **session_config**  in the following example\), and add  **precision_mode_v2**  to the run configuration parameter, as shown in the following:

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

- Manual porting

    In  **sess.run\(\)**  mode, set the precision mode by using the session configuration option  **precision_mode_v2**.

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

## In Keras Mode

- Automated porting
    1. Check whether  **init_resource**  exists in the ported script.
        - If it exists, refer to the following example to pass the  **session_config**  configuration to the  **init_resource**  function and add the  **precision_mode_v2**  parameter to  **session_config**.

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

            Note that only the configuration options supported in  [initialize_system](../../apiref/npu_ops/initialize_system.md)  can be configured in  **config**  of the  **init_resource**  function. For other functions, configure them in  **config**  of the  **set_keras_session_npu_config**  function.

        - If it does not exist, go to the next step.

    2. Search for the  **set_keras_session_npu_config**  function in the script, find the run configuration, for example,  **config_proto**, and add  **precision_mode_v2**  to the run configuration, as shown in the following:

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
        
        # Preprocess data...
        # Construct a model...
        # Build the model...
        # Train the model...
        ```

- Manual porting

    The configuration method is similar to that of manual porting in  **sess.run**  mode. For details, see  [In sess.run Mode](#in-sessrun-mode).

## Modifying the Blocklist, Trustlist, and Graylist for Mixed Precision

When automatic mixed precision is enabled, the system automatically reduces the precisions of some data types on a network based on the built-in tiling policy. This improves the system performance while reducing the memory usage at low accuracy loss.

You can find the built-in tiling policy in  **/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>/aic-<soc_version\>-ops-info-<opType\>.json**  under the CANN installation directory.

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

- Scenarios where  **precision_mode_v2**  is set to  **mixed_float16**  and  **precision_mode**  is set to  **allow_mix_precision_fp16/allow_mix_precision**:
  - If the field value is  **true**, the operator is on the mixed precision trustlist and its precision will be reduced from float32 to float16.
  - If the field value is  **false**, the operator is on the mixed precision blocklist and its precision will not be reduced from float32 to float16.
  - If an operator does not have the  **precision_reduce**  option configured, the operator is on the graylist and will follow the same precision processing as the upstream operator.

- Scenarios where  **precision_mode**  is set to  **allow_mix_precision_bf16**  \(only on the  Ascend 950PR&950DT products,  Atlas A3 products,  Atlas A2 products\):
  - If the field value is  **true**, the operator is on the mixed precision trustlist and its precision will be reduced from float32 to bfloat16.
  - If the field value is  **false**, the operator is on the mixed precision blocklist and its precision will not be reduced from float32 to bfloat16.
  - If an operator does not have the  **precision_reduce**  option configured, the operator is on the graylist and will follow the same precision processing as the upstream operator.

You can specify operators to reduce or preserve the precision based on the built-in tiling policy.

The following describes two configuration methods.

### \(Recommended\) Using  **modify_mixlist**  to Specify the Blocklist, Trustlist, and Graylist for Mixed Precision

In the training script, use the  **modify_mixlist**  parameter to specify the configuration file of the blocklist, trustlist, and graylist for mixed precision. The following is a configuration example:

- Automated porting

    ```python
    custom_op.parameter_map["modify_mixlist"].s = tf.compat.as_bytes("/home/test/ops_info.json")
    ```

- Manual porting

    ```python
    from npu_bridge.npu_init import *
    # In Estimator mode
    npu_config=NPURunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      session_config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False),
      precision_mode_v2="mixed_float16",
      modify_mixlist="/home/test/ops_info.json"
      )
    # In sess.run mode
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
    
    # In Keras mode
    The modification method is similar to that in sess.run mode.
    ```

**ops_info.json**  is the configuration file of the blocklist, trustlist, and graylist for mixed precision. Multiple operators are separated by commas \(,\). An example is as follows:

```json
{
  "black-list": {                  // Blocklist
     "to-remove": [                // Move an operator from the blocklist to the graylist.
     "Xlog1py"
     ],
     "to-add": [                   // Move an operator from the trustlist or graylist to the blocklist.
     "MatMul",
     "Cast"
     ]
  },
  "white-list": {                  // Trustlist
     "to-remove": [                // Move an operator from the trustlist to the graylist.
     "Conv2D"
     ],
     "to-add": [                   // Move an operator from the blocklist or graylist to the trustlist.
     "Bias"
     ]
  }
}
```

Assume that operator A is in the trustlist by default. If you want to move it to the blocklist, follow any of the positive examples below:

1. \(Positive example\) Directly add the operator to the blocklist.

    ```json
    {
      "black-list": { 
         "to-add": ["A"]
      }
    }
    ```

    The operator will be deleted from the trustlist and added to the blocklist.

2. \(Positive example\) Delete the operator from the trustlist and add it to the blocklist.

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

    The operator will be deleted from the trustlist and added to the blocklist. You can find it in the blocklist.

3. \(Negative example\) Simply delete the operator from the trustlist. In this case, the operator will be moved to the graylist instead of the blocklist.

    ```json
    {
      "white-list": {
         "to-remove": ["A"]
      }
    }
    ```

    The operator will be deleted from the trustlist and added to the graylist.

    > [!NOTE]NOTE
    > If an operator is simply removed from the blocklist or trustlist, it will be added to the graylist.

### Modifying the Operator Information Library

> [!CAUTION]NOTICE
>Modifying the built-in operator information library may affect other networks. Proceed with caution.

1. Go to  **/opp/built-in/op_impl/ai_core/tbe/config/<soc_version\>**  under the CANN installation directory.
2. Grant the write permission on the  **aic-_<soc_version\>_-ops-info-<opType\>.json**  file.

    ```bash
    chmod u+w aic-<soc_version>-ops-info-<opType>.json
    ```

    All .json files in the current directory will be loaded to the operator information library. If you need to back up the original .json files, back them up to another directory.

3. Modify or add the  **precision_reduce**  field of the corresponding operator in the  **aic-_<soc_version\>_-ops-info-<opType\>.json**  file in the operator information library.

## Loss Scaling

In mixed precision computing, when the float16 data type is used, the dynamic range of data is narrowed, leading to floating-point overflow/underflow in gradient calculations and causing partial parameter updates to fail. Loss scaling can prevent the divergence during mixed precision training.

Loss scaling is a method that amplifies gradients during backward propagation by multiplying the loss obtained from forward computation by a loss scale factor  **S**. This effectively prevents underflow caused by small gradient values being unrepresentable in float16 during floating-point computation. After the parameter gradient aggregation and before the optimizer updates parameters, the aggregated parameter gradient is multiplied by 1/**S**.

Dynamic loss scaling checks the gradient floating-point exceptions during training and selects the loss scale  **S**  adaptively with the gradient change in the training process.

**In specific implementation:**

For the  Ascend 950PR&950DT products, Atlas A3 products and Atlas A2 products, the overflow/underflow mode of floating-point computation can be saturation or Inf/NaN. Retain the default Inf/NaN mode. The saturation mode is used only for compatibility with earlier versions and will not evolve in the future. In addition, the computing accuracy in this mode may be unreliable.

For  Atlas training products, the default overflow/underflow mode of floating-point computation is saturation mode, and only the saturation mode is supported. This means when an overflow occurs during computation, the computation result is saturated to a floating-point extreme value \(**+-MAX**\).

- In saturation mode, operations such as floating-point exception check of the  AI processor  are different from those of the GPU due to various floating-point computation features. In this scenario, you need to enable loss scaling or port scripts based on the original loss scaling by referring to this section.
- In Inf/NaN mode, directly use the native loss scaling of TensorFlow, without porting the function. If you have ported loss scaling by referring to this section, your network scripts can still run properly.

### Principles

- Dynamic loss scaling works as follows:
    1. Maintain a primary copy of weights in float32.
    2. Initialize the loss scaling factor  **S**  to a large value.
    3. For each iteration:

        1. Cast the primary copy of weights from float32 to float16.
        2. Perform forward propagation to obtain the loss.
        3. Multiply the resulting loss with  **S**.
        4. Perform backpropagation to obtain the gradients.
        5. Perform gradient aggregation in distributed training.
        6. If Inf or NaN is detected in the gradients, reduce  **S**, skip the parameter update, and proceed to the next iteration.
        7. Multiply the weight gradient with 1/S.
        8. Update weights using the optimizer.
        9. If no Inf or NaN is found in the last  _N_  iterations, increase  **S**.  _N_  is configurable.

        **Figure  1**  Compute procedure with dynamic loss scale  
        ![](../figures/Loss-Scale_compute_process.png)

### Using Loss Scale

- Automated porting

    If loss scaling is enabled on the original network, in automated porting scenarios, the tool automatically ports  **LossScaleManager**  of TensorFlow to  **ExponentialUpdateLossScaleManager**  or  **FixedLossScaleManager**  of NPUs. If loss scaling is not used on the original network, you can add it as required by referring to this section.

- Manual porting

    If loss scaling is enabled on the original network, you need to port  **LossScaleOptimizer**  to the  **NPULossScaleOptimizer**  or  **NPUOptimizer**  constructor. The following uses  **NPULossScaleOptimizer**  as an example.

  - Static loss scaling: You can use a fixed loss scale factor during mixed precision training.

    When enabling static loss scaling, instantiate a  **FixedLossScaleManager**  class before creating  **NPULossScaleOptimizer**  to specify the loss scaling parameters.

  - Dynamic loss scaling: You can adjust the loss scale based on the abnormal status of floating-point computation during mixed precision training.

    When using the dynamic loss scaling, instantiate a  **ExponentialUpdateLossScaleManager**  class before creating  **NPULossScaleOptimizer**  to dynamically specify the loss scale.

    > [!CAUTION]NOTICE
    >The objects of the  **ExponentialUpdateLossScaleManager**  class cannot be constructed within the influence range of the  **tf.control_dependencies\(\)**  interface. Otherwise, the graph structure execution sequence may be different from the expected sequence. For details, see  [What Do I Do If an NPULossScaleOptimizer Error Occurs?](../faq/npu_loss_scale_optimizer_faq.md).

    In distributed training, set  **is_distributed**  in  **NPULossScaleOptimizer**  to  **True**  to include loss scaling support in distributed training. In single-device training, retain the default value  **False**  for  **is_distributed**  in  **NPULossScaleOptimizer**. Failure to do so may invite training exceptions.

    Original TensorFlow code:

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

    Code after porting:

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
      # Check whether the number of devices is greater than 1. If yes, perform distributed training.
      if ops_adapter.size() > 1:
        opt_tmp = npu_distributed_optimizer_wrapper(opt_tmp)
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager, is_distributed=True)
      else:
        opt = NPULossScaleOptimizer(opt_tmp, loss_scale_manager)
    ```

    In addition, if loss scaling is not enabled in the original code, add the following lines, which use static loss scaling as an example:

    ```python
    loss_scale_manager = FixedLossScaleManager(loss_scale=1024)
    optimizer=NPULossScaleOptimizer(optimizer,loss_scale_manager)
    optimizer=optimizer.minimize(self.loss)
    ```

> [!CAUTION]NOTICE
> You may need to modify  **LossScaleManager**  parameters, as the NPU differs from the GPU in mixed precision computing. Modify the loss scaling parameters, if accuracy loss occurs as underflow is detected on too many iterations proceeding with default loss scaling parameters. This helps reduce floating-point exceptions.
>Modification method: Print the loss scale value by following  [Printing the Loss Scale Value](#printing-the-loss-scale-value), check the number of times overflow \(or underflow\) occurs based on the said value, and then adjust  **LossScaleManager**  parameters.

### Updating the Global Step

After the Loss Scale function is enabled, the step where the loss scaling overflow/underflow occurs needs to be discarded. For details, see the update step logic of the optimizer.

- In most cases,  **tf.train.MomentumOptimizer**  used in networks such as ResNet-50HC updates the global step in  **apply_gradients**. This ensures the step is not updated when overflow/underflow occurs, so no script modifications are required.
- However, for some networks \(such as BERT\), the global step update, including the judgment logic, is implemented in  **create_optimizer**. In this case, the global step update needs to be moved to the optimizer. The following is a porting example:

In the original TensorFlow code, the global step is updated in  **create_optimizer**, including the judgment logic.

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

During the porting to the Ascend platform, you need to update the global step in the optimizer as follows:

1. Comment out the global step update logic implemented in  **create_optimizer**  in the script.

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

2. Before the last return statement of the  **apply_gradients**  function, add the logic for updating the global step in the  **AdamWeightDecayOptimizer**  and  **LAMBOptimizer**  classes, respectively. The  **apply_gradients**  function is called only when no loss scaling overflow/underflow is detected in status check.

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

## Printing the Loss Scale Value

In  **Estimator**  mode, the loss scale value can be printed by adding a hook.

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

Note that the preceding hook does not apply to all networks because the loss scale value is printed by operator name. If the names of some operators in the network are specified by using  **scope**  or the like, the hook needs to be changed to the name of the desired operator.

In  **sess.run**  mode, you can call the  **get_loss_scale**  API to obtain the loss scale value from the loss scaling optimizer of the NPU.

```python
# Original code
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

# Tweaked code
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
