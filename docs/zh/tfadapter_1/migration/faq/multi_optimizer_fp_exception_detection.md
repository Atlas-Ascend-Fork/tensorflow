# 多优化器共用导致浮点异常检测存在问题

## 问题现象

使用AI处理器进行混合精度计算时，在溢出情况下使用了饱和的计算方式，输出结果在溢出时不会有Inf的数值，而是以饱和到float16最大值的方式进行处理。饱和模式在检查溢出时，无法直接通过输出tensor中的数值进行检查，需要根据AI处理器上的状态位判断计算过程是否出现溢出。

部分GAN网络的模型训练中，会计算两个loss，并对loss结果分别调用各自的优化器进行反向计算和梯度更新。与一般网络训练场景中使用一个loss进行梯度更新的情况不同，此时如果两个部分的计算在时间上是混合进行的，因此分别检查溢出的结果可能会出现无法区分、不符合预期的情况。

## 可直接使用LossScale多优化器的场景

并不是所有使用多个优化器的情况下，LossScale的浮点异常检查都会出现无法区分的情况。如果在模型定义中，多个优化器是作为同一个迭代中的两个session分别运行，则两者的计算是可以保证时间上分开的，两者的计算任务不会交叉执行，异常状态可以在时间上区分。

我们以TensorFlow中的GAN网络为例：

[https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/train.py](https://github.com/tensorflow/gan/blob/master/tensorflow_gan/python/train.py)

```python
def sequential_train_steps(sess, train_ops, global_step, train_step_kwargs):
  """A thin wrapper around slim.learning.train_step, for GANs.
  Args:
    sess: A TensorFlow session.
    train_ops: A GANTrainOps tuple of train ops to run.
    global_step: The global step.
    train_step_kwargs: Dictionary controlling `train_step` behavior.
  Returns:
    A scalar final loss and a bool whether or not the train loop should stop.
  """
  # Only run `should_stop` at the end, if required. Make a local copy of
  # `train_step_kwargs`, if necessary, so as not to modify the caller's
  # dictionary.
  should_stop_op, train_kwargs = None, train_step_kwargs
  if 'should_stop' in train_step_kwargs:
    should_stop_op = train_step_kwargs['should_stop']
    train_kwargs = train_step_kwargs.copy()
    del train_kwargs['should_stop']
 
  # Run generator training steps.
  gen_loss = 0
  for _ in range(train_steps.generator_train_steps):
    cur_gen_loss, _ = train_step(
        sess, train_ops.generator_train_op, global_step, train_kwargs)
    gen_loss += cur_gen_loss
 
  # Run discriminator training steps.
  dis_loss = 0
  for _ in range(train_steps.discriminator_train_steps):
    cur_dis_loss, _ = train_step(
        sess, train_ops.discriminator_train_op, global_step, train_kwargs)
    dis_loss += cur_dis_loss
 
  sess.run(train_ops.global_step_inc_op)
```

其中，train_step\(\)的定义（相关部分）如下：

```python
def train_step(sess, train_op, global_step, train_step_kwargs):
  start_time = time.time()
 
  trace_run_options = None
  run_metadata = None
  if 'should_trace' in train_step_kwargs:
    if 'logdir' not in train_step_kwargs:
      raise ValueError('logdir must be present in train_step_kwargs when '
                       'should_trace is present')
    if sess.run(train_step_kwargs['should_trace']):
      trace_run_options = tf.compat.v1.RunOptions(
          trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
      run_metadata = tf.compat.v1.RunMetadata()
 
  total_loss, np_global_step = sess.run([train_op, global_step],
                                        options=trace_run_options,
                                        run_metadata=run_metadata)
  time_elapsed = time.time() - start_time
```

在上述场景中，两个优化器是作为不同的session.run\(\)执行的。这保证了不同优化器相关的计算部分在时间上是分开的，此时按照现有的NPULossScaleOptimizer修改方式可以正常工作。

## 不可直接使用LossScale多优化器的场景

在TensorFlow中，如果是使用两个优化器，并作为同一个session的fetch_dict运行，则无法直接使用NPULossScaleOptimizer。

下面的代码中是一个GAN网络中的情况，在定义优化器的部分分别对两个优化器进行计算和参数更新：

```python
d_optimizer = self.optimizer(self.d_lr)
e_optimizer = self.optimizer(self.e_lr)
 
self.e_opt = e_optimizer.minimize(
    self.e_loss, global_step=self.global_step, var_list=self.E_var)
if not self.encoder_only:
    self.d_opt = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
```

在迭代计算时，两个优化器的op被同时加到同一个session的fetch_dict进行计算：

```python
with tf.Session(config=config) as sess:
    # tf.io.write_graph(sess.graph, '/home/username/tfproject/hmr_npu/profiling', 'train.pbtxt')
    while not should_stop:
        fetch_dict = {
            "summary": self.summary_op_always,
            "step": self.global_step,
            "e_loss": self.e_loss,
            # The meat
            "e_opt": self.e_opt,
            "loss_kp": self.e_loss_kp
        }
        if not self.encoder_only:
            fetch_dict.update({
                # For D:
                "d_opt": self.d_opt,
                "d_loss": self.d_loss,
                "loss_disc": self.e_loss_disc,
            })
        if self.use_3d_label:
            fetch_dict.update({
                "loss_3d_params": self.e_loss_3d,
                "loss_3d_joints": self.e_loss_3d_joints
            })
    
        if step % self.log_img_step == 0:
            fetch_dict.update({
                "input_img": self.show_imgs,
                "gt_kp": self.show_kps,
                "e_verts": self.all_verts,
                "joints": self.all_pred_kps,
                "cam": self.all_pred_cams,
            })
            if not self.encoder_only:
                fetch_dict.update({
                    "summary_occasional":
                    self.summary_op_occ
                })
    
        t0 = time()
        result = sess.run(fetch_dict)
        t1 = time()
```

针对上述脚本中出现的使用同一session run计算多个优化器的情况，提供两种修改方案：

- 方案一：对模型中定义的多个优化器，分别使用各自的session run进行计算，即可在时间上进行分割，使用当前的Loss Scale优化器即可，此方法对于脚本修改较为直接简单，推荐使用。
- 方案二：如果一定需要使用一个session对多个优化器进行计算和更新，也可以对多个loss，多个优化器场景下，共同使用同一组溢出检查结果，来保证溢出状态的一致性。但本方案的前提是多个优化器同时放弃或进行参数更新，无法处理多优化器需要分别检查和更新的情况。

    为了保证浮点异常状态检测的正确性，可以在多优化器场景下对多个优化器共用同一次浮点异常检测的方式，保证不会出现错误检测的情况。我们仍以GAN网络为例，上述双优化器场景的Loss Scale可以使用如下方式改写：

    ```python
    from tensorflow.python.ops import gen_math_ops
    from tensorflow.python.ops import math_ops
    from npu_bridge.helper import helper
    gen_npu_ops = helper.get_gen_ops()
     
    def down_scale(grads_vars, loss_scale):
        # Down scale grads by the loss_scale.
        gv = []
        inv_loss_scale = gen_math_ops.reciprocal(loss_scale)
        for g, v in grads_vars:
            if g is not None:
                gv.append((g * math_ops.cast(inv_loss_scale, g.dtype.base_dtype), v))
            else:
                gv.append((g, v))
        return gv
     
    d_optimizer = self.optimizer(self.d_lr)
    e_optimizer = self.optimizer(self.e_lr)
     
    loss_scale_mngr = FixedLossScaleManager(loss_scale=1)
    loss_scale = loss_scale_mngr.get_loss_scale()
     
    self.float_status = gen_npu_ops.npu_alloc_float_status()
     
    e_scaled_loss = self.e_loss * math_ops.cast(loss_scale, self.e_loss.dtype.base_dtype)
    e_grads_and_vars = e_optimizer.compute_gradients(e_scaled_loss, var_list=self.E_var)
    e_grads_and_vars = down_scale(e_grads_and_vars, loss_scale)
     
    grads = []
    for (g, _) in e_grads_and_vars:
        if g is not None:
            grads.append(g)
     
    if not self.encoder_only:
        d_scaled_loss = self.d_loss * math_ops.cast(loss_scale, self.d_loss.dtype.base_dtype)
        d_grads_and_vars = d_optimizer.compute_gradients(d_scaled_loss, var_list=self.D_var)
        d_grads_and_vars = down_scale(d_grads_and_vars, loss_scale)
     
        for (g, _) in d_grads_and_vars:
            if g is not None:
                grads.append(g)
     
    with tf.get_default_graph().control_dependencies(grads):
        local_float_status = gen_npu_ops.npu_get_float_status(self.float_status)
        cleared_float_status = gen_npu_ops.npu_clear_float_status(local_float_status)
     
    is_overall_finite = math_ops.reduce_all(tf.equal(self.float_status,
                                            cleared_float_status), name="overflow_status_reduce_all")
     
    def true_apply_gradients_fn():
        def true_apply_gradients(e_grads_and_vars, d_grads_and_vars, global_step=None, name=None):
            e_opt = e_optimizer.apply_gradients(e_grads_and_vars, global_step)
            if not self.encoder_only:
                d_opt = d_optimizer.apply_gradients(d_grads_and_vars)
            return tf.group(e_opt, d_opt)
        return true_apply_gradients(e_grads_and_vars, d_grads_and_vars, self.global_step)
     
    self.update_opt_ops = control_flow_ops.cond(is_overall_finite,
                          true_apply_gradients_fn,
                          tf.no_op)
    ```

    使用新优化器返回的更新op由于并不是分开的，因此没有self.e_opt和self.d_opt两个变量，在使用session.run\(\)进行计算时，feed_dict中的self.e_opt和self.d_opt需要更改为改写代码中的self.update_opt_ops。
