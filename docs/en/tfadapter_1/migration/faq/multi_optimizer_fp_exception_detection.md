# What Do I Do If Floating-Point Exception Detection Is Abnormal Because Multiple Optimizers Are Used?

## Symptom

When the  AI processor  is used for mixed-precision computing, saturation is performed in the case of overflow/underflow. Specifically, the output does not contain Inf when overflow/underflow occurs; instead, it is saturated to the extreme representable by float16. In this mode, overflow/underflow cannot be directly detected based on the output tensor. It is determined based on the status bit on the  AI processor.

For the training of certain GAN networks, two loss values are computed, and their optimizers are called separately to perform back propagation and gradient update for the loss results. This is different from other common networks in which only one loss is used for gradient update. If the compute time of the two losses is mixed, the overflow/underflow detection results may fail to be distinguished or be not as expected otherwise.

## If Multiple LossScale Optimizers Can Be Used

In some cases, floating-point exception detection of LossScale can work properly even if multiple optimizers are used. In the model definition, if multiple optimizers run as two sessions in the same iteration, the compute time of the two optimizers can be separated and the computing tasks will not be executed crosswise. Therefore, the abnormal status of the two optimizers can be distinguished from each other.

The following uses a GAN network in TensorFlow as an example:

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

The related definition of  **train_step\(\)**  is as follows:

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

In the preceding scenario, the two optimizers are executed as different  **session.run\(\)**. This ensures that the computing time of different optimizers is separated from each other. In this case, floating-point exception detection can work properly by using the existing modification method of NPULossScaleOptimizer.

## If Multiple LossScale Optimizers Cannot Be Used Directly

In TensorFlow, if two optimizers are used to run as fetch_dict of the same session, NPULossScaleOptimizer cannot be directly used.

The following uses a GAN network as an example. The two optimizers are computed and their parameters are updated separately.

```python
d_optimizer = self.optimizer(self.d_lr)
e_optimizer = self.optimizer(self.e_lr)
 
self.e_opt = e_optimizer.minimize(
    self.e_loss, global_step=self.global_step, var_list=self.E_var)
if not self.encoder_only:
    self.d_opt = d_optimizer.minimize(self.d_loss, var_list=self.D_var)
```

During iterative computing, the Op of two optimizers is added to  **fetch_dict**  of the same session.

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

Two solutions are provided for the preceding scenario:

- Solution 1: If multiple optimizers are defined in the model, use their own  **session.run\(\)**  for computing. You can use the current Loss Scale optimizer. This method is recommended because it is simple for script modification.
- Solution 2: If only one  **session.run\(\)**  is used to compute and update multiple optimizers, you can use the same group of overflow/underflow detection results for multiple losses and optimizers to ensure the consistency of the overflow status. However, this solution presupposes that multiple optimizers skip or perform parameter updates simultaneously. In this method, multiple optimizers cannot be checked and updated separately.

    To ensure the correctness of floating-point exception detection, use the same floating-point exception detection mode for multiple optimizers. The following still uses a GAN network as an example. The Loss Scale in the dual-optimizer scenario can be modified as follows:

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

    The updated Op returned by the new optimizer is not separated. Therefore, the  **self.e_opt**  and  **self.d_opt**  variables do not exist. When  **session.run\(\)**  is used for computing, change  **self.e_opt**  and  **self.d_opt**  in  **feed_dict**  to  **self.update_opt_ops**  in the new code.
