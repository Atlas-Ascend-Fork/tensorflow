# How Do I Manually Port the tf.is_finite API?

## Porting Cause

CANN does not support the  **tf.is_finite**  API. You need to manually port it.

## Porting Example

The original script checks whether gradient overflow/underflow exists. If it exists, the gradient update operation does not take effect. If it does not exist, perform  **tf.clip_by_global_norm**  on the gradient and then update the gradient.

```python
  else:
      grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
      grads, tvars = list(zip(*grads_and_vars))
      all_are_finite = tf.reduce_all(
          [tf.reduce_all(tf.is_finite(g)) for g in grads]) if use_fp16 or manual_fp16 else tf.constant(True, dtype=tf.bool)

      # This is how the model was pre-trained.
      # ensure global norm is a finite number
      # to prevent clip_by_global_norm from having a hissy fit.
      (clipped_grads, _) = tf.clip_by_global_norm(
          grads, clip_norm=1.0,
          use_norm=tf.cond(
              all_are_finite,
              lambda: tf.global_norm(grads),
              lambda: tf.constant(1.0)))

      train_op = optimizer.apply_gradients(
          list(zip(clipped_grads, tvars)), global_step=global_step)

      new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
      new_global_step = tf.identity(new_global_step, name='step_update')
      train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op
```

During script porting, the logic for determining whether the gradient overflows/underflows is implemented by  **NPULossScaleOptimizer**. The user script does not need to independently determine whether the gradient overflows/underflows.

```python
else:
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    grads, tvars = list(zip(*grads_and_vars))
   # NPULossScaleOptimizer handles the finite-gradient check.

    # This is how the model was pre-trained.
    # ensure global norm is a finite number
    # to prevent clip_by_global_norm from having a hissy fit.
    (clipped_grads, _) = tf.clip_by_global_norm(
        grads, clip_norm=1.0,
        use_norm=tf.global_norm(grads))

    train_op = optimizer.apply_gradients(
        list(zip(clipped_grads, tvars)), global_step=global_step)  # The optimizer needs to be nested before being called.

    # new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
   # new_global_step = tf.identity(new_global_step, name='step_update')
   # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
return train_op
```
