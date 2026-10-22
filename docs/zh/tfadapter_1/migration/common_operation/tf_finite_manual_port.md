# tf.is_finite接口手工迁移

## 迁移原因

CANN不支持tf.is_finite接口，需要用户手工迁移。

## 迁移示例

用户原始脚本会判断梯度是否溢出，如果有溢出，梯度更新操作不生效；如果无溢出，先对梯度进行tf.clip_by_global_norm操作，然后进行梯度更新：

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

脚本迁移时，将梯度是否溢出的判断逻辑放在NPULossScaleOptimizer进行，不需要用户脚本单独判断：

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
        list(zip(clipped_grads, tvars)), global_step=global_step)  # optimizer调用前需要嵌套NPULossScaleOptimizer

    # new_global_step = tf.cond(all_are_finite, lambda: global_step + 1, lambda: global_step)
   # new_global_step = tf.identity(new_global_step, name='step_update')
   # train_op = tf.group(train_op, [global_step.assign(new_global_step)])
return train_op
```
