# 仅在某个Device上保存Checkpoint数据

分布式训练场景下，如果用户只希望在某个device上保存checkpoint数据，而不希望在其他device上保存checkpoint数据，可以按照如下方法修改训练脚本。

TensorFlow原始代码：

```python
self._classifier=tf.estimator.Estimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=tf.estimator.RunConfig(
      save_checkpoints_steps=50 if hvd.rank() == 0 else None,
      keep_checkpoint_max=1))
```

迁移后的代码：

```python
self._classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=NPURunConfig(
      save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
      keep_checkpoint_max=1))
```
