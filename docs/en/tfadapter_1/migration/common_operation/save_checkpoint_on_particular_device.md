# How Do I Save Checkpoints on a Particular Device?

In distributed training, to save checkpoints only on a particular device, edit your training script as follows:

Original TensorFlow code:

```python
self._classifier=tf.estimator.Estimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=tf.estimator.RunConfig(
      save_checkpoints_steps=50 if hvd.rank() == 0 else None,
      keep_checkpoint_max=1))
```

Code after porting:

```python
self._classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=NPURunConfig(
      save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
      keep_checkpoint_max=1))
```
