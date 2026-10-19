# NPUEstimator Constructor

## Description

Constructor of the  **NPUEstimator**  class. The  **NPUEstimator**  class inherits the  **Estimator**  class of  TensorFlow  and can call the native APIs of the base class to train and evaluate TensorFlow models.

## Prototype

```python
class NPUEstimator(estimator_lib.Estimator):
    def __init__(self,
                 model_fn=None,
                 model_dir=None,
                 config=None,
                 params=None,
                 job_start_file='',
                 warm_start_from=None
                 )
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| model_fn | Input | Model function definition. This function returns an object of the NPUEstimatorSpec class.<br>For details about the constructor of the NPUEstimatorSpec class, see [NPUEstimatorSpec Constructor](NPUEstimatorSpec_constructor.md). |
| model_dir | Input | Model directory, which is used to save or restore model files. Defaults to None.<br>If model_dir set in NPURunConfig is different from that in NPUEstimator, an error is reported.<br>If either NPURunConfig or NPUEstimator is configured with model_dir, the configured path applies.<br>If neither NPURunConfig nor NPUEstimator is configured with model_dir, a model_dir_xxxxxxxxxx directory is created in the current script execution path to save the model file. |
| config | Input | Object of the NPURunConfig class.<br>For details about the constructor of the NPURunConfig class, see [NPURunConfig Constructor](../npu_config/npurunconfig_constructor/README.md). |
| params | Input | Argument of model_fn, which is of the dictionary type. The key is the name of the argument, and the value is the basic Python type value. |
| job_start_file | Input | Startup file path of the CSA job. |
| warm_start_from | Input | Path of the checkpoint. The checkpoint will be imported for training. |

## Returns

An object of the  **NPUEstimator**  class

## Example

```python
from npu_bridge.npu_init import *
...
self._classifier=NPUEstimator(
  model_fn=cnn_model_fn,
  model_dir=self._model_dir,
  config=NPURunConfig(
      save_checkpoints_steps=50 if get_rank_id() == 0 else 0,
      keep_checkpoint_max=1))
```
