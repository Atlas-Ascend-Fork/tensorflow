# Training in Data Parallel Mode (PS-Worker)

On recommendation networks, the volume of feature data can be up to a level of TB (1 TB = 10<sup>12</sup>  bytes), which is too large for the device. Therefore, data needs to be stored in the memory of the host in PS-Worker mode. This section describes how to use the TensorFlow-based training script to perform distributed training on NPUs through the PS-Worker architecture.

## Configuring the Cluster

> [!CAUTION]NOTICE
>
> - Currently, distributed training on NPUs based on the PS-Worker architecture supports only the  **NPUEstimator**  mode.
> - Process of a worker can only be executed on one device.
> - In the PS-Worker architecture scenario, you are advised to use high-speed NICs.

In the PS-Worker architecture, cluster is configured using the TensorFlow environment variable  **TF_CONFIG**.  **TF_CONFIG**  consists of  **cluster**  and  **task**.  **cluster**  provides information about the entire cluster, namely the workers and parameter servers in the cluster.  **task**  provides information about the current task. For details, visit the  [TensorFlow official website](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_estimator).

The following uses the two-server \(each server has one parameter server and 8 workers\) scenario as an example:

1. Set  **TF_CONFIG**.

    ```python
    os.environ['TF_CONFIG'] = json.dumps({
           'cluster': {
                #'chief':chief_hosts, # Optional
                'worker': worker_hosts,
                'ps': ps_hosts,
                'evaluator':evaluator_hosts, # Not required if evaluation is not performed
            },
            'task': {'type': job_name, 'index': task_index}
    })
    ```

2. Configure  **ps_hosts**  and  **worker_hosts**  using  **FLAGS**  as follows:

    ```python
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    evaluator_hosts = FLAGS.evaluator_hosts.split(',')
    task_index = FLAGS.task_index
    job_name = FLAGS.job_name
    flags.DEFINE_string("ps_hosts", '192.168.1.100:2222,192.168.1.200:2222',)
    flags.DEFINE_string("worker_hosts",
                        '192.168.1.100:2223,192.168.1.100:2224,192.168.1.100:2225,192.168.1.100:2226,'
                        '192.168.1.100:2227,192.168.1.100:2228,192.168.1.100:2229,192.168.1.100:2230,'
                        '192.168.1.200:2223,192.168.1.200:2224,192.168.1.200:2225,192.168.1.200:2226,'
                        '192.168.1.200:2227,192.168.1.200:2228,192.168.1.200:2229,192.168.1.200:2230',)
    flags.DEFINE_string("evaluator_hosts", '192.168.1.100:2231',)
    flags.DEFINE_string("job_name", '', "One of 'ps', 'worker', 'evaluator', chief")
    flags.DEFINE_integer("task_index", 0, "Index of task within the job")
    ```

    Configuration description:

    - **worker_hosts**  and  **ps_hosts**: Separate the items by commas \(,\) without spaces.
    - **chief_hosts**: Only one argument can be set, which can also be left empty as in the preceding example. If  **chief**  is not specified, the first worker is used as the chief by default. The chief worker performs model training as other workers, and also manages other work, for example, checkpoint saving and restoration as well as summary writing.
    - **evaluator_hosts**: Only one argument can be set, which is not required if evaluation is not performed.

        Next, you need to configure  **TF_CONFIG**  for all workers.

## Defining the ParameterServerStrategy Instance

To support distributed training in the PS-Worker architecture, the  **tf.distribute.experimental.ParameterServerStrategy**  instance needs to be defined first. For details about this strategy, see  [tf.distribute.experimental.ParameterServerStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy).

```python
strategy = tf.distribute.experimental.ParameterServerStrategy()
```

## Training and Evaluating the Model

In  **NPURunConfig**, you need to specify the distribution policy for  **NPUEstimator**  by using the  **distribute**  parameter, and then call  **tf.estimator.train_and_evaluate**  to train and evaluate the model.

Ensure that  **NPURunConfig.model_dir**  of all workers is set to the same directory. For example, for a shared file system that can be read and written by all workers, if a directory is set for worker 1, this shared directory must be mounted to worker 2, and the values of  **NPURunConfig.model_dir**  must be the same.

```python
from npu_bridge.npu_init import *

run_config = NPURunConfig(
            model_dir=flags_obj.model_dir,
            session_config=session_config,
            keep_checkpoint_max=5,
            save_summary_steps=1,
            log_step_count_steps=1,
            save_checkpoints_steps=100,
            enable_data_pre_proc=True,
           mix_compile_mode=True, # PS-Worker supports only mixed precision.
           iterations_per_loop=1, # This value must be 1 in mixed precision mode.
            precision_mode='allow_mix_precision',
            distribute=strategy)

classifier = tf.estimator.NPUEstimator(
    model_fn=model_fn,
    model_dir='/tmp/multiworker',
    config=run_config)

tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn))
```

> [!CAUTION]NOTICE
>The evaluation process can be executed on the device or the host CPU. You can decide as required.
>The following uses the single-server, 8-device scenario as an example. One parameter server process and eight worker processes are needed, and the eight worker processes are executed on the device.
>
>- **To perform evaluation and training at the same time**, the number of processes started by the evaluator and workers at the same time cannot exceed the number of devices on the server \(that is, eight in the given example\). Since the eight devices are already used by the worker processes in the example, evaluation needs to be performed on the host CPU. In this case, although training and evaluation can be performed in parallel, the compute capability of  AI processor  is not utilized for evaluation. If evaluation is performed in this mode, it is recommended that checkpoint storage duration be set longer than the evaluation duration.
>    To perform evaluation on the host, call the native TensorFlow  **Estimator**.  **Estimator**  should not be converted into  **NPUEstimator**  to avoid using device resources. Otherwise, evaluation fails because the devices are already used for training.
>- **To perform evaluation after training**, ensure that the evaluator is executed after the workers complete the training. In this case, both the training and evaluation processes are executed on the devices to achieve optimal performance.

## Running the Script

To run the script using the  **ps_hosts**  and  **worker_hosts**  information in the Python script \(**chief**  is not defined in the Python script\):

```bash
python resnet50_ps_strategy.py --job_name=ps --task_index=0
python resnet50_ps_strategy.py --job_name=ps --task_index=1
python resnet50_ps_strategy.py --job_name=worker --task_index=0
python resnet50_ps_strategy.py --job_name=worker --task_index=1
python resnet50_ps_strategy.py --job_name=worker --task_index=2
python resnet50_ps_strategy.py --job_name=worker --task_index=3
python resnet50_ps_strategy.py --job_name=worker --task_index=4
python resnet50_ps_strategy.py --job_name=worker --task_index=5
python resnet50_ps_strategy.py --job_name=worker --task_index=6
python resnet50_ps_strategy.py --job_name=worker --task_index=7
```

To redefine  **ps_hosts**  and  **worker_hosts**  \(**chief**  is not defined in the Python script\):

```bash
python resnet50_ps_strategy.py \
       --ps_hosts=192.168.1.79:2222,192.168.1.80:2222 \
       --worker_hosts=192.168.1.79:2223,192.168.1.79:2224,192.168.1.79:2225,192.168.1.79:2226,192.168.1.79:2227,192.168.1.79:2228,192.168.1.79:2229,192.168.1.79:2230,192.168.1.80:2223,192.168.1.80:2224,192.168.1.80:2225,192.168.1.80:2226,192.168.1.80:2227,192.168.1.80:2228,192.168.1.80:2229,192.168.1.80:2230 \
       --job_name=ps \
       --task_index=0
```

To run  **chief**  and  **evaluator**, modify  **job_name**  to the defined type value as follows:

```bash
python resnet50_ps_strategy.py --job_name=chief --task_index=0
python resnet50_ps_strategy.py --job_name=evaluator --task_index=0
```

> [!NOTE]NOTE
> For details about the dependent environment variables, see  [Training with a Single Device](../../model_training/single_device_training.md).
