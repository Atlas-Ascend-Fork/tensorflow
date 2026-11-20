# About TF Adapter

The Ascend Adapter for TensorFlow 2._x_  \(or TF Adapter\) is provided to developers for porting models trained on TensorFlow 2._x_  \(or TF2\) to the  AI processor  \(or NPU\) for execution. In the Ascend AI software stack, the TF Adapter layer is located between the framework layer and the compute architecture for neural networks \(CANN\) layer. The TF Adapter is a non-intrusive Ascend release that works with TF2.

The following figure shows the position of the TF Adapter in the Ascend AI software stack.

**Figure  1**  Ascend AI software stack architecture
![](./migration/figures/ascend_architecture.png)

## Key Concepts of TF2

Key TF2 concepts related to the TF Adapter are as follows:

- **Eager execution**

    In TF2, eager execution is enabled by default. It executes the operations instantly and outputs operation results, without the need of building graphs. Click  [here](https://tensorflow.google.cn/guide/eager?hl=en)  for more details.

- **Eager Context**

    An eager context for TF2 eager execution is globally unique and contains thread variables to meet different context requirements in different threads.

- **tf.function**

    It is a decorator provided by TF2 that encapsulates the TF2 operators called in Python functions into a graph for execution, thus improving the performance. For more information, click  [here](https://tensorflow.google.cn/api_docs/python/tf/function).

- **TF2 custom device**

    The TF2 provides the C API  **TFE_RegisterCustomDevice**  for registering custom devices. The TF Adapter calls this API to register the  AI processor  as the TF2 custom device, which is equivalent to the built-in CPUs and GPUs. For TF2 source code, click  [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/c/eager/c_api.cc).

## TF Adapter Principles

The TF Adapter registers the  AI processor  as a TF2 custom device and sets it as the default device. After the registration, operators that user schedules to the  AI processor  or not scheduled will be executed on the  AI processor. The operator execution API of the  AI processor  invokes CANN's operator/graph execution capability to complete the operator execution on the  AI processor.

**Figure  2**  TF Adapter interconnection framework
![](./migration/figures/TF-Adapter_architecture.png)

The following shows the interaction between the TF Adapter and other modules.

A typical training workflow goes through the device initialization, model \(variable\) initialization, training execution, and checkpoint saving steps.

**Figure  3**  TF Adapter interconnection sequence
![](./migration/figures/TF-Adapter_sequence_diagram.png)

The concepts involved in the workflow diagram are described as follows:

- **CANN**

    Architecture of the  AI processor-based user programming interfaces. For details, click  [here](https://www.hiascend.com/software/cann).

- **TF2 Runtime**

    TensorFlow's native runtime API.

- **Iterator**

    It is recommended using an iterator to iterate over a dataset when building a TensorFlow input pipeline. This mode also shows high performance affinity on the  AI processor. For details, click  [here](https://www.tensorflow.org/guide/data).

- **Host Device Communication \(HDC\) channel**

    Data transfer channel from the TensorFlow process to the  AI processor  hardware memory. TF Adapter 2._x_  asynchronously feeds training data to the training job on the  AI processor  through the HDC channel in the TensorFlow process.

## Advantages

The TF Adapter solution has the following advantages:

- Supports NPU registration as a custom device of TF2. From the developer's perspective, the NPU exists on equal terms as the GPU/CPU, and is forward-compatible with TF2.
- Enables operator-level adaptation, compatible with the native features of TF2. You can utilize the graph processing capability of CANN to accelerate operators \(especially function operators\) execution.
- Provides plugin-based non-intrusive interconnection with CANN across platforms. TF2 does not need to be recompiled or redeployed.
