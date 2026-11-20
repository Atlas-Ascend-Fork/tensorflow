# About TF Adapter

TF Adapter is a TensorFlow plugin that enables TensorFlow graphs to run on the NPU. Its main role is to convert TensorFlow graphs into a format executable on the NPU.

The following figure shows the position of TF Adapter in the Ascend AI software stack.

**Figure  1**  Ascend AI software stack architecture
![](./migration/figures/ascend_architecture.png "ascend-ai-software-stack-architecture")

The following figure  shows the TF Adapter architecture.

**Figure  2**  TF Adapter architecture
![](./migration/figures/tfadapter_architecture.png "tf-adapter-architecture")

In the preceding figure, the left section illustrates the TensorFlow 1.15 architecture, while the right section depicts the TF Adapter architecture. Each layer of the TensorFlow framework is matched by a corresponding implementation within TF Adapter.

- Python API

    TF Adapter provides Python APIs that adapt to the TensorFlow framework and supports the following functions:

  - Session policies, including configuration items for function debugging, precision optimization, and performance tuning.
  - Advanced NPUEstimator APIs to facilitate model training on NPUs.
  - APIs for resource initialization and distributed training.

- Graph optimizer

    The graph optimizer receives subgraphs delivered by TensorFlow, identifies operators that can be offloaded to the device for execution, and offloads the subgraphs containing these operators to the device for execution.

- GEOP

    GEOP is a TensorFlow operator extended by TF Adapter. It is used to offload the identified subgraphs to the device for execution.

- GE Model

    GE executable graph adapted by the TF Adapter.
