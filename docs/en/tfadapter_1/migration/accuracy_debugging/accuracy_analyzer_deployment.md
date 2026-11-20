# Model Accuracy Analyzer Deployment

## Overview

To facilitate accuracy tuning in training, the  one-click accuracy analyzer  provides common functions for accuracy analysis of training networks, including:

- [Floating-Point Exception Detection](floating-point_exception_detection.md)
- [Fusion Exception Detection](fusion_exception_detection.md)
- [Network-wide Accuracy Comparison](network_accuracy_comparison.md)
- [Random Error Detection](random_error_detection.md)

This tool encapsulates the run parameters of TF Adapter and extends functions of the Model Accuracy Analyzer \(**msaccucmp.py**\) in the CANN package, facilitating fast accuracy fault location.

## Restrictions

- Overflow/Underflow data collection is mutually exclusive with accuracy data dump.
- Set a proper epoch number, which helps avert running out of disk space caused by a large number of files generated during overflow/underflow data collection or accuracy data dump.

## One-Click Accuracy Analyzer Deployment

This tool is installation-free. Download the  **precision_tool**  directory from  [https://gitee.com/ascend/tools](https://gitee.com/ascend/tools)  and upload it to the training directory.

The directory structure is as follows:

```text
├── resnet                              // Training working directory.
│    ├── __init__.py
│    ├── imagenet_main.py              // Script for training the network based on the ImageNet dataset.
│    ├── imagenet_preprocessing.py     // ImageNet preprocessing module.
 │    ├── resnet_model.py               // ResNet model file.
│    ├── resnet_run_loop.py            // Data input processing and run loop (for training, validation, and test).
│    ├── cifar10_main                  // Training entry point file.
│    ├── ...
│    ├── precision_tool           // Directory of the one-click accuracy analyzer
│    │    ├── cli.py
│    │    ├── ...
```

- If the CANN development and operating environments are set up on the same server, simply upload the  **precision_tool**  directory to the training directory.
- If the CANN development and operating environments are set up on separate servers, upload the  **precision_tool**  directory to the training directory in the CANN operating environment and any directory in the CANN development environment.

    > [!NOTE]NOTE
    >
    > - The CANN operating environment \(where NPU training is run on the  AI processor\) is mainly used for collecting accuracy data during training accuracy tuning.
    > - The CANN development environment is mainly used for accuracy analysis during training accuracy tuning.

## Typical Workflow

The following figure shows the workflow of using the one-click accuracy analyzer  **precision_tool**  to analyze accuracy when the CANN development and operating environments are deployed on the same server.

**Figure  1**  CANN development and operating environments set up on the same server
![](../figures/dev_run_merge.png "cann-development-and-operating-environments-set-up-on-the-same-server")

The following figure shows the workflow of using the one-click accuracy analyzer  **precision_tool**  to analyze accuracy when the CANN development and operating environments are deployed separately.

**Figure  2**  CANN development and operating environments set up separately
![](../figures/dev_run_part.png "cann-development-and-operating-environments-set-up-separately")
