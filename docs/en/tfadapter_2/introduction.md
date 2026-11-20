# Introduction

This section describes the intended audience and precautions for using this document.

## Intended Audience

This document is intended for AI algorithm engineers. It describes how to port a training script developed by TensorFlow 2.6.5 Python APIs to the NPU to execute training, and debug and optimize the accuracy and performance.

To better understand this document, you are supposed to:

- Be familiar with the basic CANN architecture and features.
- Be familiar with TensorFlow APIs.
- Possess knowledge of machine learning and deep learning, especially network training basics.
- Be proficient in Python programming.

## Supported  Products

Ascend 950PR&950DT products

Atlas A3 products

Atlas A2 products

Atlas training products

## Precautions

- Before model porting, prepare a model developed in TensorFlow 2.6.5 and the dataset, and ensure that the model runs normally on the GPU or CPU and meet the accuracy and performance requirements. In addition, record the accuracy and performance results for later comparison with those on the  AI processor.
- The code snippets in this document are only examples. Manual tweaking is needed.

## System Constraints and Limitations

- This is a specific guide for TensorFlow 2.6.5.
- Data types float64, complex64, complex128, and DT_VARIANT are not supported.
- Operations related to **tf.Variable** must be performed on the NPU.
- Function operators \(**tf.function**\) must be executed on the NPU.
- The **tf.compat.v1** API must not be used together with eager APIs in TensorFlow 2.6.5.
- The TensorFlow 2.6.5 data preprocessing is performed on the host by default. However, variables need to be offloaded to the device for initialization. If the data preprocessing script contains variables, it may fail to be executed on the NPU. To solve this problem, you need to embed the data preprocessing code that contains variables in **context.device\('CPU:0'\)** to ensure that the variables in the preprocessing code are initialized on the host.

    ```python
    import tensorflow as tf
    from tensorflow.python.eager import context
    with context.device('cpu:0'):
        # Write the data preprocessing code here. The following is only an example.
        x = tf.Variable([1, 2, 3])
        y = tf.square(x)
    ```

- If you spawn processes using the Python package **multiprocessing**, you are advised to use the **forkserver** method instead of the **fork** method.
