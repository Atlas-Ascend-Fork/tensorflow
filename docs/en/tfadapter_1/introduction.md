# Learning Wizard

This section describes the intended audience, porting process of a TensorFlow 1.15-based model, and precautions for using this document.

## Intended Audience

This document is intended for AI algorithm engineers and describes how to port network scripts developed using TensorFlow 1.15-based Python APIs to run on the NPU. The NPU supports porting of network scripts developed with three TensorFlow 1.15 APIs: Estimator, sess.run, and Keras.

To better understand this document, you should:

- Be familiar with the basic CANN architecture and features.
- Be familiar with TensorFlow 1.15 APIs.
- Possess knowledge on machine learning and deep learning, especially network training basics.
- Be proficient in Python programming.

## Supported  Products

- Ascend 950PR&950DT products
- Atlas A3 products
- Atlas A2 products
- Atlas training products
- Atlas inference products  (only supporting online inference)

## Precautions

- Before model porting to the NPU, prepare the training model developed based on TensorFlow 1.15 and the corresponding dataset. Ensure that it runs successfully on a GPU or CPU and meets the expected accuracy and performance requirements, and record the relevant accuracy and performance metrics for subsequent comparison on the NPU.
- The code snippets in this document are for reference only. Modify and adapt them accordingly before use.

## Restrictions

1. This is a specific guide for TensorFlow 1.15.
2. Data types float64, complex64, complex128, and DT_VARIANT are not supported.
3. Supported data formats include NCHW, NHWC, NC, HWCN, and CN.
4. For condition branches and loop branches, only **tf.cond**, **tf.while_loop**, and **tf.case** are supported.
5. During multi-device training, **NPURunConfig** does not support **save_checkpoints_secs** in **tf.estimator.RunConfig**.
6. During multi-device training, saving the summary information \(via the **tf.summary** API\) of only a single device is not supported.
7. For the  Atlas training products, the operators do not support the Inf or NaN inputs.
8. During data preprocessing, only dataset and placeholder modes are supported for data reading; queue-based data reading is not supported.
9. If you spawn processes using the Python package **multiprocessing**, you are advised to use the **forkserver** method as opposed to the **fork** method.

    In Python versions 3.8 to 3.11, using the **fork** method may copy the lock state of the main process when a child process is created. If this child process subsequently attempts to acquire the lock, a deadlock may occur, which in turn causes the service process to hang.
