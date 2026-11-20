# What Do I Do If Operator Overflows/Underflows Due to Abnormal Model Hyperparameter in Mixed-Precision Scenarios?

## Symptom

In the SSD-ResNet50V1-FPN network, static loss scaling is set to **1** when the **allow_mix_precision** mode is used. After the single-device training is performed, it is found that **global_step** is not updated and overflow occurs for Sub fusion operators in each iteration.

![](../figures/SSD-ResNet50V1-FPN-faq.png)

## Possible Cause

In mixed-precision scenarios, the compute in line 75 has been optimized to float16 high-performance compute. However, when the constant is converted to float16, the expression range of float16 is exceeded and becomes 0. As a result, a division-by-zero error occurs in line 81, causing floating-point overflow.

In a model, the constant is a common avoidance coefficient for division-by-zero errors, which occurs in the coordinate system conversion phase of the box encoder. It is used in the scenario in which the width and height are 0. This parameter is not user-friendly in supporting mixed-precision training, thereby resulting in overflow.

![](../figures/SSD-ResNet50V10FPN-reason.png)

## Solution

If model convergence becomes poor due to a few extreme model hyperparameters, you can adjust EPSILON to a value that can be expressed by float16. For example, 1e-4 is recommended. If the model hyperparameters can be stably converged, this modification does not affect model convergence.
