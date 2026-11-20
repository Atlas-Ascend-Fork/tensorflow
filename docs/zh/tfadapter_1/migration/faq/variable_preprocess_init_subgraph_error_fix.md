# 变量初始化和数据预处理初始化在同一子图导致训练异常

## 问题现象

某些网络，例如LeNet网络迁移至AI处理器上训练时，loss不收敛，且精度未达预期。

![](../figures/lenet_loss_faq.png)

同时发现变量初始化值全部为0。

## 原因分析

分析用户的训练脚本：

```python
sess.run(tf.group(
    tf.global_variables_initializer(),  # 变量初始化
    tf.local_variables_initializer(),   # 变量初始化
    iterator.initializer                # 数据预处理初始化
    ))
```

基于上面这种写法，TensorFlow在图拆分时，会将变量初始化和数据预处理初始化在同一子图中，变量初始化需要下沉到Device侧执行，而数据预处理初始化无法下沉，导致整个子图无法下沉到AI处理器。

变量初始化在Host侧执行，会导致初始化全0。

## 解决方案

为保证TensorFlow在图拆分时，将变量初始化和数据预处理初始化拆分成不同的子图，建议用户脚本分开执行：

```python
sess.run(tf.group(
    tf.global_variables_initializer(),  # 变量初始化
    tf.local_variables_initializer()    # 变量初始化
    ))
sess.run(iterator.initializer)        # 数据预处理初始化
```
