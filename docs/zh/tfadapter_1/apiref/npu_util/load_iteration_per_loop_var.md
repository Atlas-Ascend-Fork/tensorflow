# load_iteration_per_loop_var

## 功能说明

该接口和[create_iteration_per_loop_var](create_iteration_per_loop_var.md)接口配合使用，用来实现sess.run模式下设置小循环次数，即每次sess.run\(\)在Device侧执行训练迭代的次数。

## 函数原型

```python
def load_iteration_per_loop_var(self, sess, iterations_per_loop=1)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sess | 输入 | 已经创建的TensorFlow会话。 |
| iterations_per_loop | 输入 | 每次sess.run()，在Device侧执行训练迭代的次数，默认为1，且训练迭代总次数必须为iterations_per_loop的整数倍。|

## 返回值

无

## 约束说明

混合计算模式（mix_compile_mode为True）时，iterations_per_loop必须为1。

## 调用示例

```python
from npu_bridge.npu_init import *

config = tf.ConfigProto(allow_soft_placement=True)
custom_op =  config.graph_options.rewrite_options.custom_optimizers.add()
custom_op.name =  "NpuOptimizer"
custom_op.parameter_map["enable_data_pre_proc"].b = True # 若网络中存在GetNext算子，需要设置GetNext算子下沉，GetNext算子下沉是迭代循环下沉的必要条件
custom_op.parameter_map["iterations_per_loop"].i = 10  # 此处设置的值和load_iteration_per_loop_var接口设置的iterations_per_loop参数值保持一致，用于功能校验
config = npu_config_proto(config_proto=config)

# 训练模型
with tf.Session(config=config) as sess:
    sess.run(init)
    # sess.run模式下设置小循环次数为10
    iteration = util.IterationPerLoop()
    train_op = iteration.create_iteration_per_loop_var(optimizer) #修改图
    tf.train.Supervisor(logdir="/home/xxxx",init_op=init)  #冻结图
    iteration.load_iteration_per_loop_var(sess, 10)  #设置小循环次数

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_op, cost], feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch
```
