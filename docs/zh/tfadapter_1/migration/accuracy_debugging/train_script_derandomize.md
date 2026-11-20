# 训练脚本去随机处理

## 背景介绍

整网数据比对前，需要先检查并去除训练脚本内部使用到的随机处理，避免由于输入数据不一致导致数据比对结果不可用。

## 操作方法

修改训练脚本，去除随机处理：

```python
# 此处给出一些典型示例，需要根据自己的脚本进行排查
# 1. 对输入数据做shuffle操作
dataset = tf.data.TFRecordDataset(tf_data)
dataset = dataset.shuffle(batch_size*10)    # 直接注释掉该行

# 2. 使用Dropout
net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b') # 建议注释该行

# 3. 图像预处理使用随机的操作(根据实际情况注释，或者替换成其他固定的预处理操作)
# Random rotate
random_angle = tf.random_uniform([], - self.degree * 3.141592 / 180, self.degree * 3.141592 / 180)
image = tf.contrib.image.rotate(image, random_angle, interpolation='BILINEAR')
depth_gt = tf.contrib.image.rotate(depth_gt, random_angle, interpolation='NEAREST')

# Random flipping
do_flip = tf.random_uniform([], 0, 1)
image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
depth_gt = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(depth_gt), lambda: depth_gt)

# Random crop
image_depth = tf.concat([image, depth_gt], 2)
image_depth_cropped = tf.random_crop(image_depth, [self.params.height, self.params.width, 4])

# 其他……
```

## 验证方法

修改完训练脚本后，有两种检查方法，验证所有的随机处理是否已经规避掉：

1. NPU训练两次，进行整网精度比对，检查精度数据的余弦相似度是否大于0.98。具体方法为：
    1. [基于NPU Dump精度数据](network_accuracy_comparison.md#基于NPU-Dump精度数据)。
    2. 使用[vc -lt \[left_path\] -rt \[right_path\] -g \[graph\]](precision_tool_command_ref.md#vc--lt-left_path--rt-right_path--g-graph)命令比较，生成csv文件。

        **PrecisionTool \> vc -lt /path/left -rt /path/right**

    3. 使用[vcs -f \[file_name\] -c \[cos_sim_threshold\] -l \[limit\]](precision_tool_command_ref.md#vcs--f-file_name--c-cos_sim_threshold--l-limit)命令，如果余弦相似度大于0.98，表明已经去随机。

        **PrecisionTool \> vcs**

2. GPU或CPU训练两次，根据文件npy名比较tensor数据相似度，相似度结果达到一定阈值，表明已经去随机。
