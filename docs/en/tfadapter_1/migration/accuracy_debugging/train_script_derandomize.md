# Disabling Random Preprocessings in the Training Script

## Background

Before starting network comparison, disable all random operations for image preprocessing in your training script. Failure to do so will result in unavailable comparison result due to inconsistent input data.

## Procedure

Edit the training script to disable random preprocessing.

```python
# The following example is for reference only.
# 1. Shuffle the input data.
dataset = tf.data.TFRecordDataset(tf_data)
dataset = dataset.shuffle(batch_size*10)    # Comment out the line.

# 2. Use Dropout.
net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b') # Best comment out the line.

# 3. Perform random operations for image preprocessing. (Comment out the operations or replace them with non-random operations.)
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

# Other...
```

## Validation

Validate that your training script will not execute random preprocessing.

1. Perform NPU training twice and perform network accuracy comparison. Check whether the cosine similarity is greater than 0.98.
    1. Perform  [Dumping User Model on NPU](network_accuracy_comparison.md#dumping-user-model-on-npu).
    2. Run the  [vc -lt \[left_path\] -rt \[right_path\] -g \[graph\]](precision_tool_command_ref.md#vc--lt-left_path--rt-right_path--g-graph)  command to compare the two networks. A CSV file will be generated.

        **PrecisionTool \> vc -lt /path/left -rt /path/right**

    3. Run the  [vcs -f \[file_name\] -c \[cos_sim_threshold\] -l \[limit\]](precision_tool_command_ref.md#vcs--f-file_name--c-cos_sim_threshold--l-limit) command. If the cosine similarity is greater than 0.98, random preprocessings are disabled.

        **PrecisionTool \> vcs**

2. Perform GPU or CPU training twice and compare the similarity of tensors based on .npy file names. If the similarity reaches the specified threshold, random preprocessing is disabled.
