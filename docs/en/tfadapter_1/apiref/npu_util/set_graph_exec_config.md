# set_graph_exec_config

## Description

Sets the compilation and execution options for a computational graph. After this API is called, configured attributes are added to the fetch node.

## Prototype

```python
def set_graph_exec_config(fetch, dynamic_input=False,
                          dynamic_graph_execute_mode="dynamic_execute",
                          dynamic_inputs_shape_range=None,
                          is_train_graph=False,
                          experimental_config=None)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| fetch | Input | Node executable in the graph. It can be an operation, list, tuple, or tensor, or the name of a tensor.<br>Do not set it to tf.no_op, because this node will be removed when TensorFlow optimizes the graph. |
| dynamic_input | Input | Note: This parameter will be discarded in later versions. You are advised not to set it.<br>Dynamic input flag.<br><br>  - True<br>  - False (default) |
| dynamic_graph_execute_mode | Input | Note: This parameter will be discarded in later versions. You are advised not to set it.<br>Execution mode of a dynamic input. That is, this parameter takes effect when dynamic_input is set to True.<br>dynamic_execute: dynamic graph compilation. In this mode, the shape range configured in dynamic_inputs_shape_range is used for compilation. |
| dynamic_inputs_shape_range | Input | Note: This parameter will be discarded in later versions. You are advised not to set it.<br>Shape range of each dynamic input. If a graph has two dataset inputs and one placeholder input, the configuration is as the example shown below.<br>dynamic_inputs_shape_range="getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]"<br>Precautions:<br><br>  - When this parameter is used, constants cannot be set to user inputs.<br>  - getnext indicates the dataset inputs and data indicates the placeholder inputs.<br>  - The size of a static dimension is specified by a determinant value. The size range of a dynamic dimension is specified by using a tilde (~). A dynamic dimension without size range specified is denoted by –1.<br>  - Assume that your graph has three dataset inputs but the first dataset input has a static shape; the static shape must be specified as shown below.dynamic_inputs_shape_range="getnext:[3,3,4,10],[-1,3,2~1000,-1],[-1,-1,-1,-1]"<br>  - For scalar inputs, you also need to fill in the shape range by using square brackets ([]). No space is allowed before [].<br>  - If there are multiple getnext inputs or data inputs on the network, the input ordering must be preserved. For example:If there are multiple dataset inputs on the network:def func(x):<br>   x = x + 1<br>   y = x + 2<br>   return x,y<br>dataset = tf.data.Dataset.range(min_size, max_size)<br>dataset = dataset.map(func)<br>Assume that the first input of the network is x (with shape range [3~5]) and the second input is y (with shape range [3~6]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.<br>dynamic_inputs_shape_range ="getnext:[3~5],[3~6]"<br>If there are multiple placeholder inputs on the network:If the placeholder names are not specified, for example:<br>x = tf.placeholder(tf.int32)<br>y = tf.placeholder(tf.int32)<br># Set the dynamic ranges of the placeholder inputs in dynamic_inputs_shape_range in the same order as that defined in the script. That is, the first input x (with shape range [3-5]) goes first and the second input y (with shape range [3-6]) follows.<br>dynamic_inputs_shape_range= "data:[3~5],[3~6]"<br>If the placeholder names are specified, for example:<br>x = tf.placeholder(tf.int32,)<br>y = tf.placeholder(tf.int32,)<br>The inputs are in the alphabetical order of the name fields,<br>that is, when setting dynamic_inputs_shape_range, the first input y (with shape range [3~6]) goes first and the second input x (with shape range [3~5]) follows.<br>dynamic_inputs_shape_range = "data:[3~6],[3~5]"<br> NOTICE: If the placeholder names are not specified in the network script, the placeholders are named in the following format:<br>xxx_0, xxx_1, xxx_2, ...<br>The content following the underscore (_) is the sequence index of a placeholder in the network script. Placeholders are arranged in alphabetical order of the index. If the number of placeholders is greater than 10, the sequence is xxx_0 -> xxx_10 -> xxx_2 -> xxx_3. In the network script, the placeholder with index 10 is placed before the placeholder with index 2. As a result, the defined shape range does not match the input placeholder.<br>To avoid this problem, when the number of input placeholders is greater than 10, you are advised to specify the placeholder names in the network script. In this case, the placeholders are named based on the specified names, to associate the shape ranges with the placeholder names.<br>  - If there are multiple dataset inputs on the network:def func(x):<br>   x = x + 1<br>   y = x + 2<br>   return x,y<br>dataset = tf.data.Dataset.range(min_size, max_size)<br>dataset = dataset.map(func)<br>Assume that the first input of the network is x (with shape range [3~5]) and the second input is y (with shape range [3~6]). When configuring the dynamic ranges in dynamic_inputs_shape_range, the ordering must be preserved.<br>dynamic_inputs_shape_range ="getnext:[3~5],[3~6]"<br>  - If there are multiple placeholder inputs on the network:If the placeholder names are not specified, for example:<br>x = tf.placeholder(tf.int32)<br>y = tf.placeholder(tf.int32)<br># Set the dynamic ranges of the placeholder inputs in dynamic_inputs_shape_range in the same order as that defined in the script. That is, the first input x (with shape range [3-5]) goes first and the second input y (with shape range [3-6]) follows.<br>dynamic_inputs_shape_range= "data:[3~5],[3~6]"<br>If the placeholder names are specified, for example:<br>x = tf.placeholder(tf.int32,)<br>y = tf.placeholder(tf.int32,)<br>The inputs are in the alphabetical order of the name fields,<br>that is, when setting dynamic_inputs_shape_range, the first input y (with shape range [3~6]) goes first and the second input x (with shape range [3~5]) follows.<br>dynamic_inputs_shape_range = "data:[3~6],[3~5]"<br> NOTICE: If the placeholder names are not specified in the network script, the placeholders are named in the following format:<br>xxx_0, xxx_1, xxx_2, ...<br>The content following the underscore (_) is the sequence index of a placeholder in the network script. Placeholders are arranged in alphabetical order of the index. If the number of placeholders is greater than 10, the sequence is xxx_0 -> xxx_10 -> xxx_2 -> xxx_3. In the network script, the placeholder with index 10 is placed before the placeholder with index 2. As a result, the defined shape range does not match the input placeholder.<br>To avoid this problem, when the number of input placeholders is greater than 10, you are advised to specify the placeholder names in the network script. In this case, the placeholders are named based on the specified names, to associate the shape ranges with the placeholder names. |
| is_train_graph | Input | Computational graph flag.<br><br>  - True<br>  - False (default) |
| experimental_config | Input | Not recommended in the current version. |

## Returns

fetch

## Restrictions

If both graph-level and session-level parameters are set, the graph-level parameters take precedence over session-level ones.

## Example

Generally, gradient update operations are performed in training networks. The return value of the gradient update operation can be used as the  **fetch**  argument in  **set_graph_exec_config**.

```python
from npu_bridge.estimator.npu import util
train_op = util.set_graph_exec_config(train_op,
                                     dynamic_input=True,
                                     dynamic_inputs_shape_range="data:[1~2];getnext:[1~50,1~50],[1~50,1~50]")
```
