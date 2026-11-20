# set_graph_exec_config

## 功能说明

图级别的配置项接口，用于按计算图设置编译和运行选项。通过该接口调用之后，fetch节点会被打上设置的属性。

## 函数原型

```python
def set_graph_exec_config(fetch, dynamic_input=False,
                          dynamic_graph_execute_mode="dynamic_execute",
                          dynamic_inputs_shape_range=None,
                          is_train_graph=False,
                          experimental_config=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| fetch | 输入 | 图上任意能够执行到的节点，取值包含tensor、operation、list、tuple或者tensor的name。<br>由于tf.no_op节点会在TensorFlow自身进行图处理时优化掉，因此不能输入该节点 |
| dynamic_input | 输入 | 说明：该参数后续版本将废弃，建议不要配置此参数。<br>当前输入是否为动态输入，取值包括：<br> - True：动态输入。<br>  - False：固定输入，默认False。 |
| dynamic_graph_execute_mode | 输入 | 说明：该参数后续版本将废弃，建议不要配置此参数。<br>对于动态输入场景，需要通过该参数设置执行模式，即dynamic_input为True时该参数生效。取值为：<br>dynamic_execute：动态图编译模式。该模式下获取dynamic_inputs_shape_range中配置的shape范围进行编译。 |
| dynamic_inputs_shape_range | 输入 | 说明：该参数后续版本将废弃，建议不要配置此参数。<br>动态输入的shape范围。例如全图有3个输入，两个为dataset输入，一个为placeholder输入，则配置示例为：<br>dynamic_inputs_shape_range="getnext:[128 ,3~5, 2~128, -1],[64 ,3~5, 2~128, -1];data:[128 ,3~5, 2~128, -1]"<br>使用注意事项：<br>  - 使用此参数时，不支持将常量设置为用户输入。<br>  - dataset输入固定标识为“getnext”，placeholder输入固定标识为“data”，不允许用其他表示。<br>  - 动态维度有shape范围的用波浪号“~”表示，固定维度用固定数字表示，无限定范围的用-1表示。<br>  - 对于多输入场景，例如有三个dataset输入时，如果只有第二个第三个输入具有shape范围，第一个输入为固定输入时，仍需要将固定输入shape填入：dynamic_inputs_shape_range="getnext:[3,3,4,10],[-1,3,2~1000,-1],[-1,-1,-1,-1]"<br>  - 对于标量输入，也需要填入shape范围，表示方法为：[]，"[]"前不允许有空格。<br>  - 若网络中有多个getnext输入，或者多个data输入，需要分别保持顺序关系，例如：若网络中有多个dataset输入：def func(x):<br>   x = x + 1<br>   y = x + 2<br>   return x,y<br>dataset = tf.data.Dataset.range(min_size, max_size)<br>dataset = dataset.map(func)<br>网络的第一个输入是x（假设shape range为：[3~5]），第二个输入是y（假设shape range为：[3~6]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range ="getnext:[3~5],[3~6]"<br>若网络中有多个placeholder输入：如果不指定placeholder的name，例：<br>x = tf.placeholder(tf.int32)<br>y = tf.placeholder(tf.int32)<br>placeholder的顺序和脚本中定义的位置一致，即网络的第一个输入是x（假设shape range为：[3~5]），第二个输入是y（shape range为：[3~6]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range= "data:[3~5],[3~6]"<br>如果指定了placeholder的name，例：<br>x = tf.placeholder(tf.int32,)<br>y = tf.placeholder(tf.int32,)<br>则网络输入的顺序按name的字母序排序，即<br>即网络的第一个输入是y（假设shape range为：[3~6]），第二个输入是x（shape range为：[3~5]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range = "data:[3~6],[3~5]"<br> 须知： 若网络脚本中未指定placeholder的name，则placeholder会按照会如下格式命名：<br>xxx_0，xxx_1，xxx_2，……<br>其中下划线后为placeholder在网络脚本中的定义顺序索引，placeholder会按照此索引的字母顺序进行排布，所以当placeholder的个数大于10时，则排序为“xxx_0 -> xxx_10 -> xxx_2 -> xxx_3”，网络脚本中定义索引为10的placeholder排在了索引为2的placeholder前面，导致定义的shape range与实际输入的placeholder不匹配。<br>为避免此问题，当placeholder的输入个数大于10时，建议在网络脚本中指定placeholder的name，则placeholder会以指定的name进行命名，实现shape range与placeholder name的关联。<br>  - 若网络中有多个dataset输入：def func(x):<br>   x = x + 1<br>   y = x + 2<br>   return x,y<br>dataset = tf.data.Dataset.range(min_size, max_size)<br>dataset = dataset.map(func)<br>网络的第一个输入是x（假设shape range为：[3~5]），第二个输入是y（假设shape range为：[3~6]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range ="getnext:[3~5],[3~6]"<br>  - 若网络中有多个placeholder输入：如果不指定placeholder的name，例：<br>x = tf.placeholder(tf.int32)<br>y = tf.placeholder(tf.int32)<br>placeholder的顺序和脚本中定义的位置一致，即网络的第一个输入是x（假设shape range为：[3~5]），第二个输入是y（shape range为：[3~6]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range= "data:[3~5],[3~6]"<br>如果指定了placeholder的name，例：<br>x = tf.placeholder(tf.int32,)<br>y = tf.placeholder(tf.int32,)<br>则网络输入的顺序按name的字母序排序，即<br>即网络的第一个输入是y（假设shape range为：[3~6]），第二个输入是x（shape range为：[3~5]），配置到dynamic_inputs_shape_range中时，需要保持顺序关系，即<br>dynamic_inputs_shape_range = "data:[3~6],[3~5]"<br> 须知： 若网络脚本中未指定placeholder的name，则placeholder会按照会如下格式命名：<br>xxx_0，xxx_1，xxx_2，……<br>其中下划线后为placeholder在网络脚本中的定义顺序索引，placeholder会按照此索引的字母顺序进行排布，所以当placeholder的个数大于10时，则排序为“xxx_0 -> xxx_10 -> xxx_2 -> xxx_3”，网络脚本中定义索引为10的placeholder排在了索引为2的placeholder前面，导致定义的shape range与实际输入的placeholder不匹配。<br>为避免此问题，当placeholder的输入个数大于10时，建议在网络脚本中指定placeholder的name，则placeholder会以指定的name进行命名，实现shape range与placeholder name的关联。 |
| is_train_graph | 输入 | 标记该图是否为计算图。<br><br>  - True：是计算图<br>  - False：不是计算图，默认False。 |
| experimental_config | 输入 | 当前版本暂不推荐使用。 |

## 返回值

fetch

## 约束说明

如果同时设置了图级别的参数和session级别的参数，则图级别的参数优先级高。

## 调用示例

一般训练网络中都会执行梯度更新操作，可以将梯度更新操作的返回值作为set_graph_exec_config的fetch入参：

```python
from npu_bridge.estimator.npu import util
train_op = util.set_graph_exec_config(train_op,
                                     dynamic_input=True,
                                     dynamic_inputs_shape_range="data:[1~2];getnext:[1~50,1~50],[1~50,1~50]")
```
