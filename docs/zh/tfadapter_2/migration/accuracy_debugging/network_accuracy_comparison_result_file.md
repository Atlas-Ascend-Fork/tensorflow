# 整网精度比对结果文件说明

整网精度比对会生成比对结果文件result_\*.csv，文件内容如下图所示：

**图 1**  比对结果示例
![](../figures/comparison_result_example.png "比对结果示例")

文件参数说明如下表所示：

| 参数 | 说明 |
| :--- | :--- |
| Index | 网络模型中算子的ID。 |
| OpSequence | 部分算子比对时算子运行的序列。即-f参数指定的全网层信息文件中算子的ID。仅配置-r或-s参数时展示。 |
| OpType | 算子类型。指定-f参数时获取算子类型。 |
| NPUDump | 表示My Output模型的算子名。 |
| DataType | 表示NPU侧dump数据的算子数据类型。 |
| Address | dump tensor的内存地址。用于判断算子的内存问题。仅基于AI处理器运行生成的dump数据文件在整网比对时可提取该数据。 |
| GroundTruth | 表示Ground Truth模型的算子名。 |
| DataType | 表示Ground Truth侧数据算子的数据类型。 |
| TensorIndex | 表示基于AI处理器运行生成的dump数据的算子的input ID和output ID。 |
| Shape | 比对的Tensor的Shape。 |
| OverFlow | 溢出算子。显示YES表示该算子存在溢出；显示NO表示算子无溢出；显示NaN表示不做溢出检测。配置-overflow_detection参数时展示。 |
| CosineSimilarity | 进行余弦相似度算法比对出来的结果，取值范围为：[-1, 1]，比对的结果如果越接近1，表示两者的值越相近，越接近-1意味着两者的值越相反。 |
| MaxAbsoluteError | 进行最大绝对误差算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| AccumulatedRelativeError | 进行累积相对误差算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| RelativeEuclideanDistance | 进行欧氏相对距离算法比对出来的结果，取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| KullbackLeiblerDivergence | 进行KL散度算法比对出来的结果，取值范围为0到无穷大。KL散度越小，真实分布与近似分布之间的匹配越好。 |
| StandardDeviation | 进行标准差算法比对出来的结果，取值范围为0到无穷大。标准差越小，离散度越小，表明越接近平均值。 |
| MeanAbsoluteError | 表示平均绝对误差。取值范围为0到无穷大，MeanAbsoluteError趋于0，RootMeanSquareError趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RootMeanSquareError ≥ MeanAbsoluteError恒成立。 |
| RootMeanSquareError | 表示均方根误差。取值范围为0到无穷大，MeanAbsoluteError趋于0，RootMeanSquareError趋于0，说明测量值与真实值越近似；MeanAbsoluteError趋于0，RootMeanSquareError越大，说明存在局部过大的异常值；MeanAbsoluteError越大，RootMeanSquareError等于或近似MeanAbsoluteError，说明整体偏差越集中；MeanAbsoluteError越大，RootMeanSquareError越大于MeanAbsoluteError，说明存在整体偏差，且整体偏差分布分散；不存在以上情况的例外情况，因为RootMeanSquareError ≥ MeanAbsoluteError恒成立。 |
| MaxRelativeError | 表示最大相对误差。取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| MeanRelativeError | 表示平均相对误差。取值范围为0到无穷大，值越接近于0，表明越相近，值越大，表明差距越大。 |
| CompareFailReason | 算子无法比对的原因。若余弦相似度为1，则查看该算子的输入或输出Shape是否为空或全部为1，若为空或全部为1则算子的输入或输出为标量，提示：this tensor is scalar。 |
