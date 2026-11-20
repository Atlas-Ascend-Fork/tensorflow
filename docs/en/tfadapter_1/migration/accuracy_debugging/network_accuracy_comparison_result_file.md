# Network Accuracy Comparison Result File

The comparison result file **result_\*.csv** is generated after the network accuracy comparison is complete. The following figure shows an example of the file content.

![](../figures/precision_compare_result.png "comparison-result-example")

The following table describes the parameters in the file.

| Parameter | Description |
| --- | --- |
| Index | ID of an operator in a network model. |
| OpSequence | Sequence in which an operator runs during comparison on some operators. That is, ID of the operator in the network-wide information file specified by the -f parameter. This option is displayed only when -r or -s is configured. |
| OpType | Operator type. It is used to obtain the operator type when the -f option is specified. |
| NPUDump | Operator name of the My Output model. |
| DataType | Dump data type of operators on the NPU side. |
| Address | Memory address of the dump tensor, which detects memory faults of an operator. The address can be extracted only for network-wide comparison of dump data files generated during network running on the AI processor. |
| GroundTruth | Operator name of the Ground Truth model. |
| DataType | Data type of operators on the Ground Truth side. |
| TensorIndex | Input ID and output ID of the operator that generates the dump data during running on the AI processor. |
| Shape | Shape of the compared tensor. |
| OverFlow | Overflow/Underflow operator. YES indicates that overflow/underflow occurs on an operator. NO indicates that no overflow/underflow occurs on the operator. NaN indicates that overflow/underflow detection is not performed. This option is displayed when -overflow_detection is set. |
| CosineSimilarity | Result of the cosine similarity comparison. The value ranges from –1 to 1. A value closer to 1 indicates higher similarity. |
| MaxAbsoluteError | Result of the maximum absolute error comparison. The value ranges from 0 to infinity. A value closer to 0 indicates higher similarity. |
| AccumulatedRelativeError | Result of the accumulated relative error comparison. The value ranges from 0 to infinity. A value closer to 0 indicates higher similarity. |
| RelativeEuclideanDistance | Result of the Euclidean relative distance comparison. The value ranges from 0 to infinity. A value closer to 0 indicates higher similarity. |
| KullbackLeiblerDivergence | Result of the Kullback-Leibler divergence comparison. The value ranges from 0 to infinity. The smaller the Kullback-Leibler divergence, the closer the approximate distribution is to the true distribution. |
| StandardDeviation | Result of the standard deviation comparison. The value ranges from 0 to infinity. The smaller the standard deviation is, the smaller the dispersion is, and the closer the value is to the average value. |
| MeanAbsoluteError | Mean absolute error. The value ranges from 0 to infinity. If values of both MeanAbsoluteError and RootMeanSquareError are close to 0, the measured value is more approximate to the actual value. If the value of MeanAbsoluteError is close to 0, a larger value of RootMeanSquareError indicates that some values are excessively large. A larger value of MeanAbsoluteError and RootMeanSquareError value equal to or approximate to that of MeanAbsoluteError indicate that the overall deviation is more centralized. A larger value of MeanAbsoluteError and RootMeanSquareError value larger than that of MeanAbsoluteError indicate that the overall deviation exists and its distribution is scattered. Other situations do not exist because "RootMeanSquareError ≥ MeanAbsoluteError" is always true. |
| RootMeanSquareError | Root mean square error. The value ranges from 0 to infinity. If values of both MeanAbsoluteError and RootMeanSquareError are close to 0, the measured value is more approximate to the actual value. If the value of MeanAbsoluteError is close to 0, a larger value of RootMeanSquareError indicates that some values are excessively large. A larger value of MeanAbsoluteError and RootMeanSquareError value equal to or approximate to that of MeanAbsoluteError indicate that the overall deviation is more centralized. A larger value of MeanAbsoluteError and RootMeanSquareError value larger than that of MeanAbsoluteError indicate that the overall deviation exists and its distribution is scattered. Other situations do not exist because "RootMeanSquareError ≥ MeanAbsoluteError" is always true. |
| MaxRelativeError | Max. relative error. The value ranges from 0 to infinity. A value closer to 0 indicates higher similarity. |
| MeanRelativeError | Mean relative error. The value ranges from 0 to infinity. A value closer to 0 indicates higher similarity. |
| CompareFailReason | Cause of the comparison failure.<br>If the cosine similarity is 1, check whether the input or output shapes of the operator are empty or all 1. If yes, the input or output of the operator is a scalar. In this case, the following message is displayed: "this tensor is scalar." |
