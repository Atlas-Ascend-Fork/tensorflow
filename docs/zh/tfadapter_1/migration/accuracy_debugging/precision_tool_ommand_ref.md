# precision_tool命令参考

## ac -l [limit_num] -c

- 命令说明：

  自动化检测命令，列出Fusion信息，解析算子溢出信息。

  - -c：可选，进行整网比对。
  - -l：可选，限制输出结果的条数（overflow解析的条数等）。

- 命令示例：

    ```bash
    PrecisionTool > ac -c
    ```

- 执行结果：

    ```text
    ╭─────────────────────────────────────────────────╮
    │ [TransData][327] trans_TransData_1170                                                            │
    │  - [AI Core][Status:32][TaskId:327] ['浮点计算有溢出']                                           │
    │  - First overflow file timestamp [1619347786532995] -                                            │
    │  |- TransData.trans_TransData_1170.327.1619347786532995.input.0.npy                              │
    │   |- [Shape: (32, 8, 8, 320)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.11950836181640626] │
    │  |- TransData.trans_TransData_1170.327.1619347786532995.output.0.npy                             │
    │   |- [Shape: (32, 20, 8, 8, 16)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.07781982421875] │ ╰─────────────────────────────────────────────────╯
    ```

## run [command]

- 命令说明：

    不退出交互命令环境执行shell命令，当内置命令冲突时，需要加run前缀。

- 命令示例：

    ```bash
    PrecisionTool > run vim cli.py
    PrecisionTool > vim cli.py
    ```

## ls -n [op_name] -t [op_type] -f [fusion_pass] -k [kernel_name]

- 命令说明：

    通过[算子名]/[算子类型]查询网络里的算子，模糊匹配。

  - -n：可选，算子节点名称。
  - -t：可选，算子类型。
  - -f：可选，融合类型。
  - -k：可选，kernel name。

    **说明：**-n与-t需要存在其中一个输入。

- 命令示例：

    ```bash
    PrecisionTool > ls -t Mul -n mul_3 -f TbeMulti
    ```

- 执行结果：

    ```text
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5b/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5c/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5d/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_6b/Branch_1/mul_3
    ```

## ni -n [op_name] -g [graph] -a [attr] -s [save subgraph depth]

- 命令说明：

    通过[算子名]查询算子节点信息。

  - -n：可选，指定节点名称。
  - -g：可选，graph名称。
  - -a：可选，显示attr信息。
  - -s：可选，保存一个以当前算子节点为根，深度为参数值的子图。

    **说明：**-n与-g需要存在其中一个输入。

- 命令示例：

    ```text
    PrecisionTool >  ni -n gradients/InceptionV3/InceptionV3/Mixed_7a/Branch_0/Maximum_1_grad/GreaterEqual -s 3
    ```

- 执行结果：

    ```text
    ╭─────[GreaterEqual]gradients/InceptionV3/InceptionV3/Mixed_7a/Branch_0/Maximum_1_grad/GreaterEqual───────────────╮
    │ [GreaterEqual] gradients/InceptionV3/InceptionV3/Mixed_7a/Branch_0/Maximum_1_grad/GreaterEqual                                       │
    │ Input:                                                                                                                               │
    │  -[0][DT_FLOAT][NHWC][32, 8, 8, 320] InceptionV3/InceptionV3/Mixed_7a/Branch_0/add_3:0                                               │
    │  -[1][DT_FLOAT][NHWC][1, 8, 1, 1] InceptionV3/Mixed_7a/Branch_0/Conv2d_1a_3x3tau:0                                                   │
    │  -[2][][[]][] atomic_addr_clean0_21:-1                                                                                               │
    │ Output:                                                                                                                              │
    │  -[0][DT_BOOL][NHWC][32, 8, 8, 320] ['trans_TransData_1170']                                                                         │
    │ NpuDumpInput:                                                                                                                        │
    │  -[0] GreaterEqual.gradients_InceptionV3_InceptionV3_Mixed_7a_Branch_0_Maximum_1_grad_GreaterEqual.325.1619494134722860.input.0.npy  │
    │   |- [Shape: (32, 8, 8, 320)] [Dtype: float32] [Max: 5.846897] [Min: -8.368301] [Mean: -0.72565556]                                  │
    │  -[1] GreaterEqual.gradients_InceptionV3_InceptionV3_Mixed_7a_Branch_0_Maximum_1_grad_GreaterEqual.325.1619494134722860.input.1.npy  │
    │   |- [Shape: (1, 8, 1, 1)] [Dtype: float32] [Max: 0.0] [Min: 0.0] [Mean: 0.0]                                                        │
    │ NpuDumpOutput:                                                                                                                       │
    │  -[0] GreaterEqual.gradients_InceptionV3_InceptionV3_Mixed_7a_Branch_0_Maximum_1_grad_GreaterEqual.325.1619494134722860.output.0.npy │
    │   |- [Shape: (32, 8, 8, 320)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.1176300048828125]                                      │
    │ CpuDumpOutput:                                                                                                                       │
    │  -[0] gradients_InceptionV3_InceptionV3_Mixed_7a_Branch_0_Maximum_1_grad_GreaterEqual.0.1619492699305998.npy                         │
    │   |- [Shape: (32, 8, 8, 320)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.11764373779296874]                                     │
    ╰───────────────────────────────────────────────────────────────────╯
    2021-04-27 14:39:55 (15178) -[DEBUG]write 14953 bytes to './precision_data/dump/temp/op_graph/GreaterEqual.gradients_InceptionV3_InceptionV3_Mixed_7a_Branch_0_Maximum_1_grad_GreaterEqual.3.gv'
    2021-04-27 14:39:55 (15178) -[INFO]Sub graph saved to /root/sym/inception/precision_data/dump/temp/op_graph
    ```

## pt -n [\*.npy]

- 命令说明：

    查看某个dump数据块的数据信息，并保存到txt文件。

    -n：可选，待查看的数据文件名。

- 命令示例：

    ```bash
    PrecisionTool > pt TransData.trans_TransData_1170.327.1619347786532995.input.0.npy
    ```

- 执行结果：

    ```text
    ╭────────────────────────────────────────────────────────────╮
    │ Shape: (32, 8, 8, 320)                                                                                                  │
    │ Dtype: bool                                                                                                             │
    │ Max: True                                                                                                               │
    │ Min: False                                                                                                              │
    │ Mean: 0.11950836181640626                                                                                               │
    │ Path: ./precision_data/dump/temp/overflow_decode/TransData.trans_TransData_1170.327.1619347786532995.input.0.npy        │
    │ TxtFile: ./precision_data/dump/temp/overflow_decode/TransData.trans_TransData_1170.327.1619347786532995.input.0.npy.txt │
    ╰────────────────────────────────────────────────────────────╯
    ```

## cp -n [left \*.npy] [right \*.npy] -p [print num] -al [atol] -rl [rtol] -s

- 命令说明：

    对比两个numpy文件中的tensor数据。

  - -n：必选，指定需要对比的两个numpy文件的文件名。
  - -p：可选，指定输出前多少个错误数据。
  - -al/rl：可选，al为绝对误差，rl为相对误差，使用示例如下：

    ```python
    # 示例1：
    np.allclose(left, right, atol=al, rtol=rl)
    # 示例2：
    err_cnt += 1 if abs(data_left[i] - data_right[i]) > (al + rl * abs(data_right[i])) else 0
    ```

  - -s：可选，保存成txt文件，默认打开。

- 命令示例：

    ```bash
    PrecisionTool > cp -n file0.npy file1.npy -p 10 -al 0.002 -rl 0.005 -s
    ```

- 执行结果：

    ```text
                      Error Item Table                                        Top Item Table
    ┏━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
    ┏━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    ┃ Index ┃ Left          ┃ Right        ┃ Diff         ┃ ┃ Index ┃ Left        ┃ Right       ┃ Diff          ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩ ┡━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    │ 155   │ 0.024600908   │ 0.022271132  │ 0.002329776  │ │ 0     │ -0.9206961  │ -0.9222216  │ 0.0015255213  │
    │ 247   │ 0.015752593   │ 0.017937578  │ 0.0021849852 │ │ 1     │ -0.6416973  │ -0.64051837 │ 0.0011789203  │
    │ 282   │ -0.0101207765 │ -0.007852031 │ 0.0022687456 │ │ 2     │ -0.35383835 │ -0.35433492 │ 0.0004965663  │
    │ 292   │ 0.019581757   │ 0.02240482   │ 0.0028230622 │ │ 3     │ -0.18851271 │ -0.18883198 │ 0.00031927228 │
    │ 640   │ -0.06593232   │ -0.06874806  │ 0.0028157383 │ │ 4     │ -0.43508735 │ -0.43534422 │ 0.00025686622 │
    │ 1420  │ 0.09293677    │ 0.09586689   │ 0.0029301196 │ │ 5     │ 1.4447614   │ 1.4466647   │ 0.0019032955  │
    │ 1462  │ -0.085207745  │ -0.088047795 │ 0.0028400496 │ │ 6     │ -0.3455438  │ -0.3444429  │ 0.0011008978  │
    │ 1891  │ -0.03433288   │ -0.036525503 │ 0.002192624  │ │ 7     │ -0.6560242  │ -0.6564579  │ 0.0004336834  │
    │ 2033  │ 0.06828873    │ 0.07139922   │ 0.0031104907 │ │ 8     │ -2.6964858  │ -2.6975214  │ 0.0010356903  │
    │ 2246  │ -0.06376442   │ -0.06121233  │ 0.002552092  │ │ 9     │ -0.73746175 │ -0.73650354 │ 0.00095820427 │
    └───────┴───────────────┴──────────────┴──────────────┘ └───────┴─────────────┴─────────────┴───────────────┘
    ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ Left:                                                                                                                                    │
    │  |- NpyFile: ./precision_data/dump/temp/decode/Add.InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.323.1619494134703053.output.0.npy     │
    │  |- TxtFile: ./precision_data/dump/temp/decode/Add.InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.323.1619494134703053.output.0.npy.txt │
    │  |- NpySpec: [Shape: (32, 8, 8, 320)] [Dtype: float32] [Max: 5.846897] [Min: -8.368301] [Mean: -0.72565556]                              │
    │ DstFile:                                                                                                                                 │
    │  |- NpyFile: ./precision_data/dump/cpu/InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.0.1619492699305998.npy                            │
    │  |- TxtFile: ./precision_data/dump/cpu/InceptionV3_InceptionV3_Mixed_7a_Branch_0_add_3.0.1619492699305998.npy.txt                        │
    │  |- NpySpec: [Shape: (32, 8, 8, 320)] [Dtype: float32] [Max: 5.8425903] [Min: -8.374472] [Mean: -0.7256237]                              │
    │ NumCnt:   655360                                                                                                                         │
    │ AllClose: False                                                                                                                          │
    │ CosSim:   0.99999493                                                                                                                     │
    │ ErrorPer: 0.023504638671875  (rl= 0.005, al= 0.002)                                                                                      │
    ╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```

## vcs -f [file_name] -c [cos_sim_threshold] -l [limit]

- 命令说明：

    查看精度比对结果的概要信息，可以根据余弦相似度阈值过滤出低于阈值的算子/信息。

  - -f (--file) ：可选，指定csv文件，不设置则默认遍历precision_data/temp/vector_compare/目录下最近产生的对比目录内的所有csv。
  - -c (--cos_sim) ：可选，指定筛选所使用的余弦相似度阈值，默认值是0.98。
  - -l (--limit) ：可选，指定输出前多少个结果，默认值是3。

- 命令示例：

    ```bash
    PrecisionTool > vcs -c 0.98 -l 2
    ```

- 执行结果：

    ```text
     2021-05-31 14:48:56 (2344298) -[INFO]Sub path num:[1]. Dirs[['20210529145750']], choose[20210529145750]
     2021-05-31 14:48:56 (2344298) -[DEBUG]Find ['result_20210529145751.csv', 'result_20210529145836.csv', 'result_20210529145837.csv', 'result_20210529145849.csv', 'result_20210529150404.csv', 'result_20210529151102.csv'] result files in dir precision_data/temp/vector_compare/20210529145750
     2021-05-31 14:48:56 (2344298) -[INFO]Find 0 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529145751.csv
     2021-05-31 14:48:56 (2344298) -[INFO]Find 0 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529145836.csv
     2021-05-31 14:48:56 (2344298) -[INFO]Find 1 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529145837.csv
     2021-05-31 14:48:56 (2344298) -[INFO]Find 2 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529145849.csv
     2021-05-31 14:48:56 (2344298) -[INFO]Find 2 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529150404.csv
     2021-05-31 14:48:56 (2344298) -[INFO]Find 0 ops less than 0.98 in precision_data/temp/vector_compare/20210529145750/result_20210529151102.csv
     ╭── [578] pixel_cls_loss/cond_1/TopKV2 ───╮
     │ Left:  ['pixel_cls_loss/cond_1/TopKV2']       │
     │ Right: ['pixel_cls_loss/cond_1/TopKV2']       │
     │ Input:                                        │
     │  - [0]1.0        - [1]nan                     │
     │ Output:                                       │
     │  - [0]0.999999   - [1]0.978459                │
     ╰───────────────────────╯
     ╭── [490] gradients/AddN_5 ───╮
     │ Left:  ['gradients/AddN_5']       │
     │ Right: ['gradients/AddN_5']       │
     │ Input:                            │
     │  - [0]nan        - [1]1.0         │
     │ Output:                           │
     │  - [0]0.05469                     │
     ╰─────────────────╯
    ```

## vc -lt [left_path] -rt [right_path] -g [graph]

- 命令说明：

    手动指定两个目录，进行整网精度比对。

  - -lt：必选，其中一个文件目录。
  - -rt：必选，另一个目录，一般指标杆目录。

    > [!NOTE]说明
    > 需要指定到dump数据所在的目录层级，例如：precision_data/npu/debug_0/dump/20220217095546/3/ge_default_20220217095547_1/1/0/。

  - -g：可选，指定-g将尝试解析graph内的映射关系比对（一般用于NPU和TensorFlow之间的数据比对，NPU与NPU之间比对不需要，直接按照算子name对比）。

- 命令示例：

    ```bash
    PrecisionTool > vc -lt /path/left -rt /path/right
    ```
