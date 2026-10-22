# precision_tool Command Reference

## ac -l [limit_num] -c

- Description:

    Runs automatic detection (including displaying fusion information and parsing operator overflow/underflow information).

  - **-c**: (optional) performs network-wide comparison.
  - **-l**: (optional) limits the number of output results, such as the number of overflow parsing results.

- Example:

    ```bash
    PrecisionTool > ac -c
    ```

- Command output:

    ```text
    ╭─────────────────────────────────────────────────╮
    │ [TransData][327] trans_TransData_1170                                                            │
    │  - [AI Core][Status:32][TaskId:327] ['Overflow or underflow occurs']                                           │
    │  - First overflow file timestamp [1619347786532995] -                                            │
    │  |- TransData.trans_TransData_1170.327.1619347786532995.input.0.npy                              │
    │   |- [Shape: (32, 8, 8, 320)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.11950836181640626] │
    │  |- TransData.trans_TransData_1170.327.1619347786532995.output.0.npy                             │
    │   |- [Shape: (32, 20, 8, 8, 16)] [Dtype: bool] [Max: True] [Min: False] [Mean: 0.07781982421875] │ ╰─────────────────────────────────────────────────╯
    ```

## run [command]

- Description:

    Runs a shell command without exiting the interactive command-line environment, used when the command to run conflicts with a built-in command.

- Example:

    ```bash
    PrecisionTool > run vim cli.py
    PrecisionTool > vim cli.py
    ```

## ls -n [op_name] -t [op_type] -f [fusion_pass] -k [kernel_name]

- Description:

    Lists all operators on the network that match the specified operator name and operator type. Fuzzy matching is supported.

  - **-n**: (optional) operator node name.
  - **-t**: (optional) operator type.
  - **-f**: (optional) fusion type.
  - **-k**: (optional) kernel name.

    **Note**: Either **-n** or **-t** must be specified.

- Example:

    ```bash
    PrecisionTool > ls -t Mul -n mul_3 -f TbeMulti
    ```

- Command output:

    ```text
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5b/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5c/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_5d/Branch_1/mul_3
    [Mul][TbeMultiOutputFusionPass] InceptionV3/InceptionV3/Mixed_6b/Branch_1/mul_3
    ```

## ni -n [op_name] -g [graph] -a [attr] -s [save subgraph depth]

- Description:

    Queries the operator node information based on the operator name.

  - **-n**: (optional) specifies the node name.
  - **-g**: (optional) specifies the graph name.
  - **-a**: (optional) displays the attribute information.
  - **-s**: (optional) saves a subgraph with the specified operator as the root and specified value as the depth.

    **Note**: Either **-n** or **-g** must be specified.

- Example:

    ```bash
    PrecisionTool >  ni -n gradients/InceptionV3/InceptionV3/Mixed_7a/Branch_0/Maximum_1_grad/GreaterEqual -s 3
    ```

- Command output:

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

- Description:

    Displays the information about a dump block and saves the information to a  **.txt**  file.

    **-n**: optional, name of the data file.

- Example:

    ```bash
    PrecisionTool > pt TransData.trans_TransData_1170.327.1619347786532995.input.0.npy
    ```

- Command output:

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

- Description:

    Compares the tensor data in the two NumPy files.

  - **-n**: (required) specifies the names of the two NumPy files to be compared.
  - **-p**: (optional) specifies the number of error data records to be output.
  - **-al/rl**: (optional)  **al**  indicates the absolute error, and  **rl**  indicates the relative error. The following are examples:

    ```python
    # Example 1:
    np.allclose(left, right, atol=al, rtol=rl)
    # Example 2:
    err_cnt += 1 if abs(data_left[i] - data_right[i]) > (al + rl * abs(data_right[i])) else 0
    ```

  - **-s**: (optional) saves the file as a .txt file, which is enabled by default.

- Example:

    ```bash
    PrecisionTool > cp -n file0.npy file1.npy -p 10 -al 0.002 -rl 0.005 -s
    ```

- Command output:

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

- Description:

    Displays the summary of the accuracy analysis result and optionally filters operators based on the specified cosine similarity threshold.

  - **-f (--file)**: (optional) specifies the CSV file. If not set, traverses all CSV files in the latest comparison directory under  **precision_data/temp/vector_compare/**.
  - **-c (--cos_sim)**: (optional) specifies the cosine similarity threshold, which defaults to  **0.98**.
  - **-l (--limit)**: (optional) specifies the top  _N_  results to output, which defaults to  **3**.

- Example:

    ```bash
    PrecisionTool > vcs -c 0.98 -l 2
    ```

- Command output:

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

- Description:

    Compares two networks.

  - **-lt**: (required) specifies the directory of the left network.
  - **-rt**: (required) specifies the directory of the right network.

    > [!NOTE]NOTE
    > The input directories must be the ones where dump data is stored, for example,  **precision_data/npu/debug_0/dump/20220217095546/3/ge_default_20220217095547_1/1/0/**.

  - **-g**: (optional) analyzes the mapping between graphs in NPU and TensorFlow comparison scenarios. (NPU and NPU comparison can be done based on the operator name.)

- Example:

    ```bash
    PrecisionTool > vc -lt /path/left -rt /path/right
    ```
