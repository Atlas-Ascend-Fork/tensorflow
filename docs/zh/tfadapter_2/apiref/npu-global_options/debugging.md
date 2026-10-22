# 功能调试

## op_debug_config

Global Memory内存检测功能开关。

取值为.cfg配置文件路径，配置文件内多个选项用英文逗号分隔：

- oom：在算子执行过程中，检测Global Memory是否内存越界。

  算子编译时会在当前执行路径下的kernel_meta文件夹中保留.o（算子二进制文件）和.json文件（算子描述文件），并加入如下检测逻辑：

  ```c
  inline __aicore__ void  CheckInvalidAccessOfDDR(xxx) {
      if (access_offset < 0 || access_offset + access_extent > ddr_size) {
          if (read_or_write == 1) {
              trap(0X5A5A0001);
          } else {
              trap(0X5A5A0002);
          }
      }
  }
  ```

  用户可配合使用**dump_cce**参数，在生成的.cce文件中查看上述代码。

  编译过程中，若存在内存越界，会抛出“**EZ9999**”错误码。

- dump_cce：算子编译时，在当前执行路径下的kernel_meta文件夹中保留算子的cce文件\*.cce，算子二进制文件\*.o，以及算子描述文件\*.json。
- dump_loc：算子编译时，在当前执行路径下的kernel_meta文件夹中保留算子的cce文件\*.cce，算子二进制文件\*.o，算子描述文件\*.json，以及python-cce映射文件\*_loc.json。
- ccec_O0：算子编译时，开启ccec编译器的默认编译选项-O0，此编译选项针对调试信息不会执行任何优化操作。
- ccec_g ：算子编译时，开启ccec编译器的编译选项-g，此编译选项相对于-O0，会生成优化调试信息。
- check_flag：算子执行时，检测算子内部流水线同步信号是否匹配。

  算子编译时，在生成的kernel_meta文件夹中保留.o（算子二进制文件）和.json文件（算子描述文件），并加入如下检测逻辑：

  ```text
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    ....
    pipe_barrier(PIPE_MTE3);
    pipe_barrier(PIPE_MTE2);
    pipe_barrier(PIPE_M);
    pipe_barrier(PIPE_V);
    pipe_barrier(PIPE_MTE1);
    pipe_barrier(PIPE_ALL);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID3);
    ...
  ```

  用户可配合使用**dump_cce**参数，在生成的.cce文件中查看上述代码。

  编译过程中，若存在算子内部流水线同步信号不匹配的情况，会在**有问题的算子处超时报错**，报错信息示例为：

  ```text
  Aicore kernel execute failed, ..., fault kernel_name=算子名,...
  rtStreamSynchronizeWithTimeout execute failed....
  ```

配置示例：

```python
npu.global_options().op_debug_config="/root/test0.cfg"
```

其中，test0.cfg文件信息为：

```text
op_debug_config=ccec_g,oom
```

**使用约束**：

算子编译时，如果用户不想编译所有AI Core算子，而是指定某些AI Core算子进行编译，则需要在上述test0.cfg配置文件中新增**op_debug_list**字段，算子编译时，只编译该列表指定的算子，并按照op_debug_config配置的选项进行编译。op_debug_list字段要求如下：

- 支持指定算子名称或者算子类型。
- 算子之间使用英文逗号分隔，若为算子类型，则以“OpType::typeName”格式进行配置，支持算子类型和算子名称混合配置。
- 要编译的算子，必须放在op_debug_config参数指定的配置文件中。

test0.cfg文件配置示例如下：

```text
op_debug_config= ccec_g,oom
op_debug_list=GatherV2,opType::ReduceSum
```

模型编译时，GatherV2、ReduceSum算子按照ccec_g,oom选项进行编译。

开启ccec编译选项的场景下（即ccec_O0、ccec_g选项），会增大算子Kernel（\*.o文件）的大小。动态shape场景下，由于算子编译时会遍历可能存在的所有场景，最终可能会导致由于算子Kernel文件过大而无法进行编译的情况，此种场景下，建议不要开启ccec编译选项。
由于算子kernel文件过大而无法编译的日志显示如下：

```text
message:link error ld.lld: error: InputSection too large for range extension thunk ./kernel_meta_xxxxx.o:(xxxx)
```

ccec编译选项ccec_O0和oom选项不可同时开启，会导致AI Core Error报错，报错信息示例如下：

```text
...there is an aivec error exception, core id is 49, error code = 0x4 ...
```

- 此参数取值为dump_cce、dump_loc时，可通过“debug_dir”参数指定调试相关过程文件的存放路径。
- 配置编译选项oom、dump_cce、dump_loc时，若模型中含有如下通算融合算子，算子编译目录kernel_meta中，不会生成下述算子的\*.o、\*.json、\*.cce文件。
  
  ```text
  MatMulAllReduce
  MatMulAllReduceAddRmsNorm
  AllGatherMatMul
  MatMulReduceScatter
  AlltoAllAllGatherBatchMatMul
  BatchMatMulReduceScatterAlltoAll
  ```

- 若配置了NPU_COLLECT_PATH环境变量，不支持打开“检测Global Memory是否内存越界”的开关，即不支持将此参数指定的配置文件中配置“oom”，否则编译出来的模型文件或算子kernel包在使用时会报错。

## enable_exception_dump

是否dump异常算子数据。

- 0：关闭异常算子数据dump功能。
- 1：开启普通ExceptionDump，dump异常算子的输入输出数据、tensor描述信息（shape、dtype、format等）以及workspace信息。

    dump数据存储路径优先级为：环境变量NPU_COLLECT_PATH \> 环境变量ASCEND_WORK_PATH \> 默认路径（当前脚本执行路径下的extra-info目录）。

- 2（默认值）：开启LiteExceptionDump，dump异常算子的输入输出数据、workspace信息、Tiling信息等，导出的数据用于分析AI Core Error问题（关于AI Core Error问题的信息收集及定位，详细说明请参见《[故障处理](https://hiascend.com/document/redirect/CannCommunitytrouble)》手册中的“典型故障专题 \> AI Core Error问题定位专题”）。

    dump数据存储路径优先级为：环境变量ASCEND_WORK_PATH \> 默认路径（指当前脚本执行路径下的extra-info/data-dump/<device_id\>目录）。

> [!NOTE]说明
>若配置了环境变量NPU_COLLECT_PATH，不论配置项“enable_exception_dump”的取值如何，都按照“1：普通ExceptionDump”进行异常算子数据dump，且dump数据存储在环境变量NPU_COLLECT_PATH的指定目录下。
> 关于环境变量ASCEND_WORK_PATH的详细说明可参见《[环境变量参考](https://hiascend.com/document/redirect/CannCommunityEnvRef)》

配置示例：

```python
npu.global_options().enable_exception_dump=1
```

## debug_dir

用于配置保存算子编译生成的调试相关的过程文件的路径，包括算子.o/.json/.cce等文件。

算子编译生成的调试文件存储优先级为：

配置参数“debug_dir” \> 环境变量ASCEND_WORK_PATH \> 默认存储路径（当前脚本执行路径）。

配置示例：

```python
npu.global_options().debug_dir="/home/test"
```

## export_compile_stat

用户配置图编译过程中是否生成算子融合信息的结果文件fusion_result.json。

支持如下取值：

- 0：不生成算子融合信息结果文件。
- 1（默认值）：程序运行正常退出时生成算子融合信息结果文件。
- 2：图编译完成时即生成算子融合信息结果文件，即如果图编译已完成，后续程序中断，也会生成算子融合信息结果文件。

fusion_result.json文件用于记录图编译过程中使用的融合规则，文件中关键字段含义如下：

- session_and_graph_id__xx_xx_：表示融合结果所属线程和图编号。
- graph_fusion：表示图融合。
- ub_fusion：表示UB融合

    **Ascend 950PR&950DT系列产品不支持UB融合，不会生成该信息**。

- match_times：表示图编译过程中匹配到的融合规则次数。
- effect_times：表示实际生效的次数。
- repository_hit_times：优化UB融合知识库命中的次数

    **Ascend 950PR&950DT系列产品不支持UB融合，不会生成该信息**。

> [!NOTE]说明
>
>- 若环境中未配置环境变量ASCEND_WORK_PATH，算子融合信息结果保存至当前执行目录的fusion_result.json文件；若环境中配置了环境变量ASCEND_WORK_PATH，则保存至`$ASCEND_WORK_PATH/FE/${进程号}/fusion_result.json`文件。
>- 通过“fusion_switch_file”参数关闭的融合规则不会在fusion_result.json中呈现。

配置示例：

```python
npu.global_options().export_compile_stat=1
```
