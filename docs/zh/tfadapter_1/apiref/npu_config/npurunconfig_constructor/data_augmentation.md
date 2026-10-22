# 数据增强

## local_rank_id

该参数用于推荐网络场景的数据并行场景，在主进程中对数据进行去重操作，去重之后的数据再分发给其他进程的Device进行前后向计算。

![](../../figures/local_rank_id.png)

该模式下，一个主机上多Device共用一个进程做数据预处理，但实际还是多进程的场景，在主进程上进行数据预处理，其他进程不再接收本进程上的Dataset，而是接收主进程预处理后的数据。

具体使用方法一般是通过集合通信的get_local_rank_id\(\)接口获取当前进程在其所在Server内的rank编号，用来判断哪个进程是主进程。

配置示例：

```python
config = NPURunConfig(local_rank_id=0, local_device_list="0,1")
```

## local_device_list

该参数配合local_rank_id使用，用来指定主进程给哪些其他进程的Device发送数据。

```python
config = NPURunConfig(local_rank_id=0, local_device_list="0,1")
```
