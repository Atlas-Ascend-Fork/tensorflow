# HCCL初始化网卡失败，HCCP返回错误ret\[-17\]

## 问题现象

HCCL初始化网卡失败，HCCP（Huawei Collective Communication Process）返回错误：ra rdev init failed, ret \[-17\]

![](../figures/rdev_init_failed_ret17.png)

## 原因分析

HCCL在初始化时会根据rank table中的Device IP初始化Device网卡。如果初始化使用的Device IP和实际网卡的IP不一致，HCCP会初始化网卡失败并返回错误码“-17”。

## 解决方法

1. 确认该Device的rank id，并在rank table中找到对应的device_ip配置，rankid获取方式如下：

    在用户态Host日志（需打开EVENT日志）中， grep关键字“Entry-HcomInit”，其identify中内容即为rankid。

2. 确认该server的Device IP是否配置正确，若出现rank table中device_ip配置和查询结果不一致的情况，请以查询结果为准，并修改对应rank table配置文件中的"device_ip"字段。

    使用hccn_tool可查看Device网卡信息。

    ```bash
    hccn_tool -i 0 -ip -g
    hccn_tool -i 1 -ip -g
    hccn_tool -i 2 -ip -g
    hccn_tool -i 3 -ip -g
    hccn_tool -i 4 -ip -g
    hccn_tool -i 5 -ip -g
    hccn_tool -i 6 -ip -g
    hccn_tool -i 7 -ip -g
    或
    for i in {0..7}; do hccn_tool -i $i -ip -g ; done
    ```
