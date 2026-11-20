# What Do I Do If the HCCL Fails to Initialize the NIC and the HCCP Returns Error Code ret\[-17\]?

## Symptom

The HCCL fails to initialize the NIC, and the Huawei Collective Communication Process \(HCCP\) returns the error code  **ra rdev init failed, ret \[–17\]**.

![](../figures/rdev_init_failed_ret17.png)

## Possible Cause

During initialization, the HCCL initializes the device NIC based on the device IP address in the rank table. If the device IP address used for initialization is different from the actual NIC IP address, the HCCP fails to initialize the NIC and returns error code  **-17**.

## Solution

1. Obtain the rank ID of the device and its  **device_ip**  configuration in the rank table:

    In the user-mode host log \(EVENT-level logging needs to be enabled\), grep for the keyword  **Entry-HcomInit**. The content in  **identify**  is the rank ID.

2. Check the device IP address of the server. If the value of  **device_ip**  in the rank table is different from that in the query result, change the value of  **device_ip**  in the rank table to the query result.

    You can run the  **hccn_tool**  command to view the device NIC information.

    ```bash
    hccn_tool -i 0 -ip -g
    hccn_tool -i 1 -ip -g
    hccn_tool -i 2 -ip -g
    hccn_tool -i 3 -ip -g
    hccn_tool -i 4 -ip -g
    hccn_tool -i 5 -ip -g
    hccn_tool -i 6 -ip -g
    hccn_tool -i 7 -ip -g
    Or
    for i in {0..7}; do hccn_tool -i $i -ip -g ; done
    ```
