# Sample Running

This section uses the single-server 8-device networking and ranktable file as an example to describe how to run the sample code in  [Code Example](code_example.md).

## Procedure

1. Prepare the ranktable file.

    For details about the configuration examples and parameter description of the ranktable file for different product forms, see  "Reference" \> "Cluster Information Configuration"  in the  _[HCCL User Guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/latest/API/hcclug/hcclug_000001.html)_.

2. Construct the startup script.

    The following uses the script  **hccl_start_8p.sh**  as an example:

    ```bash
    # Configure environment variables of the CANN software. The default installation path of the root user is used as an example.
    source /usr/local/Ascend/cann/set_env.sh

    # TF Adapter Python library. ${TFPLUGIN_INSTALL_PATH} indicates the installation path of the TF Adapter package.
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH

    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/ranktable.json    # Path of the ranktable resource configuration file. Replace it with the actual path.
    export JOB_ID=10087      # User-defined task ID, which can contain uppercase letters, lowercase letters, digits, hyphens (-), and underscores (_).

    for((RANK_ID=0;RANK_ID<$((RANK_SIZE));RANK_ID++));
    do
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID
        # Execute the script. Replace the script path and name as required.
        nohup python3 /home/test/hccl_test.py &
    done
    ```

3. Execute the startup script.

    ```bash
    bash hccl_start_8p.sh
    ```

    Result example:

    ```text
    ... ...
    'reduce_sum': array([[ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           ...,
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ..., 44, 44, 44]]), 'reduce_max': array([[4097, 4098, 4099, ..., 4222, 4223, 4224],
           [4225, 4226, 4227, ..., 4350, 4351, 4352],
           [4353, 4354, 4355, ..., 4478, 4479, 4480],
           ...,
           [4737, 4738, 4739, ..., 4862, 4863, 4864],
           [4865, 4866, 4867, ..., 4990, 4991, 4992],
           [4993, 4994, 4995, ...,    9,    9,    9]]), 'reduce_min': array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 2, 2, 2]]), 'reduce_prod': array([[     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           ...,
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ..., 362880, 362880, 362880]]), 'alltoallv_tensor': array([   1,    2,    3, ..., 8246, 8247, 8248]), 'check_tensors': array([   1,    2,    3, ..., 8246, 8247, 8248])
    train success
    ```
