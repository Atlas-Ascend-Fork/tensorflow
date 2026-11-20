# How Do I Enable Auto Cast for AI CPU Operators?

## Overview

The Auto Cast feature casts a data type to another one compatible with an AI CPU operator at model build time, which aims to prevent build failure when the AI CPU operator does not support a specific data type, ensuring the successful operator execution on the network.

The following figure is an example when the MatrixInverse operator's input  **x**  does not support float16.

**Figure  1**  Error message
![](../figures/aicpu_cast_error.png  "error-message")

In this scenario, it is advisable to enable Auto Cast. The Procedure below provides detailed instructions.

## Procedure

1. Enable Auto Cast.

    Modify the  **$\{INSTALL_DIR\}/lib64/plugin/opskernel/config/init.conf**  file and change the value of  **AutoCastMode**  to  **1**, as shown in the following.

    ```text
    ...
    AutoCastMode = 1
    ```

2. Modify the corresponding operator information library by inserting the cast rule into the operator to be modified. \(The built-in AI CPU operator information library is stored in the  **built-in/op_impl/aicpu/aicpu_kernel/config**  directory under the OPP installation directory.\)

    The following is the operator information library of the MatrixInverse operator whose input  **x**  does not support float16.

    ```text
    "MatrixInverse":{
            "input0":{
                "name":"x",
                "type":"DT_FLOAT,DT_DOUBLE,DT_COMPLEX128,DT_COMPLEX64"
            },
            "opInfo":{
                "computeCost":"100",
                "engine":"DNN_VM_AICPU",
                "flagAsync":"False",
                "flagPartial":"False",
                "formatAgnostic":"False",
                "opKernelLib":"TFKernel",
                "opsFlag":"OPS_FLAG_OPEN",
                "subTypeOfInferShape":"1"
            },
            "output0":{
                "name":"y",
                "type":"DT_FLOAT,DT_DOUBLE,DT_COMPLEX128,DT_COMPLEX64"
            }
        },
    ```

    To include support for float16, modify the code as follows:

    1. Add the target data type in the input description and a data type cast rule.

        Take the MatrixInverse operator as an example. To include the support for float16, add float16 to the list of the input's data types and add a Cast rule by inserting a Cast operator before MatrixInverse's input to cast float16 to float32.

        ```text
                "input0":{
                    "name":"x",
                    "type":"DT_FLOAT,DT_DOUBLE,DT_COMPLEX128,DT_COMPLEX64,DT_FLOAT16",
                    "srcAutoCastType":"DT_FLOAT16",
                    "dstAutoCastType":"DT_FLOAT"
                },
        ```

        - Add  **DT_FLOAT16**  to the  **type**  field. For details about the supported data types, see the Cast operator's definition in the operator information library.
        - Add  **srcAutoCastType**  to specify the source data type.
        - Set  **dstAutoCastType**  to specify the destination data type.

    2. Add the target data type in the output description and a data type cast rule.

        Take the MatrixInverse operator as an example. To include the support for float16, add float16 to the list of the output's data types and add a Cast rule by inserting a Cast operator after MatrixInverse's output to cast float32 to float16.

        ```text
                "output0":{
                    "name":"y",
                    "type":"DT_FLOAT,DT_DOUBLE,DT_COMPLEX128,DT_COMPLEX64,DT_FLOAT16",
                    "srcAutoCastType":"DT_FLOAT",
                    "dstAutoCastType":"DT_FLOAT16"
                }
        ```

        - Add  **DT_FLOAT16**  to the  **type**  field. For details about the supported data types, see the Cast operator's definition in the operator information library.
        - Add  **srcAutoCastType**  to specify the source data type.
        - Set  **dstAutoCastType**  to specify the destination data type.

    > [!CAUTION]NOTICE
    >
    > - To include data type support for multiple inputs and outputs of an operator, modify them one by one as described above.
    > - Owing to data type cast, the accuracy is compromised with the loss varying depending on the cast types.
