# CM_CHIEF_PORT

## Description

In TensorFlow distributed training or inference scenarios, you can choose not to use the rank table file. By jointly configuring environment variables [CM_CHIEF_IP](CM_CHIEF_IP.md), [CM_CHIEF_PORT](CM_CHIEF_PORT.md), [CM_CHIEF_DEVICE](CM_CHIEF_DEVICE.md), [CM_WORKER_SIZE](CM_WORKER_SIZE.md), and [CM_WORKER_IP](CM_WORKER_IP.md), resource information can be automatically generated to complete the initialization of the collective communication component.

Configures the listening port of the master node.

The value of this environment variable must be an integer ranging from 0 to 65520. Ensure that the port is not occupied by other processes.

## Example

```bash
export CM_CHIEF_PORT=6000
```

## Constraints

This environment variable cannot be used together with [RANK_TABLE_FILE](RANK_TABLE_FILE.md), [RANK_ID](RANK_ID.md) or [RANK_SIZE](RANK_SIZE.md).

## Applicability

Atlas training products

Atlas A2 products
