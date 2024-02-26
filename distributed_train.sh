#!/bin/bash
NNODES=4
NUM_PROC=4
MASTER_IP=10.28.1.29
MASTER_PORT=11234

shift
OMP_NUM_THREADS=20 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    train.py "$@" \
    > train_imagenet.log 2>&1

