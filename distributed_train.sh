#!/bin/bash
NNODES=4
NUM_PROC=4
MASTER_IP=10.28.1.29
MASTER_PORT=11234

OMP_NUM_THREADS=20 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    train.py "$@" 

