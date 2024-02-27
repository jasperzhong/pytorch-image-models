#!/bin/bash
NNODES=2
NUM_PROC=4
MASTER_IP=10.28.1.29
MASTER_PORT=11234
export NCCL_SOCKET_IFNAME=enp225s0

# get the IP address of the current node
IP=`hostname -I | awk '{print $1}'`
IS_MASTER=$([ "$IP" = "$MASTER_IP" ] && echo "is master")

OMP_NUM_THREADS=20 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    --is_host=$IS_MASTER \
    train.py "$@" 
    > train_imagenet.log 2>&1 

