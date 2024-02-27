#!/bin/bash
NNODES=4
NUM_PROC=4
MASTER_IP=10.28.1.29
MASTER_PORT=11234
export NCCL_SOCKET_IFNAME=enp225s0

# get the IP address of the current node
IP=`hostname -I | grep -o "10\.28\.1\.[0-9]\+"`
if [ "$IP" = "$MASTER_IP" ]; then
    IS_MASTER=true
    echo "This is the master node"
else
    IS_MASTER=false
    echo "This is a worker node"
fi

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=20 NCCL_DEBUG=INFO NCCL_IB_DISABLE=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_PROC \
    --rdzv_id=1234 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_IP:$MASTER_PORT \
    --rdzv_conf is_host=$IS_MASTER \
    train.py "$@" 
    > train_imagenet_with_undo.log 2>&1 

