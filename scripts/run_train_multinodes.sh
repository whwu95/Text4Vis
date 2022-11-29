#!/usr/bin/env bash

#--- Multi-nodes training hyperparams ---
nnodes=2
master_addr="10.67.212.21"

# Note:
# 0. You need to set the master ip address according to your own machines.
# 1. You'd better to scale the learning rate when you use more gpus.
# 2. Command: sh scripts/run_train_multinodes.sh node_rank
############################################# 
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

python -m torch.distributed.launch --master_port 12355 --nproc_per_node=8 \
            --nnodes=${nnodes} --node_rank=$2  \
            --master_addr=${master_addr} \
            train.py  --config ${config}  ${@:3}
