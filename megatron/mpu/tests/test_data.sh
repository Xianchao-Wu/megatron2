#########################################################################
# File Name: test_data.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun  1 12:59:29 2021
#########################################################################
#!/bin/bash

# multi-gpu (single-node) run test_data.py

GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6000

args="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# failed...
python -m pdb -m torch.distributed.launch $args test_data.py

