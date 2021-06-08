#########################################################################
# File Name: test_cross_entropy.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Wed Jun  2 07:04:13 2021
#########################################################################
#!/bin/bash

# 1. single-gpu
#python -m pdb test_cross_entropy.py
python test_cross_entropy.py


# 2. multi-gpu
#python -m torch.distributed.launch --nproc_per_node 8 test_cross_entropy.py

