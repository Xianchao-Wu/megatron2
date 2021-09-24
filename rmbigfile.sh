#########################################################################
# File Name: rmbigfile.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Sep 12 09:40:12 2021
#########################################################################
#!/bin/bash

git filter-branch --force --index-filter 'git rm -rf --cached --ignore-unmatch examples/13.pretrain_bert_distributed_fsi_ch_32k_jieba_bpe_case.sh.log' --prune-empty --tag-name-filter cat -- --all
