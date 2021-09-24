#########################################################################
# File Name: rmbigfile2.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Sep 12 09:43:02 2021
#########################################################################
#!/bin/bash

git filter-branch --tree-filter 'rm -rf examples/12.pretrain_bert_distributed_nttreso_jp_32k_mecab_bpe_case.sh.log' HEAD
