#########################################################################
# File Name: test_mecab.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sun Aug  8 09:10:16 2021
#########################################################################
#!/bin/bash

apath="/workspace/megatron/datasets/nttreso_qa/export"
afile=$apath"/export_readable_20210727_simp_2read_v3.txt.gz"

zcat $afile | python test_mecab.py > $afile.unidic_lite.txt 2>$afile.unidic_lite.txt.log
