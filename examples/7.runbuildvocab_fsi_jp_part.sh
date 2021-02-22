#########################################################################
# File Name: runbuildvocab.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2021年02月14日 23時19分22秒
#########################################################################
#!/bin/bash

#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/MainichiNews.t0.txt.gz.simp.txt"
#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/jp.fsi.8files.txt"
infile="/workspace/megatron/datasets/japanese-fsi/MainichiNews.t0.txt.gz.simp.txt"
outfile=$infile".vocab.50k"

#python build_vocab.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 40000 
python megatron/tokenizer/build_vocab.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 50000 
