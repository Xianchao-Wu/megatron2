#########################################################################
# File Name: runbuildvocab.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2021年02月14日 23時19分22秒
#########################################################################
#!/bin/bash

# NOTE IMPORTANT:
# run this bash at the directory of the parent folder of "examples"
# e.g., bash examples/8.runbuildvocab_cc100_jp.sh > examples/8.runbuildvocab_cc100_jp.sh.log 2>&1 &

#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/MainichiNews.t0.txt.gz.simp.txt"
#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/jp.fsi.8files.txt"
infile="/workspace/megatron/datasets/japanese-cc-100/cc-100-ja.txt"
outfile=$infile".vocab"

#python build_vocab.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 40000 
python megatron/tokenizer/build_vocab.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 50000 
