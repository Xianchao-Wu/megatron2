#########################################################################
# File Name: runbuildvocab.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2021年02月14日 23時19分22秒
#########################################################################
#!/bin/bash

# NOTE IMPORTANT:
# run this bash at the directory of the parent folder of "examples"
# e.g., bash examples/7.runbuildvocab_fsi_jp.sh > examples/7.runbuildvocab_fsi_jp.sh.log 2>&1 &

#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/MainichiNews.t0.txt.gz.simp.txt"
#infile="/raid/xianchaow/megatron/datasets/japanese-fsi/jp.fsi.8files.txt"
#infile="/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.txt"

# NOTE do not work
# should not in this docker, should be in dgx-1 and conda activate pytorch
# even "pip list" does not include "MeCab", but import MeCab still works in "pytorch" env! TODO

# please use: 
# runbuildvocab_nttreso_jp3_mecabipa.sh
# (pytorch) xianchaow@dgx-1:/raid/xianchaow/megatron/bert-japanese$


infile="/workspace/megatron/datasets/nttreso_qa/export/export_readable_20210727_simp_2read_v3.txt"

outfile=$infile".vocab.50k"
python megatron/tokenizer/build_vocab_bert_jp.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 50000 

outfile=$infile".vocab.40k"
python megatron/tokenizer/build_vocab_bert_jp.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 40000 

outfile=$infile".vocab.30k"
python megatron/tokenizer/build_vocab_bert_jp.py --input_file $infile --output_file $outfile --subword_type bpe --vocab_size 30000 
