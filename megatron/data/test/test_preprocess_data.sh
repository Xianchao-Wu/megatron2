#!/bin/bash

# 2021 June 01 xianchao wu tested ok

#IMPL=cached
IMPL=mmap


prep=/workspace/megatron/megatron2/tools/preprocess_data.py

jsonfile="/workspace/megatron/megatron2/pretrained/bert_pretrain/small_data_line_jp.json"
vocabfile="/workspace/megatron/gpt2-japanese/ja-bpe.txt"
emojifn="/workspace/megatron/gpt2-japanese/emoji.json"

python -m pdb $prep \
       --input $jsonfile \
       --vocab $vocabfile \
       --emoji-file $emojifn \
       --dataset-impl ${IMPL} \
       --output-prefix test_samples_${IMPL} \
       --tokenizer-type GPT2BPETokenizerJp \
       --workers 1 \
       --log-interval 2
