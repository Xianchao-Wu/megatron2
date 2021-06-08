#########################################################################
# File Name: test_indexed_dataset.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Jun  1 05:33:34 2021
#########################################################################
#!/bin/bash

vocabfile="/workspace/megatron/gpt2-japanese/ja-bpe.txt"
emojifn="/workspace/megatron/gpt2-japanese/emoji.json"

python -m pdb test_indexed_dataset.py \
	--data test_samples_mmap_text_sentence \
	--dataset-impl mmap \
	--tokenizer-type GPT2BPETokenizerJp \
	--vocab $vocabfile \
    --emoji-file $emojifn

