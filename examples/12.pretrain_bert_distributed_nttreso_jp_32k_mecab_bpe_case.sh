#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6003
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#DATA_PATH=<Specify path and file prefix>_text_sentence
#CHECKPOINT_PATH=<Specify path>


#DATA_PATH=/workspace/megatron/megatron2/fsi-en-bert-8files-bert-large-cased-vocab-bwplc_text_sentence
DATA_PATH=/workspace/megatron/megatron2/nttreso-ja-bert-vocab-32k-mecab-bpe-case_text_sentence

CHECKPOINT_PATH_IN=/workspace/megatron/ngc_models/release_bert_345m_cased_nttreso_jp_32k_mecab_bpe
CHECKPOINT_PATH_OUT=/workspace/megatron/ngc_models/release_bert_345m_cased_nttreso_jp_32k_mecab_bpe

vocabfn=/workspace/megatron/datasets/nttreso_qa/export/export_readable_20210727_simp_2read_v3.mecab.txt.vocab.32000.v3.bpe

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# NOTE use BertWordPieceCaseJp
# the old BertWordPieceJp was used for data processing and it was okay

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 10000000 \
       --save $CHECKPOINT_PATH_OUT \
       --load $CHECKPOINT_PATH_IN \
       --data-path $DATA_PATH \
       --vocab-file $vocabfn \
	   --tokenizer-type BertWordPieceCaseJp \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1000 \
       --save-interval 50000 \
       --eval-interval 5000 \
       --eval-iters 10 \
       --fp16
