#!/bin/bash

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#DATA_PATH=<Specify path and file prefix>_text_sentence
#CHECKPOINT_PATH=<Specify path>


#DATA_PATH=<Specify path and file prefix>_text_sentence
DATA_PATH=/workspace/megatron/megatron2/fsi-en-bert-8files-bert-large-cased-vocab-bwplc-small_text_sentence
#CHECKPOINT_PATH=<Specify path>
CHECKPOINT_PATH_IN=/workspace/megatron/ngc_models/release_bert_345m_uncased
CHECKPOINT_PATH_OUT=/workspace/megatron/ngc_models/release_bert_345m_uncased_small

vocabfn=/workspace/megatron/ngc_models/bert-large-cased-vocab.txt

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 1000 \
       --save $CHECKPOINT_PATH_OUT \
       --load $CHECKPOINT_PATH_IN \
       --data-path $DATA_PATH \
       --vocab-file $vocabfn \
       --tokenizer-type BertWordPieceCase \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 99 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 500 \
       --eval-interval 500 \
       --eval-iters 10 \
       --fp16
