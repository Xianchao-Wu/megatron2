#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

#DATA_PATH=<Specify path and file prefix>_text_document
#CHECKPOINT_PATH=<Specify path>

DATA_PATH="/workspace/megatron/megatron2/fsi-ja-gpt2-12files-vocab-20k-bpedict_text_sentence"
CHECKPOINT_PATH="/workspace/megatron/ngc_models/release_gpt2_345m_uncased_fsijp"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

vocabfile="/workspace/megatron/megatron2/pretrained/gpt2_ja_bpe.txt"
emojifile="/workspace/megatron/megatron2/pretrained/gpt2_ja_emoji.json"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 8 \
       --global-batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1000000 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $vocabfile \
       --emoji-file $emojifile \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --lr-decay-style cosine \
       --min-lr 1.0e-5 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --checkpoint-activations \
       --log-interval 1000 \
       --save-interval 20000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16
