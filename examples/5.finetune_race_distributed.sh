#!/bin/bash

WORLD_SIZE=8

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

TRAIN_DATA="/workspace/megatron/ngc_models/bert_evaluate/RACE/train/middle"
VALID_DATA="/workspace/megatron/ngc_models/bert_evaluate/RACE/dev/middle \
            /workspace/megatron/ngc_models/bert_evaluate/RACE/dev/high"

#VOCAB_FILE=/workspace/megatron/ngc_models/bert-large-uncased-vocab.txt
#PRETRAINED_CHECKPOINT=/workspace/megatron/ngc_models/release_bert_345m_uncased

VOCAB_FILE=/workspace/megatron/ngc_models/bert-large-cased-vocab.txt
PRETRAINED_CHECKPOINT=/workspace/megatron/ngc_models/release_bert_345m_cased

CHECKPOINT_PATH=/workspace/megatron/ngc_models/bert_evaluate/checkpoints/bert_345m_race

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task RACE \
               --seed 1234 \
               --train-data $TRAIN_DATA \
               --valid-data $VALID_DATA \
               --tokenizer-type BertWordPieceLowerCase \
               --vocab-file $VOCAB_FILE \
               --epochs 3 \
               --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
               --tensor-model-parallel-size 1 \
               --num-layers 24 \
               --hidden-size 1024 \
               --num-attention-heads 16 \
               --micro-batch-size 4 \
               --checkpoint-activations \
               --lr 1.0e-5 \
               --lr-decay-style linear \
               --lr-warmup-fraction 0.06 \
               --seq-length 512 \
               --max-position-embeddings 512 \
               --save-interval 100000 \
               --save $CHECKPOINT_PATH \
               --log-interval 10 \
               --eval-interval 100 \
               --eval-iters 50 \
               --weight-decay 1.0e-1 \
               --clip-grad 1.0 \
               --hidden-dropout 0.1 \
               --attention-dropout 0.1 \
               --fp16
