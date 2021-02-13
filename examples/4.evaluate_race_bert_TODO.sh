DIR="/workspace/megatron/ngc_models/bert_evaluate/RACE"
TRAIN_DATA=$DIR+"/train/middle"
VALID_DATA=$DIR+"/dev/middle" + " " + $DIR + "/dev/high"

VOCAB_FILE="/workspace/megatron/ngc_models/bert-large-cased-vocab.txt"

# TODO
PRETRAINED_CHECKPOINT=checkpoints/bert_345m
#CHECKPOINT_PATH=checkpoints/bert_345m_race # TODO no checkpoint yet!
CHECKPOINT_PATH=/workspace/megatron/ngc_models/bert_evaluate/checkpoints/bert_345m_race

COMMON_TASK_ARGS="--num-layers 24 \
	          --hidden-size 1024 \
		  --num-attention-heads 16 \
		  --seq-length 512 \
		  --max-position-embeddings 512 \
		  --fp16 \
		  --vocab-file $VOCAB_FILE"

COMMON_TASK_ARGS_EXT="--train-data $TRAIN_DATA \
	              --valid-data $VALID_DATA \
		      --pretrained-checkpoint $PRETRAINED_CHECKPOINT \
		      --checkpoint-activations \
		      --save-interval 10000 \
		      --save $CHECKPOINT_PATH \
		      --log-interval 100 \
		      --eval-interval 1000 \
		      --eval-iters 10 \
		      --weight-decay 1.0e-1"

python tasks/main.py \
	       --task RACE \
	       $COMMON_TASK_ARGS \
	       $COMMON_TASK_ARGS_EXT \
	       --tokenizer-type BertWordPieceLowerCase \
	       --epochs 3 \
	       --micro-batch-size 4 \
	       --lr 1.0e-5 \
	       --lr-warmup-fraction 0.06
	       
