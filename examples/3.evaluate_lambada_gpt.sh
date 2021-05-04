
TASK="LAMBADA"

VALID_DATA=/workspace/megatron/ngc_models/gpt_evaluate/bflm/lambada_test.jsonl
VOCAB_FILE=/workspace/megatron/ngc_models/gpt2-vocab.json
MERGE_FILE=/workspace/megatron/ngc_models/gpt2-merges.txt
CHECKPOINT_PATH=/workspace/megatron/ngc_models/release_gpt_345m

COMMON_TASK_ARGS="--num-layers 24 --hidden-size 1024 \
	--num-attention-heads 16 --seq-length 1024 \
	--max-position-embeddings 1024 --fp16 --vocab-file $VOCAB_FILE"

python tasks/main.py --task $TASK $COMMON_TASK_ARGS \
	--valid-data $VALID_DATA --tokenizer-type GPT2BPETokenizer \
	--strict-lambada --merge-file $MERGE_FILE \
	--load $CHECKPOINT_PATH --micro-batch-size 8 \
	--checkpoint-activations --log-interval 10 \
	--no-load-optim --no-load-rng
