
PATH=/workspace/megatron/ngc_models
CHECKPOINT_PATH=$PATH/release_gpt_345m/mp_rank_00/model_optim_rng.pt
VOCAB_FILE=$PATH/gpt2-vocab.json
MERGE_FILE=$PATH/gpt2-merges.txt

python tools/generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 1024 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 16 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 2 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --genfile unconditional_samples.json \
       --num-samples 2 \
       --top_p 0.9 \
       --recompute
