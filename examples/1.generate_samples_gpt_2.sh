#pt=/workspace/megatron/ngc_models/megatron_lm_345m_v0.0.zip
pt=/workspace/megatron/ngc_models/release_gpt_345m
outjson=gpt3-gen-examples.json

#python tools/generate_samples_gpt.py --tensor-model-parallel-size 1 --num-layers 24 --hidden-size 1024 --load /workspace/megatron/ngc_models/release_gpt_345m/mp_rank_00/model_optim_rng.pt --num-attention-heads 16 --max-position-embeddings 1024 --tokenizer-type GPT2BPETokenizer --fp16 --micro-batch-size 2 --seq-length 1024 --out-seq-length 1024 --temperature 1.0 --vocab-file /workspace/megatron/ngc_models/gpt2-vocab.json --merge-file /workspace/megatron/ngc_models/gpt2-merges.txt --genfile x.json --num-samples 2 --top_p 0.9 --recompute

python tools/generate_samples_gpt.py --tensor-model-parallel-size 1 --num-layers 24 --hidden-size 1024 --load $pt --num-attention-heads 16 --max-position-embeddings 1024 --tokenizer-type GPT2BPETokenizer --fp16 --micro-batch-size 2 --seq-length 1024 --out-seq-length 1024 --temperature 1.0 --vocab-file /workspace/megatron/ngc_models/gpt2-vocab.json --merge-file /workspace/megatron/ngc_models/gpt2-merges.txt --genfile $outjson --num-samples 2 --top_p 0.9 --recompute
