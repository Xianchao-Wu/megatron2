
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data.json"
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line1.json" # TODO work
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line2.json" # TODO

#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line3.json"
#prefix="my-bert"

#jsonfile="/workspace/megatron/datasets/english-fsi/temp.testin.json"
jsonfile="/workspace/megatron/datasets/english-fsi/eight.files.json"
vocabfile="/workspace/megatron/ngc_models/bert-large-cased-vocab.txt"
prefix="fsi-en-bert-8files-bert-large-cased-vocab-bwplc"

python tools/preprocess_data.py \
	       --input $jsonfile \
	       --output-prefix $prefix \
	       --vocab $vocabfile \
	       --dataset-impl mmap \
	       --tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent
