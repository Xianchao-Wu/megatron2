
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data.json"
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line1.json" # TODO work
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line2.json" # TODO

#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line3.json"
#prefix="my-bert"

#jsonfile="/workspace/megatron/datasets/english-fsi/temp.testin.json"
# 在tools.preprocess_data_win10_enfsi.py中，使用具体的本地化文件，default 参数赋值来实现：

jsonfile="/workspace/megatron/datasets/english-fsi/eight.files3.json"
vocabfile="/workspace/megatron/ngc_models/bert-large-cased-vocab.txt"
prefix="fsi-en-bert-8files-bert-large-cased-vocab-bwplc-small3"

python -m ipdb tools/preprocess_data.py \
	       --input $jsonfile \
	       --output-prefix $prefix \
	       --vocab $vocabfile \
	       --dataset-impl mmap \
	       --tokenizer-type BertWordPieceCase #\
	       #--tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent
