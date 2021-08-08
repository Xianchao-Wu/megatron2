
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data.json"
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line1.json" # TODO work
#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line2.json" # TODO

#jsonfile="/workspace/megatron/ngc_models/bert_pretrain/small_data_line3.json"
#prefix="my-bert"

#jsonfile="/workspace/megatron/datasets/english-fsi/temp.testin.json"
#jsonfile="/workspace/megatron/datasets/english-fsi/eight.files.json"
#vocabfile="/workspace/megatron/ngc_models/bert-large-cased-vocab.txt"
#prefix="fsi-en-bert-8files-bert-large-cased-vocab-bwplc"

# TODO actually 8files are actually 12 files: mainichi, nikkei, and sankei
#jsonfile="/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.json"
#vocabfile="/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.txt.vocab"
#prefix="fsi-ja-bert-12files-bert-large-vocab-bwplc-debugonly"

adir="/workspace/megatron/datasets/nttreso_qa/export/"
jsonfile=$adir"/export_readable_20210727_simp_2read_v2.json"

vocabfile=$adir"/export_readable_20210727_simp_2read_v3.mecab.txt.vocab.64000.v3.bpe"

prefix="nttreso-ja-bert-vocab-64k-mecab-bpe-case"

# mecab+ipa -> bpe/wordpiece

python tools/preprocess_data.py \
	       --input $jsonfile \
	       --output-prefix $prefix \
	       --vocab $vocabfile \
	       --dataset-impl mmap \
	       --tokenizer-type BertWordPieceJp #\
	       #--tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent
