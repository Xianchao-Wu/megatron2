
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

adir="/workspace/megatron/datasets/ch.finance.news.simp/"
#jsonfile=$adir"/12files.all.su.txt.10000lines.json"
jsonfile=$adir"/12files.all.su.txt.json"

vocabfile=$adir"/12files.all.su.txt.jieba.vocab.32000.v3.bpe"

prefix="fsi-ch-bert-vocab-32k-jieba-bpe-case-worker1"

# mecab+ipa -> bpe/wordpiece

python tools/preprocess_data.py \
	       --input $jsonfile \
	       --output-prefix $prefix \
	       --vocab $vocabfile \
	       --dataset-impl mmap \
		   --workers 1 \
	       --tokenizer-type BertWordPieceCaseCh #\
	       #--tokenizer-type BertWordPieceJp #\
	       #--tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent
