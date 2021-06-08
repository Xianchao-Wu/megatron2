
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
#jsonfile="/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.line100.json"
#prefix="fsi-ja-gpt2-12files-vocab-20k-line100"

jsonfile="/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.json"
prefix="fsi-ja-gpt2-12files-vocab-20k-bpedict-mecab"

#vocabfile='/workspace/megatron/megatron2/pretrained/gpt2_ja_bpe.txt'
vocabfile='/workspace/megatron/datasets/japanese-fsi/jp.fsi.8files.txt.vocab'
emojifile='/workspace/megatron/megatron2/pretrained/gpt2_ja_emoji.json'

python tools/preprocess_data.py --input $jsonfile \
	       --output-prefix $prefix \
	       --vocab-file $vocabfile \
	       --emoji-file $emojifile \
	       --dataset-impl mmap \
	       --tokenizer-type GPT2BPETokenizerJpMecab 
	       #BertWordPieceJp #\
	       #--tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent

#python tools/preprocess_data.py \
#	       --input $jsonfile \ # Yes
#	       --output-prefix $prefix \ # Yes, --output-prefix
#	       --vocab-file $vocabfile \ # TODO, --vocab-file
#	       --emoji-file $emojifile \ # --emoji-file
#	       --dataset-impl mmap \
#	       --tokenizer-type GPT2BPETokenizerJp #BertWordPieceJp #\
	       #--tokenizer-type BertWordPieceLowerCase #\
	       #--split-sentences # TODO no need anymore, default is perform split sent
