# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars): # TODO 可以为日语也定制一下

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args) # 重要：第一个是tokenizer，负责类似分词的工作
        if self.args.split_sentences: # 重要：第二个是负责句子切割的
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if self.args.tokenizer_type == "BertWordPieceJp" or self.args.tokenizer_type.startswith('GPT2BPETokenizerJp'):
                # TODO for japanese langugae's sentence split: 日文段落的分句：
                #from nltk.tokenize import RegexpTokenizer
                jp_sent_splitter = nltk.RegexpTokenizer(u'[^！？。]*[！？。]')
                Encoder.splitter = jp_sent_splitter
            else:
                splitter = nltk.load("tokenizers/punkt/english.pickle") # TODO other languages?
                if self.args.keep_newlines:
                    # this prevents punkt from eating newlines after sentences
                    Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                        train_text = splitter._params,
                        lang_vars = CustomLanguageVars())
                else:
                    Encoder.splitter = splitter

        else: # 如果不切分句子的话，直接原样返回：
            Encoder.splitter = IdentitySplitter()

    #def sent_split(self, text):
    #    if args.tokenizer_type == "BertWordPieceJp":
    #        # TODO special sentence separator for Japanese language:
    #        print('TODO japanese sent split')
    #    else:
    #        return Encoder.splitter.tokenize(text)

    def encode(self, json_line): # 该方法负责把一行输入的json格式的document(text)分别进行“句子切割”和“word to id"的变换：
        data = json.loads(json_line)
        ids = {}
        for key in self.args.json_keys:
            text = data[key]
            doc_ids = []
            for sentence in Encoder.splitter.tokenize(text): # TODO 重要，这里进行句子级别的切割，日语需要特别处理！
                #for sentence in sent_split(text):
                sentence_ids = Encoder.tokenizer.tokenize(sentence) 
                # TODO 重要，这里进行从一个字符串句子到一个ids构成的句子之间的变换

                #print('sent_ids={}'.format(sentence_ids))
                if len(sentence_ids) > 0:
                    doc_ids.append(sentence_ids)
            if self.args.append_eod:
                doc_ids[-1].append(Encoder.tokenizer.eod)
            ids[key] = doc_ids
        return ids, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data') # 一组输入参数
    apath = r'C:\Users\user\source\repos\megatron\megatron\pretrained'
    #apath = r'C:\Users\xianchaow\source\repos\megatron\pretrained\'
    #definput = apath + r'\bert_pretrain\small_data_line3.json'
    #vocabfn = apath + r'\bert-large-cased-vocab.txt'
    definput = apath + r'\bert_pretrain\small_data_line_jp.json'
    #vocabfn = r'C:\Users\user\source\repos\megatron\megatron\pretrained\tohoku-u\BERT-base_mecab-ipadic-bpe-32k\vocab.txt' 
    # for bert ja
    vocabfn = r'C:\Users\user\source\repos\gpt2-japanese\ja-bpe.txt' # for gpt-2 ja
    emojifn = r'C:\Users\user\source\repos\gpt2-japanese\emoji.json'

    group.add_argument('--input', type=str, required=False,
                       default=definput,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json') # 其他keys会被直接无视
    group.add_argument('--split-sentences', action='store_false', #action='store_true', #目前是默认split sentence of a document
                       help='Split documents into sentences.')
    #store_true 是指带触发action时为真，不触发则为假，2L说的代码去掉default初始化，其功能也不会变化
    #parser.add_argument('-c', action='store_true')#python test.py -c         => c是true（触发）
    #python test.py             => c是false（无触发）

    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False,
                       #default='BertWordPieceJp', # for japanese bert; #'BertWordPieceLowerCase' for english bert,
                       default='GPT2BPETokenizer', # for japanese gpt2; 'GPT2BPETokenizer' is for english gpt2
                       choices=['BertWordPieceLowerCase','BertWordPieceCase', 'BertWordPieceJp',
                                'GPT2BPETokenizer', 'GPT2BPETokenizerJp', 'GPT2BPETokenizerJpMecab'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, 
                       default=vocabfn,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--mecab-dict-path', type=str, default=None,
                       help='Path to the dict(ipadict/unidict) of MeCab for Japanese Word Breaker.')
    group.add_argument('--emoji-file', type=str, default=emojifn, 
                       help="emoji file for Japanese GPT-2 BPE tokenizer (if necessary)")
    group.add_argument('--append-eod', action='store_true', # 如果没有--append-eod，则表示不增加<eod>
                       help='Append an <eod> token to the end of a document.')


    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=False,
                       default='my-gpt2-ja-debug', #'my-bert-ja-debug',
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap']) # memory-map?

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False # TODO

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            print("Warning: Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input) # input json file
    fin = open(args.input, 'r', encoding='utf-8') # json file

    if nltk_available and args.split_sentences:
        nltk.download("punkt", quiet=True) # punkt for sentence tokenizer

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args) # TODO why need this? 可以直接用encoder.tokenizer
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer) # TODO，构建一个Pool对象，进程池
    encoded_docs = pool.imap(encoder.encode, fin, 25) # TODO what is "25"?
    #encoded_docs = map(encoder.encode, fin)

    level = "document"
    if args.split_sentences: # True
        level = "sentence"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys: # ['text'] 只有'text'这一个重要的元素
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = indexed_dataset.make_builder(output_bin_files[key],
                                               impl=args.dataset_impl,
                                               vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time cost to startup:", startup_end - startup_start)

    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for key, sentences in doc.items(): # sentences = [[15457, 1166, 1103, 16688, 3676]] -> jumps over the lazy dog
            # sentences = [[15457, 1166, 1103, 16688, 3676, 119], -> jumps over the lazy dog .
            # [178, 1108, 1177, 6782, 1106, 2100, 1115, 119], 
            # [1142, 1110, 170, 1363, 1285, 2052, 119], 
            # [9367, 171, 19954, 1358, 119], "fuck b ##aid ##u ." = 5 tokens
            # [9367, 17599, 7301, 4964, 119]], "fuck micro ##so ##ft ." = 5 tokens
            # jumps over the lazy dog. i was so sad to hear that. this is a good day today. fuck baidu. fuck microsoft.
            # 119="."
            for sentence in sentences:
                builders[key].add_item(torch.IntTensor(sentence))
            builders[key].end_document()
        if i % args.log_interval == 0: # or i==len(encoded_docs):
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print("Processed {} documents".format(i),
                  "({:.4f} docs/s, {:.4f} MB/s).".format(i/elapsed, mbs),
                  file=sys.stderr)

    # finally, after process:
    current = time.time()
    elapsed = current - proc_start
    mbs = total_bytes_processed/elapsed/1024/1024
    print("Processed {} documents".format(i),
        "({:.4f} docs/s, {:.4f} MB/s).".format(i/elapsed, mbs),
        file=sys.stderr)

    for key in args.json_keys:
        builders[key].finalize(output_idx_files[key])

if __name__ == '__main__':
    main() # local test okay (without gpu), to test the Japanese tokenizer in the future.
