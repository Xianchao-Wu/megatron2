# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
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

"""Tokenization classes for OpenAI GPT."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import sys
import json
import logging
import os
import regex as re
import numpy as np
from io import open
import six
import collections

from .bert_tokenization_jp import MecabBasicTokenizer, WordpieceTokenizer

try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE
    # tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


logger = logging.getLogger(__name__)

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json",
}
PRETRAINED_MERGES_ARCHIVE_MAP = {
    'gpt2': "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt",
}
PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP = {
    'gpt2': 1024,
}

# 如下的vocab.json和merges.txt来自：https://huggingface.co/gpt2/tree/main
# 问题：如何准备其他语言的vocab.json和merges.txt呢？TODO
VOCAB_NAME = 'vocab.json'
MERGES_NAME = 'merges.txt' 
# vocab.json和merges.txt是使用了ByteLevelBPETokenizer，适用于English的gpt-2和roberta，但是中文gpt2,日文gpt2不适用！
SPECIAL_TOKENS_NAME = 'special_tokens.txt'

def convert_to_unicode(text): # Ja can have, OKAY
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3: # 返回一个表示当前运行环境是否为python3的boolean值 
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore") 
            # errors='ignore', 设置不同的错误处理方案，'strict'的时候，如果编码错误，则会引起一个UnicodeError.
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = convert_to_unicode(reader.readline().strip())
            if not token:
                break # TODO why break? should be 'continue'?
            token = token.strip()
            vocab[token] = index
            index += 1
    # add 'end of document' here:
    eot = '<|endoftext|>'
    if not eot in vocab:
        vocab[eot] = index
        index += 1
    unk = '[UNK]'
    if not unk in vocab:
        vocab[unk] = index
        index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
            output.append(vocab[item]) # 问题，vocab是str:id，如果item不在vocab中呢？ TODO
        else:
            output.append(vocab['[UNK]'])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on (呕吐在...).
    """
    _chr = unichr if sys.version_info[0] == 2 else chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
        list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# code reference from : https://github.com/tanreinama/gpt2-japanese
#Opening C:\Users\user\source\repos\megatron\megatron\pretrained\bert_pretrain\small_data_line_jp.json
#> building GPT2BPETokenizerJpMecab tokenizer ...
# > padded vocab (size: 28444) with 100 dummy tokens (new size: 28544)
#Vocab size: 28444
#Output prefix: my-gpt2-ja-debug
#Time cost to startup: 13.938040256500244
#> building GPT2BPETokenizerJpMecab tokenizer ...
# > padded vocab (size: 28444) with 100 dummy tokens (new size: 28544)
#orig.text= 「オタ」とも呼ばれているこのペラナカン（華人）の特製料理は、とてもおいしいスナック料理です。, 
#  ids=[36, 21577, 38, 13, 28, 501, 20, 16, 33, 70, 1, 23, 1, 24, 5, 24941, 2612, 9, 6, 8567, 7613, 485, 25339, 2612, 2992, 8], 
#  back.text=['「', 'オタ', '」', 'と', 'も', '呼ば', 'れ', 'て', 'いる', 'この', '[UNK]', '(', '[UNK]', ')', 'の', '特製', '料理', 'は', '、', 'とても', 'おい', '##しい', 'スナック', '料理', 'です', '。']

#orig.text=これは、ココナッツミルクやチリペースト、レモングラス、ガーリックと一緒に魚を砕き、それを、蒸して柔らかくしたバナナの葉に包んで炭火で軽く焼いた料理です。, 
#  ids=[171, 9, 6, 15300, 26007, 18262, 49, 9315, 1, 6, 17142, 9563, 6, 6144, 1481, 13, 4265, 7, 2171, 11, 1, 6, 218, 11, 6, 23584, 16, 1, 15, 10, 17140, 5, 2311, 7, 25773, 12, 1, 12, 15883, 16878, 10, 2612, 2992, 8], 
#  back.text=['これ', 'は', '、', 'ココ', '##ナッツ', 'ミルク', 'や', 'チリ', '[UNK]', '、', 'レモン', 'グラス', '、', 'ガー', '##リック', 'と', '一緒', 'に', '魚', 'を', '[UNK]', '、', 'それ', 'を', '、', '蒸し', 'て', '[UNK]', 'し', 'た', 'バナナ', 'の', '葉', 'に', '包ん', 'で', '[UNK]', 'で', '軽く', '焼い', 'た', '料理', 'です', '。']

#orig.text=このレシピは、アジアの数地域で知られています。, 
#  ids=[70, 17141, 9, 6, 2185, 5, 276, 535, 12, 742, 20, 16, 21, 2610, 8], 
#  back.text=['この', 'レシピ', 'は', '、', 'アジア', 'の', '数', '地域', 'で', '知ら', 'れ', 'て', 'い', 'ます', '。']

#orig.text=「オタオタ（otak otak ）」は、マレー語で「脳」を意味します。, 
#  ids=[36, 1, 23, 1, 1, 1108, 9, 6, 5805, 387, 12, 36, 4025, 38, 11, 967, 15, 2610, 8], 
#  back.text=['「', '[UNK]', '(', '[UNK]', '[UNK]', ') 」', 'は', '、', 'マレー', '語', 'で', '「', '脳', '」', 'を', '意味', 'し', 'ます', '。']

#orig.text=この「オタオタ」という名前は、この料理の柔らかくトロリとした 食 感/[mask] から由来しています。, 
#  ids=[70, 36, 1, 38, 140, 1381, 9, 6, 70, 2612, 5, 1, 1, 13, 15, 10, 761, 832, 40, 1700, 15, 16, 21, 2610, 8], 
#  back.text=['この', '「', '[UNK]', '」', 'という', '名前', 'は', '、', 'この', '料理', 'の', '[UNK]', '[UNK]', 'と', 'し', 'た', '食', '感', 'から', '由来', 'し', 'て', 'い', 'ます', '。']

#orig.text=魚を使ったオタオタが、最も一般的ですが、エビやイカ、カニ、魚の頭などを用いたものなど、そのバリエーションは豊富です。, 
#  ids=[2171, 11, 2110, 10, 1, 14, 6, 1113, 654, 81, 2992, 14, 6, 13671, 49, 14693, 6, 15665, 6, 2171, 5, 1177, 64, 11, 585, 10, 120, 64, 6, 59, 11377, 9, 7112, 2992, 8], 
#  back.text=['魚', 'を', '使っ', 'た', '[UNK]', 'が', '、', '最も', '一般', '的', 'です', 'が', '、', 'エビ', 'や', 'イカ', ' 、', 'カニ', '、', '魚', 'の', '頭', 'など', 'を', '用い', 'た', 'もの', 'など', '、', 'その', 'バリエーション', 'は', '豊富', 'です', '。']
#Processed 1 documents (0.3381 docs/s, 0.0001 MB/s).

# he kick ##ed the bucket.
# chinese -> 我 爱 【中华 ##人民 ##共和国】
# spanbert
#Press any key to continue . . .

class GPT2Tokenizer(object):
    # Usage: self.tokenizer = GPT2TokenizerJp(vocab_file, mecab_dict_path, emoji_file, errors='replace',
    #                                   special_tokens=[], max_len=None)
    def __init__(self, vocab_file, mecab_dict_path=None, emoji_file=None, 
                 errors='replace', special_tokens=[], max_len=None):
        
        self.set_special_tokens(special_tokens)
        self.errors = errors

        #with open(vocab_file, encoding='utf-8') as f:
        #    self.bpe = f.read().split('\n') # self.bpe is a list

        #self.encoder = {tokstr:id for id, tokstr in enumerate(self.bpe)}
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v:k for k, v in self.encoder.items()}

        self.maxlen = max_len if max_len else np.max([len(w) for w in self.encoder]) #+ self.special_tokens])
        if emoji_file:
            with open(emoji_file, encoding='utf-8') as f:
                self.emoji = json.loads(f.read())

        self.content_repatter1 = re.compile(r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)")
        self.content_repatter2 = re.compile(r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*")
        self.content_repatter3 = re.compile(r'[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}')
        self.content_repatter4 = re.compile(r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]((0?[1-9]|[12][0-9]|3[01])日?)*(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
        self.content_repatter5 = re.compile(r"(明治|大正|昭和|平成|令和)\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日(\d{1,2}|:|\d{1,2}時|\d{1,2}分|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\))*")
        self.content_repatter6 = re.compile(r'((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+(\(税込\)|\(税抜\)|\+tax)*')

        self.basic_tokenizer = MecabBasicTokenizer(mecab_dict_path=mecab_dict_path)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder) # keep using existing method (no change)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def clean_text(self, content):
        content = jaconv.z2h(content, kana=False, digit=True, ascii=True)
        content = self.content_repatter1.sub("<URL>" ,content)
        content = self.content_repatter2.sub("<EMAIL>" ,content)
        content = self.content_repatter3.sub("<TEL>" ,content)
        content = self.content_repatter4.sub("<DATE>" ,content)
        content = self.content_repatter5.sub("<DATE>" ,content)
        content = self.content_repatter6.sub("<PRICE>" ,content)
        return content

    def encodeold(self, text, clean=False):
        text = text.replace(' ', '<SP>')
        text = text.replace('　', '<SP>')
        text = text.replace('\r\n', '<BR>')
        text = text.replace('\n', '<BR>')
        text = text.replace('\r', '<BR>')
        text = text.replace('\t', '<TAB>')
        text = text.replace('—', 'ー')
        text = text.replace('−', 'ー')
        for k,v in self.emoji['emoji'].items():
            if k in text:
                text = text.replace(k, v)
        if clean:
            text = self.clean_text(text)
        pos = 0
        result = []
        while pos < len(text):
            bp = False
            # 特殊符号下的单词的长度vs. 普通的词的长度: TODO 算法是有些慢的
            end = min(len(text), pos+self.maxlen+1) if text[pos]=='<' else pos+2
            for e in range(end, pos, -1):
                wd = text[pos:e]
                #if wd in self.bpe:
                if wd in self.encoder:
                    result.append(self.encoder[wd])
                    pos = e
                    bp = True
                    break
            if not bp:
                end = pos+1
                wd = text[pos:end]
                for i in wd.encode('utf-8'):
                    result.append(self.encoder['<|byte%d|>'%i])
                pos = end
        # TODO for debug only
        #if True:
        #    textback = self.decode(result)
        #    print('encode str2id: in={}, out={}, in.back={}'.format(text, result, textback))
        return result

    def decodeold(self, tokens, breakline='\n'):
        words = []
        byte_tokens = []
        for i in tokens:
            word = self.decoder[i]
            if word[:6] == '<|byte' and word[-2:] == '|>':
                byte_tokens.append(int(word[6:-2]))
            else:
                if len(byte_tokens) > 0:
                    words.append(bytearray(byte_tokens).decode('utf-8', errors=self.errors))
                    byte_tokens = []
                if word[:7] == '<|emoji' and word[-2:] == '|>':
                    words.append(self.emoji['emoji_inv'][word])
                elif word == '<SP>':
                    words.append(' ')
                elif word == '<BR>':
                    words.append(breakline)
                elif word == '<TAB>':
                    words.append('\t')
                else:
                    words.append(word)
        if len(byte_tokens) > 0:
            words.append(bytearray(byte_tokens).decode('utf-8', errors=self.errors))
        text = ''.join(words)
        return text

    def tokenize(self, text, clean=False):
        if clean:
            text = self.clean_text(text)
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.encoder, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.decoder, ids)

    def vocab_size(self):
        return len(self.encoder) + len(self.special_tokens)

    def encode(self, text, clean=False):
        tokens = self.tokenize(text, clean)
        tids = self.convert_tokens_to_ids(tokens)
        textback = self.decode(tids)
        #if True: # TODO debug only
        #    print('orig.text={}, ids={}, back.text={}'.format(text, tids, textback))
        return tids

    def decode(self, ids):
        return self.convert_ids_to_tokens(ids)


class GPT2TokenizerOrig(object):
    """
    GPT-2 BPE tokenizer. Peculiarities (特性):
        - Byte-level BPE
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file.
        Download and cache the pre-trained model file if needed.
        """
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_ARCHIVE_MAP:
            vocab_file = PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_model_name_or_path]
            merges_file = PRETRAINED_MERGES_ARCHIVE_MAP[pretrained_model_name_or_path]
            special_tokens_file = None
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, VOCAB_NAME)
            merges_file = os.path.join(pretrained_model_name_or_path, MERGES_NAME)
            special_tokens_file = os.path.join(pretrained_model_name_or_path, SPECIAL_TOKENS_NAME)
            if not os.path.exists(special_tokens_file):
                special_tokens_file = None
            else:
                logger.info("loading special tokens file {}".format(special_tokens_file))
        # redirect to the cache, if necessary
        try:
            from .file_utils import cached_path
            resolved_vocab_file = cached_path(vocab_file, cache_dir=cache_dir)
            resolved_merges_file = cached_path(merges_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find files {} and {} "
                "at this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_VOCAB_ARCHIVE_MAP.keys()),
                    pretrained_model_name_or_path,
                    vocab_file, merges_file))
            return None
        if resolved_vocab_file == vocab_file and resolved_merges_file == merges_file:
            logger.info("loading vocabulary file {}".format(vocab_file))
            logger.info("loading merges file {}".format(merges_file))
        else:
            logger.info("loading vocabulary file {} from cache at {}".format(
                vocab_file, resolved_vocab_file))
            logger.info("loading merges file {} from cache at {}".format(
                merges_file, resolved_merges_file))
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP:
            # if we're using a pretrained model, ensure the tokenizer wont index sequences longer
            # than the number of positional embeddings
            max_len = PRETRAINED_VOCAB_POSITIONAL_EMBEDDINGS_SIZE_MAP[pretrained_model_name_or_path]
            kwargs['max_len'] = min(kwargs.get('max_len', int(1e12)), max_len)
        # Instantiate tokenizer.
        if special_tokens_file and 'special_tokens' not in kwargs:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]
        else:
            special_tokens = kwargs.pop('special_tokens', [])
        tokenizer = cls(
            resolved_vocab_file,
            resolved_merges_file,
            special_tokens=special_tokens,
            *inputs,
            **kwargs)
        return tokenizer

    def __init__(self, vocab_file, merges_file, errors='replace',
                 special_tokens=None, max_len=None):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = json.load(open(vocab_file)) # {"!": 0, "\"": 1, "#": 2, "$": 3, ...} 从string到id的词典dict
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        bpe_data = open(merges_file, encoding='utf-8').read().split('\n')[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_data]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for
        # capitalized versions of contractions
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.special_tokens = {}
        self.special_tokens_decoder = {}
        self.set_special_tokens(special_tokens)

    def __len__(self):
        return len(self.encoder) + len(self.special_tokens)

    def set_special_tokens(self, special_tokens):
        """ Add a list of additional tokens to the encoder.
            The additional tokens are indexed starting from the last index of the
            current vocabulary in the order of the `special_tokens` list.
        """
        if not special_tokens:
            self.special_tokens = {}
            self.special_tokens_decoder = {}
            return
        self.special_tokens = dict((tok, len(self.encoder) + i)
                                   for i, tok in enumerate(special_tokens))
        self.special_tokens_decoder = {v: k for k, v in self.special_tokens.items()}
        logger.info("Special tokens {}".format(self.special_tokens))

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except BaseException:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def convert_tokens_to_ids(self, tokens):
        """ Converts a sequence of tokens into ids using the vocab. """
        ids = []
        if isinstance(tokens, str) or (sys.version_info[0] == 2 and isinstance(tokens, unicode)): # 兼容python3和python2:
            if tokens in self.special_tokens:
                return self.special_tokens[tokens]
            else:
                return self.encoder.get(tokens, 0)
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            else:
                ids.append(self.encoder.get(token, 0))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this OpenAI GPT model ({} > {}). Running this"
                " sequence through the model will result in indexing errors".format(
                    len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in BPE tokens using the vocab."""
        tokens = []
        for i in ids:
            if i in self.special_tokens_decoder:
                if not skip_special_tokens:
                    tokens.append(self.special_tokens_decoder[i])
            else:
                tokens.append(self.decoder[i])
        return tokens

    def encode(self, text):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and merge files to a directory."""
        if not os.path.isdir(vocab_path):
            logger.error("Vocabulary path ({}) should be a directory".format(vocab_path))
            return

        # 保存到如下三个文件：
        vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        merge_file = os.path.join(vocab_path, MERGES_NAME)
        special_tokens_file = os.path.join(vocab_path, SPECIAL_TOKENS_NAME)

        with open(vocab_file, 'w', encoding='utf-8') as f: # 把self.encoder的词典，写入到vocab_file!
            f.write(json.dumps(self.encoder, ensure_ascii=False))

        index = 0
        with open(merge_file, "w", encoding="utf-8") as writer: # 把self.bpe_ranks.items()写入到merge_file!
            writer.write(u'#version: 0.2\n')
            for bpe_tokens, token_index in sorted(self.bpe_ranks.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: BPE merge indices are not consecutive." # 不连贯！
                                   " Please check that the tokenizer is not corrupted!".format(merge_file))
                    index = token_index
                writer.write(' '.join(bpe_tokens) + u'\n')
                index += 1

        index = len(self.encoder)
        with open(special_tokens_file, 'w', encoding='utf-8') as writer:
            for token, token_index in sorted(self.special_tokens.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving special tokens vocabulary to {}: BPE indices are not consecutive."
                                   " Please check that the tokenizer is not corrupted!".format(special_tokens_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1

        return vocab_file, merge_file, special_tokens_file
