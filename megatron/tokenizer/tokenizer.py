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

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .bert_tokenization_jp import FullTokenizer as FullBertTokenizerJp
from .gpt2_tokenization import GPT2Tokenizer
from .gpt2_tokenization_jp import GPT2Tokenizer as GPT2TokenizerJp
from .gpt2_tokenization_jp_mecab import GPT2Tokenizer as GPT2TokenizerJpMecab


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type), # e.g., BertWordPieceLowerCase
              flush=True)

    # Select and instantiate the tokenizer.
    assert args.vocab_file is not None
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False)
    elif args.tokenizer_type == "BertWordPieceJp": # no need to separate "case" and "lowercase" for Jp
        tokenizer = _BertWordPieceTokenizerJp(vocab_file=args.vocab_file,
                                              mecab_dict_path=args.mecab_dict_path)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None # for english only, byte level bpe tokenizer is not fittable for japanese/chinese
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'GPT2BPETokenizerJp':
        tokenizer = _GPT2BPETokenizerJp(vocab_file=args.vocab_file,
                                        mecab_dict_path=args.mecab_dict_path,
                                        emoji_file=args.emoji_file)
    elif args.tokenizer_type == 'GPT2BPETokenizerJpMecab':
        tokenizer = _GPT2BPETokenizerJpMecab(vocab_file=args.vocab_file,
                                        mecab_dict_path=args.mecab_dict_path,
                                        emoji_file=args.emoji_file)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, # 28996
                                                      args) # 28996 -> 29056

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size # wc -l bert-large-cased-vocab.txt, -> , 28996 bert-large-cased-vocab.txt
    # 128
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size # 1
    while (after % multiple) != 0: # 要求vocab_size可以整除128，从而从28996修改为29056，且有：29056/128=227是整数了
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after # 并没有修改任何东西，只是计算出来了一个after=new vocab size with pad


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True):
        if lower_case: # True
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]'] # 101
        self.sep_id = self.tokenizer.vocab['[SEP]'] # 102
        self.pad_id = self.tokenizer.vocab['[PAD]'] # 0
        self.mask_id = self.tokenizer.vocab['[MASK]'] # 103

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

class _BertWordPieceTokenizerJp(AbstractTokenizer):
    """Original BERT wordpiece tokenizer for Japanese (mecab+ipadict, wordpiece, case-insensitive)."""

    def __init__(self, vocab_file, lower_case=True, mecab_dict_path=None):
        if lower_case: # True by default
            name = 'BERT Japanese Lower Case'
        else:
            name = 'BERT Japanese Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizerJp(vocab_file, do_lower_case=lower_case, mecab_dict_path=mecab_dict_path)
        self.cls_id = self.tokenizer.vocab['[CLS]'] # 101
        self.sep_id = self.tokenizer.vocab['[SEP]'] # 102
        self.pad_id = self.tokenizer.vocab['[PAD]'] # 0
        self.mask_id = self.tokenizer.vocab['[MASK]'] # 103

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = 'GPT2 BPE'
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>'] # eod=end of document

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

class _GPT2BPETokenizerJp(AbstractTokenizer):
    """Adapted Japanese GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, mecab_dict_path=None, emoji_file=None):
        name = 'GPT2 BPE Japanese'
        super().__init__(name)

        self.tokenizer = GPT2TokenizerJp(vocab_file, mecab_dict_path, emoji_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>'] # eod=end of document
        # C:\Users\user\source\repos\gpt2-japanese\ja-bpe.txt 中有这个eod!

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

class _GPT2BPETokenizerJpMecab(AbstractTokenizer):
    """Adapted Japanese GPT2 Mecab tokenizer."""

    def __init__(self, vocab_file, mecab_dict_path=None, emoji_file=None):
        name = 'GPT2 Mecab Japanese'
        super().__init__(name)

        self.tokenizer = GPT2TokenizerJpMecab(vocab_file, mecab_dict_path, emoji_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>'] # eod=end of document
        # C:\Users\user\source\repos\gpt2-japanese\ja-bpe.txt 中有这个eod!

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

