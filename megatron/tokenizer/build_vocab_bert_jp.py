import os
import glob
import tempfile
import argparse

import sentencepiece
from logzero import logger

#from tokenization import MecabBasicTokenizer
from bert_tokenization_jp import MecabBasicTokenizer


CONTROL_SYMBOLS = ['[CLS]', '[SEP]', '[MASK]']


def main(args):
    tokenizer = MecabBasicTokenizer(do_lower_case=args.do_lower_case,
                                    mecab_dict_path=args.mecab_dict_path)
    with tempfile.TemporaryDirectory() as tempdir:
        # read input files and write to a temporary file
        linenum = 0
        concat_input_file = open(os.path.join(tempdir, 'input.txt'), 'w')
        for input_path in glob.glob(args.input_file):
            with open(input_path, 'r') as input_file:
                logger.info('Reading {}'.format(input_path))
                for line in input_file:
                    linenum += 1
                    if linenum % 10000 == 0:
                        print('read already: {} lines'.format(linenum))
                    tokens = tokenizer.tokenize(line.strip('\n'))
                    print(' '.join(tokens), file=concat_input_file)
        print('done read {} lines in total.'.format(linenum))

        # train a SentencePiece model and store the vocabulary file to a temp directory
        logger.info('Training a SentencePiece model')
        commands = {
            'input': concat_input_file.name,
            'model_prefix': os.path.join(tempdir, 'sp'),
            'model_type': args.subword_type,
            'normalization_rule_name': 'identity',
            'vocab_size': args.vocab_size,
            'character_coverage': 0.9995 if args.subword_type == 'bpe' else 1.0,
            'pad_id': 0,
            'unk_id': 1,
            'bos_id': -1,
            'eos_id': -1,
            'control_symbols': ','.join(CONTROL_SYMBOLS),
            'input_sentence_size': args.sentence_size,
            'shuffle_input_sentence': 'true'
        }
        command_line = ' '.join(['--{}={}'.format(k, v) for k, v in commands.items()])
        sentencepiece.SentencePieceTrainer.Train(command_line)
        concat_input_file.close()

        # convert SentencePiece vocabulary into WordPiece format that is used in BERT
        with open(os.path.join(tempdir, 'sp.vocab')) as vocab_file, \
             open(args.output_file, 'w') as output_file:
            for line in vocab_file:
                sp_token, _ = line.rstrip('\n').split('\t')
                if sp_token == '<pad>':
                    output_token = '[PAD]'
                elif sp_token == '<unk>':
                    output_token = '[UNK]'
                elif sp_token in CONTROL_SYMBOLS:
                    output_token = sp_token
                elif sp_token.startswith('\u2581'):
                    # e.g. "▁word" -> "word"
                    output_token = sp_token[1:]
                elif args.subword_type == 'bpe':
                    # e.g. "tion" -> "##tion"
                    output_token = '##' + sp_token
                else:
                    output_token = sp_token

                output_file.write(output_token + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True,
        help='Input raw text file (or comma-separated list of files).')
    parser.add_argument('--output_file', type=str, required=True,
        help='Output vocabulary file.')
    parser.add_argument('--subword_type', choices=('bpe', 'char', 'word'),
        help='Subword type. ("bpe", "char", or "word")')
    parser.add_argument('--vocab_size', type=int, default=32000,
        help='WordPiece vocabulary size. [32000]')
    parser.add_argument('--sentence_size', type=int, default=1000000,
        help='Limit the input sentence size. [1000000]')
    parser.add_argument('--do_lower_case', action='store_true',
        help='Lowercase the input text.')
    parser.add_argument('--mecab_dict_path', type=str,
        help='Path to MeCab custom dictionary.')
    args = parser.parse_args()

    main(args)
