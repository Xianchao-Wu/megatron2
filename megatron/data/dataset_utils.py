# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, and NVIDIA.
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


# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.
# 主要参考的是Google's ALBERT的代码！

import math
import time
import collections

import numpy as np
from megatron import get_args, print_rank_0
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset

DSET_TYPE_STD = 'standard_bert'
DSET_TYPE_ICT = 'ict' # inverse cloze task? 逆完形填空？

DSET_TYPES = [DSET_TYPE_ICT, DSET_TYPE_STD]


def get_datasets_weights_and_num_samples(data_prefix,
                                         train_valid_test_num_samples):

    # The data prefix should be in the format of:
    #   weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0]*num_datasets # 长度为num_datasets，全0数组
    prefixes = [0]*num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2*i])
        prefixes[i] = (data_prefix[2*i+1]).strip()
    # Normalize weights
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]
    print_rank_0('get_datasets_weights_and_num_samples, weights={}'.format(weights))

    # Add 0.5% (the 1.005 factor) so in case the bleding dataset does
    # not uniformly distribute the number of samples, we still have
    # samples left to feed to the network.
    datasets_train_valid_test_num_samples = []
    for weight in weights:
        datasets_train_valid_test_num_samples.append(
            [int(math.ceil(val * weight * 1.005))
             for val in train_valid_test_num_samples])
    # TODO 啥意思？会有一部分test set的数据被包括在train set里面吗？
    print_rank_0('get_datasets_weights_and_num_samples, datasets_train_valid_test_num_samples={}'.
        format(datasets_train_valid_test_num_samples))

    return prefixes, weights, datasets_train_valid_test_num_samples

# 只在megatron.initialize.py的initialize_megatron方法中被使用：
def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess
    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        print("Making C++ dataset helpers module failed, exiting.")
        import sys
        sys.exit(1)


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""
    # np_rng = np.random.RandomState(seed=(self.seed + idx))

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randint in numpy is exclusive. (low=inclusive/包括; high=exclusive/不包括)
        a_end = np_rng.randint(1, n_sentences) # 产生的是[1, n_sentences-1]之间的随机整数
    tokens_a = []
    for j in range(a_end): # sample中，编号为[0, a_end-1]的句子，给tokens_a
        tokens_a.extend(sample[j]) # 这是把多个句子合并到一起了！
        # [1,2,3], [4,5] 两个句子 -> extend -> [1,2,3,4,5]

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences): # sample中，编号为[a_end, n_sentences-1]的句子，给tokens_b
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5: # 重要：以p=0.5的概率，把两个segment的在原始段落中出现的位置互换：
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a # a b两个segments进行了互换

    return tokens_a, tokens_b, is_next_random # is_next_random=a,b两个segments进行了互换


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    #print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    assert len_b > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b: # 谁更长，删除谁
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5: # 随机地删除头部或者尾部
            del tokens[0] # 删去头部
        else:
            tokens.pop() # 删去尾部
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build 
    tokentypes = (segment type, 0 for segment_a, 1 for segment_b)."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)

    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(1)

    # [CLS] segment1 [SEP] segment2 [SEP]
    # 0     0        0     1        1

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece (BERT)."""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.

    # 开头不是##的subword/piece，则这个piece是一个word的开头piece
    return not piece.startswith("##")


# only used in megatron.data.bert_dataset.py
def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 np_rng,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    cand_indexes = []
    # Note(mingdachen? Mingda Chen?): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.

    # 如果token_boundary[i]=1, 表示的是位置i的subword是一个word的开头词！

    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        
        # 全单词遮掩-不改变代码，仍然是去预测整个word的每个wordpiece，然后在全体词表上softmax!

        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                not is_start_piece(vocab_id_to_token_dict[token])):
            # 全词mask，并且cand_indexes非空，并且当前的token不是全词的开头Piece：
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1
    # 举例子：
    # tokens = '[cls] I am AI , he ##ll ##o wor ##ld . [sep]'.split(' ')
    # 1. do_whole_word_mask=True
    # cand_indexes, token_boundary = cand_boundary(tokens, True)
    # print(cand_indexes)
    # print(token_boundary)
    # [[1], [2], [3], [4], [5, 6, 7], [8, 9], [10]] (cand_indexes)
    # [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1] (token_boundary)

    # 2. do_whole_word_mask=False
    # cand_indexes, token_boundary = cand_boundary(tokens, True)
    # print(cand_indexes)
    # print(token_boundary)
    # [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]] (cand_indexes)
    # [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1] (token_boundary)


    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0: 
        # 不需要有词被mask，所以直接返回即可，用的是刚才得到的token_boundary:
        return (output_tokens, masked_lm_positions,
                masked_lm_labels, token_boundary)

    # 被mask的piece的个数（不是whole word的个数！）
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64) # ngrams=[1,2,3]
    pvals = 1. / np.arange(1, max_ngrams + 1)
    pvals /= pvals.sum(keepdims=True)
    # [0.54545455 0.27272727 0.18181818] 代表mask 1-gram/2-gram/3-gram的分别的概率


    if favor_longer_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams: # [1,2,3]
            ngram_index.append(cand_indexes[idx:idx + n])
        ngram_indexes.append(ngram_index)
    # example:
    # [[[[1]], [[1], [2]], [[1], [2], [3]]], 
    #  [[[2]], [[2], [3]], [[2], [3], [4]]], 
    #  [[[3]], [[3], [4]], [[3], [4], [5, 6, 7]]], 
    #  [[[4]], [[4], [5, 6, 7]], [[4], [5, 6, 7], [8, 9]]], 
    #  [[[5, 6, 7]], [[5, 6, 7], [8, 9]], [[5, 6, 7], [8, 9], [10]]], 
    #  [[[8, 9]], [[8, 9], [10]], [[8, 9], [10]]], 
    #  [[[10]], [[10]], [[10]]]]

    np_rng.shuffle(ngram_indexes)

    masked_lms = []
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]: 
            # 1-gram的list，因为1-gram是针对word而言的，所以1-gram里面可能有多个pieces
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np_rng.choice(ngrams[:len(cand_index_set)],
                          p=pvals[:len(cand_index_set)] /
                          pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        # sum的含义：例如从[[1], [2]] -> sum -> [1, 2]; or, [[8, 9], 10] -> sum -> [8, 9, 10]=index_set
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index) #hash

            masked_token = None
            # 80% of the time, replace with [MASK]
            if np_rng.random() < 0.8:
                masked_token = mask_id
            else:
                # 10% of the time, keep original
                if np_rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict

    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if do_permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen -> 好吧，人名也直接copy了！):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        # covered_indexes = store masked piece indexes in MLM
                        # select_indexes = this permutation's results (selected indexes)
                        continue

            n = np.random.choice(ngrams[:len(cand_index_set)],
                                 p=pvals[:len(cand_index_set)] /
                                 pvals[:len(cand_index_set)].sum(keepdims=True))
            index_set = sum(cand_index_set[n - 1], []) # index_set代表了一个完整的n-gram
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index) # 可能是两个（或者多个）非连续的n-gram的“拼接”
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)
        # [cls] I am AI , he ##ll ##o wor ##ld . [sep]
        # -> mask -> [cls] I am [mask] , he ##ll ##o wor ##ld . [sep]
        # -> permute，重排一个连续的n-gram! 也可能是多个不连续的n-gram的拼接后结果的“重新排列”！
        # -> [cls] I am [mask] , wor ##ld he ##ll ##o . [sep]
        # 即，hello world -> world hello

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i] # 改造output_tokens
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    # 1. output_tokens = mask之后的，（+permutate之后的）token id sequence
    # 2. masked lm positions = 哪些index的piece被mask掉了（或者被permute了）
    # 3. masked lm labels = 原始的mask之前（或者permute之前）的original token id
    # 4. token boundary = 0 for pieces start with "##" and 1 otherwise (长度是piece的数量，包括[cls], [sep]等）
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


# only used in megatron.data.bert_dataset.py:
def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""
    # tokens = processed token id list, with [mask] and possible permutation of pieces
    # tokentypes = [cls] segment1 [sep] segment2 [sep] -> 0 0 0 1 1

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens # 补丁的长度
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64) # 打上补丁之后的tokens
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)): # 被mask掉的token's index in current sequence
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    # 1. tokens_np: 打上了补丁之后的tokens（也被mask了，以及可能的permutation了）
    # 2. tokentypes_np: 打上了补丁之后的tokentypes（0/[cls] 0/segment1 0/[sep] 1/segment2 1/[sep] 而后是0)
    # 3. labels_np: 缺省值为-1，如果一个i位置的piece被mask了，则其labels[i]=原始的token id
    # 4. padding_mask_np: 如果是补丁，取0；否则取1
    # 5. loss_mask_np: 缺省值为0，如果一个i位置的piece被mask了，则其loss_mask[i]=1
    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np

# used in pretrain_bert.py, pretrain_gpt.py, and pretrain_ict.py
def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                    train_valid_test_num_samples,
                                    max_seq_length, masked_lm_prob,
                                    short_seq_prob, seed, skip_warmup,
                                    dataset_type='standard_bert'):

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(data_prefix[0],
                                                data_impl, splits_string,
                                                train_valid_test_num_samples,
                                                max_seq_length, masked_lm_prob,
                                                short_seq_prob, seed,
                                                skip_warmup,
                                                dataset_type=dataset_type)
    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix,
                                                  train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            prefixes[i], data_impl, splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length, masked_lm_prob, short_seq_prob,
            seed, skip_warmup, dataset_type=dataset_type)
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset,
            blending_test_dataset)


def _build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                     train_valid_test_num_samples,
                                     max_seq_length, masked_lm_prob,
                                     short_seq_prob, seed, skip_warmup,
                                     dataset_type='standard_bert'):
    
    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix,
                                           data_impl,
                                           skip_warmup)

    if dataset_type == DSET_TYPE_ICT:
        args = get_args()
        title_dataset = get_indexed_dataset_(args.titles_data_path,
                                             data_impl,
                                             skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        print_rank_0('     sentence indices in [{}, {}) total of {} '
                     'sentences'.format(start_index, end_index,
                                        end_index - start_index))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        from megatron.data.bert_dataset import BertDataset
        from megatron.data.ict_dataset import ICTDataset
        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=train_valid_test_num_samples[index],
                max_seq_length=max_seq_length,
                seed=seed
            )

            if dataset_type == DSET_TYPE_ICT:
                args = get_args()
                dataset = ICTDataset(
                    block_dataset=indexed_dataset,
                    title_dataset=title_dataset,
                    query_in_block_prob=args.query_in_block_prob,
                    use_one_sent_docs=args.use_one_sent_docs,
                    **kwargs
                )
            else:
                dataset = BertDataset(
                    indexed_dataset=indexed_dataset,
                    masked_lm_prob=masked_lm_prob,
                    short_seq_prob=short_seq_prob,
                    **kwargs
                )

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == \
                (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    print_rank_0(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix,
                                           data_impl,
                                           skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))

    print_rank_0(' > indexed dataset stats:')
    print_rank_0('    number of documents: {}'.format(
        indexed_dataset.doc_idx.shape[0] - 1))
    print_rank_0('    number of sentences: {}'.format(
        indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""

    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0
    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index