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

"""BERT Style dataset."""

import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from megatron import get_tokenizer, get_args
from megatron import print_rank_0
from megatron import mpu
from megatron.data.dataset_utils import get_a_and_b_segments
from megatron.data.dataset_utils import truncate_segments
from megatron.data.dataset_utils import create_tokens_and_tokentypes
from megatron.data.dataset_utils import pad_and_convert_to_numpy
from megatron.data.dataset_utils import create_masked_lm_predictions


class BertDataset(Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed):
        # short_seq_prob = prob of producing a short sequence 多短算是short sequence?

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        # 从硬盘文件读取：
        self.samples_mapping = get_samples_mapping_(self.indexed_dataset,
                                                    data_prefix,
                                                    num_epochs,
                                                    max_num_samples,
                                                    self.max_seq_length,
                                                    short_seq_prob,
                                                    self.seed,
                                                    self.name)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls # [CLS]
        self.sep_id = tokenizer.sep # [SEP]
        self.mask_id = tokenizer.mask # [MASK]
        self.pad_id = tokenizer.pad # [PAD]

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        # 根据idx，读取self.samples_mapping里面的索引信息，
        # 并从indexed_dataset中读取sample，然后构造sample
        #import pdb; pdb.set_trace() # TODO very important! build the sample of dataset!
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        #print_rank_0('__getitem__: idx={}, start_idx={}, end_idx={}, seq_length={}'.format(idx, start_idx, end_idx, seq_length))
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        #totalseqlen = 0
        #for asample in sample:
        #    print_rank_0('__getitem__: len(sample.shape)={}, sample[i].shape={}'.format(len(sample), asample.shape))
        #    totalseqlen += asample.shape[0]
        #print_rank_0('__getitem__, total.seq.len={}'.format(totalseqlen))
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length,  # needed for padding
                                     self.vocab_id_list, # 20%不用[mask]的时候，其中50%使用其他的piece (使用vocab_id_list)，50%不变
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, np_rng)

# 从硬盘读取信息：
def get_samples_mapping_(indexed_dataset,
                         data_prefix,
                         num_epochs,
                         max_num_samples,
                         max_seq_length,
                         short_seq_prob,
                         seed,
                         name):
    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples "
                             "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    # for example:
    # fsi-ja-bert-large-vocab-50k-cc100_text_sentence_train_indexmap_480000000mns_512msl_0.10ssp_1234s.npy
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and \
       not os.path.isfile(indexmap_filename):
        #import pdb; pdb.set_trace()
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(' > building sapmles index mapping for {} ...'.format(
            name))
        # First compile and then import.
        from megatron.data import helpers
        samples_mapping = helpers.build_mapping( # 调用cpp里面的方法
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length - 3,  # account for added tokens
            short_seq_prob,
            seed,
            verbose)
        print_rank_0(' > done building sapmles index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elasped time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
                         time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case # TODO for what here?
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=mpu.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size() //
        torch.distributed.get_world_size(group=mpu.get_tensor_model_parallel_group()))

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping

# 根据读取的原始json的信息，构造一个训练sample样本：
def build_training_sample(sample,
                          target_seq_length, max_seq_length,
                          vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, np_rng):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        target_seq_length: Desired sequence length.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        np_rng: Random number genenrator. Note that this rng state should be
              numpy and not python since python randint is inclusive for
              the opper bound whereas the numpy one is exclusive.
              应该是来自numpy的，上界开；
              如果是来自python自己的话，是上界闭。
    """
    #import pdb; pdb.set_trace()
    ###breakpoint()
    # step 0: (check) We assume that we have at least two sentences in the sample
    #print_rank_0('build_training_sample, len(sample)={}\nsample[0].shape={}'.format(len(sample), sample[0].shape))
    #totalseqlen = 0
    #for asample in sample:
    #    print_rank_0('build_training_sample: len(sample.shape)={}, sample[i].shape={}'.format(len(sample), asample.shape))
    #    totalseqlen += asample.shape[0]
    #print_rank_0('building_training_sample, total.seq.len={}'.format(totalseqlen))
    assert len(sample) > 1
    assert target_seq_length <= max_seq_length

    # step 1: (切割，segment-order prediction）Divide sample into two segments (A and B). 
    # 如果sample中句子过多，则合并句子
    # 潜在的问题：tokens_a和tokens_b的分别的长度，以及他们的distance，没有预期：
    # is_next_random=True，表示tokens_a和tokens_b在原文段落中的出现顺序被互换了.
    # is_next_random=False，表示保持了原本在原文段落中的顺序。
    tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample, np_rng)

    # step 2: (截断) Truncate to `target_sequence_length`.
    max_num_tokens = target_seq_length
    truncated = truncate_segments(tokens_a, tokens_b, len(tokens_a),
                                  len(tokens_b), max_num_tokens, np_rng)

    # step 3: (片段类型) Build tokens and toketypes.
    # [CLS] segment1 [SEP] segment2 [SEP] = tokens (in id)
    # 0     0        0     1        1     = token types
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # step 4: (MLM) Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    # step 5: Padding.
    # 1. tokens_np: 打上了补丁之后的tokens（也被mask了，以及可能的permutation了）
    # 2. tokentypes_np: 打上了补丁之后的tokentypes（0/[cls] 0/segment1 0/[sep] 1/segment2 1/[sep] 而后是0)
    # 3. labels_np: 缺省值为-1，如果一个i位置的piece被mask了，则其labels[i]=原始的token id
    # 4. padding_mask_np: 如果是补丁，取0；否则取1
    # 5. loss_mask_np: 缺省值为0，如果一个i位置的piece被mask了，则其loss_mask[i]=1
    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    # step 6: 最后组装
    train_sample = {
        'text': tokens_np,
        'types': tokentypes_np,
        'labels': labels_np,
        'is_random': int(is_next_random), # is_next_random=1 if a,b两个segments进行了互换; 0 otherwise
        'loss_mask': loss_mask_np,
        'padding_mask': padding_mask_np,
        'truncated': int(truncated)} # 1代表sequence被截取了；0代表没有被截取
    ###breakpoint()
    #import pdb; pdb.set_trace()
    return train_sample

