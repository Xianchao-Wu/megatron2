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


import torch


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor, num_partitions,
                                contiguous_split_chunks=False):
    """Split a tensor along its last dimension.
    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.
    Return:
        tuple of torch.tensor objects.
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    # 所以，在调用本方法之后，有可能需要再次调用contiguous()方法！
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class VocabUtility:
    """Split the vocabulary into `world_size` (num of gpus) chunks amd return the
        first and last index of the vocabulary belonging to the `rank`
        partition: Note that indecies in [first, last)"""

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(per_partition_vocab_size,
                                                  rank, world_size):
        """返回子词表的[index_first, index_last)，两个index """
        index_f = rank * per_partition_vocab_size # first, inclusive
        index_l = index_f + per_partition_vocab_size # last, exclusive
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(global_vocab_size, rank, world_size):
        """当前rank的gpu所覆盖的子词表index: [index_first, index_last)"""
        # 需要确保 global_vocab_size % world_size == 0!
        per_partition_vocab_size = divide(global_vocab_size, world_size) # alike [global_vocab_size//world_size] 
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size)
