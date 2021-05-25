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

"""Blendable dataset."""

import time

import numpy as np
import torch

from megatron import print_rank_0
from megatron import mpu


class BlendableDataset(torch.utils.data.Dataset):


    def __init__(self, datasets, weights):

        self.datasets = datasets
        num_datasets = len(datasets) # 数据集的个数
        assert num_datasets == len(weights) # 每个数据集对应一个weight

        self.size = 0
        for dataset in self.datasets:
            self.size += len(dataset) # 每个数据集的size的叠加，得到self.size
            # 样本的个数

        # Normalize weights.
        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights # 标准化，类似于to prob (1,2,3)->(1/6, 2/6, 3/6)

        # Build indecies.
        start_time = time.time()
        assert num_datasets < 255
        self.dataset_index = np.zeros(self.size, dtype=np.uint8) # 样本的个数
        self.dataset_sample_index = np.zeros(self.size, dtype=np.int64) # 样本的个数

        from megatron.data import helpers
        helpers.build_blending_indices(self.dataset_index, # all zeros
                                       self.dataset_sample_index, # all zeros
                                       weights, num_datasets, self.size, # 标准化之后的weights，数据集的个数，样本的个数
                                       torch.distributed.get_rank() == 0)
        print_rank_0('> elapsed time for building blendable dataset indices: '
                     '{:.2f} (sec)'.format(time.time() - start_time))


    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        # TODO->DONE, 两者都是使用idx？难道dataset_index的长度和dataset_sample_index的相同？->相同
        # 都是sample.size，这样的话,idx就是整体的[0, self.size-1]这样的取值范围了（遍历所有样本）。

        # 一个样本的下标idx，所在的数据集的idx，以及在数据集中的具体的位置[0, len(self.datasets[dataset_idx])-1].
        dataset_idx = self.dataset_index[idx]
        sample_idx = self.dataset_sample_index[idx]
        return self.datasets[dataset_idx][sample_idx] # 关键在于索引，数据集的idx，之后是sample的idx
