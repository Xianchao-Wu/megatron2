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


"""Model and data parallel groups."""

import torch

from .utils import ensure_divisibility # 保证"可除性"


# Intra-layer model parallel group that the current rank belongs to. TODO 不懂，层内
# -> Intra-layer = tensor并行！
# 相当于把一层的一个tensor，切分成很多份儿，所以是把一层分成几段儿。
# 张量-模型并行-组：
_TENSOR_MODEL_PARALLEL_GROUP = None # [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15] 中的一个
# 当前gpu进程/caller所在的 张量并行组


# Inter-layer model parallel group that the current rank belongs to. TODO 不懂，夹层，中间层
# -> Inter-layer = pipeline并行!
# 相当于单个层和下一个完整的层之间的并行化：
_PIPELINE_MODEL_PARALLEL_GROUP = None # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]中的一个
# 当前gpu进程/caller所在的 管道并行组

# Model parallel group (both intra/tensor- and inter/pipeline-) that the current rank belongs to.
# 当前gpu所在的模型并行组：
_MODEL_PARALLEL_GROUP = None # [0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]
# TODO 不懂：是把上面的管道的两个group合一吗：
# [0, 4, 8, 12], [1, 5, 9, 13] -> 并集 -> [0, 1, 4, 5, 8, 9, 12, 13]
# [2, 6, 10, 14], [3, 7, 11, 15] -> [2, 3, 6, 7, 10, 11, 14, 15]

# Embedding group.
# 当前gpu所在的embedding组：
_EMBEDDING_GROUP = None # alike, [0, 12], [1, 13], [2, 14], [3, 15]

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None # 当前gpu所在的（所属于的）数据并行组, 
# [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]

# These values enable us to change the mpu sizes on the fly. (TODO 什么是on the fly?)
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None # 2, alike [g0, g1], ..., [g14, g15]
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None # 4, alike [g0,g4,g8,g12]...

# 当前进程在其所在“张量”进程组中的相对rank，从0到group_world_size-1:
_MPU_TENSOR_MODEL_PARALLEL_RANK = None # relative rank in one group, 取值为0，1; [0, 1]

# 当前进程在其所在“管道”进程组中的相对rank，从0到group_world_size-1:
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None # relative rank in one group, 取值为0,1,2,3；[0, 4, 8, 12]

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage
_PIPELINE_GLOBAL_RANKS = None

def is_unitialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is None


def initialize_model_parallel(tensor_model_parallel_size_=1,
                              pipeline_model_parallel_size_=1):
    """
    Initialize model data parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used to parallelize model tensor. 
            (模型中的张量并行相关的gpus数量) = 2 
            [一个tensor被切割成两份，分别给两个gpu运算]
        pipeline_model_parallel_size: number of GPUs used to parallelize model pipeline. 
            (模型中的管道并行相关的gpus数量) = 4
            [一个网络，假设有12层，则被均匀切分成4份，每一份是3层，每一份给一个gpu]

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and 
    we use x=2 GPUs to parallelize the model tensor, and y=4 GPUs to parallelize the model pipeline. 
    [2GPUs - 并行化模型张量；4GPUs - 并行化模型管道]
    则z=data_parallel_size=16/(x*y)=2 (数据并行组的数量)
    TODO Why? 为什么? z=16/(x*y)?
    
    The present function will create:
    8 tensor model-parallel groups, (每个tensor被二等分，所以是一组2个gpu，一共16个gpu，则一共16/2=8组)
    4 pipeline model-parallel groups, and (每个网络被切割成四份，所以是一组4个gpu，一共16个gpu，则一共16/4=4组)
    8 data-parallel groups as: (TODO不懂)

        8 data_parallel groups (8个数据并行组):
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]

        8 tensor model-parallel groups (8个张量相关的模型并行组): 16/x=16/2=8
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]

        4 pipeline model-parallel groups (4个管道相关的模型并行组): 16/y=16/4=4
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]

    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box. (rank0-7是第一个dgx-1; rank8-15是第二个dgx-1)
    """
    if torch.distributed.get_rank() == 0: # 当前gpu的rank（序号）=0的时候，打印Log：
        print('> initializing tensor model parallel with size {}'.format(
            tensor_model_parallel_size_))
        print('> initializing pipeline model parallel with size {}'.format(
            pipeline_model_parallel_size_))

    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size() # 16，整体gpu的数量
    tensor_model_parallel_size = min(tensor_model_parallel_size_, world_size) # 2，一个tensor二等分
    pipeline_model_parallel_size = min(pipeline_model_parallel_size_, world_size) # 4, 一个网络四等分
    ensure_divisibility(world_size,
                        tensor_model_parallel_size * pipeline_model_parallel_size) # 16/(2*4)=2
    data_parallel_size = world_size // (tensor_model_parallel_size *
                                        pipeline_model_parallel_size) # z=16/(x*y)=2，一份数据(mini-batch)二等分? TODO 例如，我们有512张卡，8个gpu保存一个完整的model，那么我们就是有tensor_model_parallel_size*pipeline_model_parallel_size=8，从而一共有512/8=64个模型的复制，每个独立的模型（保存在8gpu上）的指定部分，例如gpu 1, 9, 17, ..., 505，是共享相同的参数的，所以我们可以对这64个gpu，构造一个data parallel group，则这个数据并行组的大小就是64，我们可以为gpu 1, 9, ..., 505安排64个mini-batch，并对他们进行all-reduce。同理，对于gpu 2, 10, ..., 506；以及gpu 3, 11, ..., 507； gpu 4, 12, ..., 508；gpu 5, 13, ..., 509；gpu 6, 14, ..., 510；gpu 7, 15, ..., 511；gpu 8, 16, ..., 512都可以采用类似的方法来进行操作。

    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # 张量-模型并行-组 的数量, 8
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # 管道-模型并行-组 的数量, 4
    num_data_parallel_groups = world_size // data_parallel_size # 数据并行组的数量, 8

    rank = torch.distributed.get_rank() # 当前gpu（进程，caller）的rank


    ### Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP # global 1; TODO->okay: global的范围是？本py文件内
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'
    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size): # i in (0,4). TODO why? not "data_parallel_size"?
        # TODO有意思：这里是对 管道-模型并行-size=4 进行循环：
        start_rank = i * num_pipeline_model_parallel_groups # i * 4 -> 0, 4, 8, 12
        end_rank = (i + 1) * num_pipeline_model_parallel_groups # (i+1)*4 -> 4, 8, 12, 16
        for j in range(tensor_model_parallel_size): # j in (0,2)
            # TODO有意思：这里是对 张量-模型并行-size=2 进行循环：
            ranks = range(start_rank + j, end_rank,
                          tensor_model_parallel_size) # 2
            all_data_parallel_group_ranks.append(list(ranks)) # [[0]]
            group = torch.distributed.new_group(ranks)
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group

    # i=4, i=0的时候: start_rank=0, end_rank=4; j=0 and 1 -> range(0, 4, 2)=[0, 2], and range(1, 4, 2)=[1, 3]
    #range(0, 4, 2) -> [0, 2] -> gpu0 and gpu2 share the same parameter set
    #range(1, 4, 2) -> [1, 3]

    #range(4, 8, 2) -> [4, 6]
    #range(5, 8, 2) -> [5, 7]

    #range(8, 12, 2) -> [8, 10]
    #range(9, 12, 2) -> [9, 11]

    #range(12, 16, 2) -> [12, 14]
    #range(13, 16, 2) -> [13, 15]

    # all_data_parallel_group_ranks:
    # [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]


    ### Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP # global 2,
    assert _MODEL_PARALLEL_GROUP is None, \
        'model parallel group is already initialized'
    for i in range(data_parallel_size): # 2. TODO why? not "pipeline_model_parallel_size"?
        # TODO有意思：这里是对 数据-并行-size=2 进行循环：
        ranks = [data_parallel_group_ranks[i]
                 for data_parallel_group_ranks in all_data_parallel_group_ranks]
        # i=0的时候，取的是all_data_parallel_group_ranks中的每个列表的第0个元素；
        # i=1的时候，取的是all_data_parallel_group_ranks中的每个列表的第1个元素.
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group # 当前gpu所在的模型并行group
    # [0, 1, 4, 5, 8, 9, 12, 13] -> [0, 1, 4, 5, 8, 9, 12, 13]
    # [2, 3, 6, 7, 10, 11, 14, 15] -> [2, 3, 6, 7, 10, 11, 14, 15]


    ### Build the tensor model-parallel groups. ### okay ###
    # 这部分代码最简单直接，没有涉及到其他的pipeline_parallel或者data_parallel的部分！
    global _TENSOR_MODEL_PARALLEL_GROUP # global 3,
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups): # 8
        ranks = range(i * tensor_model_parallel_size, # 2
                      (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
    # range(0, 2) -> [0, 1]
    # range(2, 4) -> [2, 3]
    # range(4, 6) -> [4, 5]
    # range(6, 8) -> [6, 7]
    # range(8, 10) -> [8, 9]
    # range(10, 12) -> [10, 11]
    # range(12, 14) -> [12, 13]
    # range(14, 16) -> [14, 15]


    ### Build the pipeline model-parallel groups and ### okay ###
    ### embedding groups (which are first and last rank in each pipeline model-parallel group, TODO why?).
    global _PIPELINE_MODEL_PARALLEL_GROUP # global 4,
    global _PIPELINE_GLOBAL_RANKS # global 5,
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP # global 6,
    assert _EMBEDDING_GROUP is None, \
        'embedding group is already initialized'

    for i in range(num_pipeline_model_parallel_groups): # 4, i=0,1,2,3
        ranks = range(i, world_size, # world_size=16
                      num_pipeline_model_parallel_groups) # 0 to 16 with step=4; 1 to 16 with step=4;
        # 2 to 16 with step=4, and 3 to 16 with step=4. 
        # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between first and last stages). TODO 什么含义?
        # 为什么要互换第一个和最后一个gpu的梯度呢？
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            # [0, 12], [1, 13], [2, 14], or [3, 15]
        else:
            embedding_ranks = ranks
        group = torch.distributed.new_group(embedding_ranks)
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group

    # _PIPELINE_MODEL_PARALLEL_GROUP: 如下四个list中取一个List：
    # range(0, 16, 4) -> [0, 4, 8, 12] -> 取头取尾 -> [0, 12]
    # range(1, 16, 4) -> [1, 5, 9, 13] -> 取头取尾 -> [1, 13]
    # range(2, 16, 4) -> [2, 6, 10, 14] -> 取头取尾 -> [2, 14]
    # range(3, 16, 4) -> [3, 7, 11, 15] -> 取头取尾 -> [3, 15]

    # _EMBEDDING_GROUP:
    # [0, 12], [1, 13], [2, 14], or [3, 15] 中取一组


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    # 要求三个group都非None的情况下，才返回True
    if _TENSOR_MODEL_PARALLEL_GROUP is None or \
        _PIPELINE_MODEL_PARALLEL_GROUP is None or \
        _DATA_PARALLEL_GROUP is None:
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'model parallel group is not initialized'
    return _MODEL_PARALLEL_GROUP
    # [0, 1, 4, 5, 8, 9, 12, 13], [2, 3, 6, 7, 10, 11, 14, 15]中的一个


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'intra_layer_model parallel group is not initialized'
    return _TENSOR_MODEL_PARALLEL_GROUP
    # [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]中的一个


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller/呼叫者 rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]中的一个


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_GROUP
    # [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]] 中的一个
    # 当前gpu进程caller, 所在的group!

def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, \
        'embedding group is not initialized'
    return _EMBEDDING_GROUP
    # [0, 12], [1, 13], [2, 14], [3, 15]中的一个


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size""" # e.g., 2，一个张量二等分
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size
    # wrong: 设置总的gpu的数量（跨dgx的）
    # correct: 设置的是张量被切割成多少份
    '''$ bash grepkey.sh "set_tensor_model_parallel_world_size"
    megatron/initialize.py:from megatron.mpu import set_tensor_model_parallel_rank, set_tensor_model_parallel_world_size
    megatron/initialize.py:        set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    tools/merge_mp_partitions.py:    mpu.initialize.set_tensor_model_parallel_world_size(1)
    tools/merge_mp_partitions.py:    mpu.initialize.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
    megatron/mpu/__init__.py:from .initialize import get_tensor_model_parallel_world_size, set_tensor_model_parallel_world_size
    megatron/mpu/initialize.py:def set_tensor_model_parallel_world_size(world_size):
    '''

def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size""" # e.g., 4, 一个网络四等分
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size
    # TODO for what?
    # not used...
    '''
    user@USER-T7HGFMUO5B MINGW64 ~/source/repos/megatron/megatron (main)
    $ bash grepkey.sh "set_pipeline_model_parallel_world_size"
    megatron/mpu/__init__.py:from .initialize import get_pipeline_model_parallel_world_size, set_pipeline_model_parallel_world_size
    megatron/mpu/initialize.py:def set_pipeline_model_parallel_world_size(world_size):
    '''

def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())
    # 2, from lengths of [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]
    # 返回(张量并行)当前进程组内的进程数。



def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())
    # 4, from lengths of [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
    # 返回（管道并行）当前进程组内的进程数。



def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank
    # not used
    '''
    user@USER-T7HGFMUO5B MINGW64 ~/source/repos/megatron/megatron (main)
    $ bash grepkey.sh "set_pipeline_model_parallel_rank"
    megatron/mpu/__init__.py:from .initialize import get_pipeline_model_parallel_rank, set_pipeline_model_parallel_rank
    megatron/mpu/initialize.py:def set_pipeline_model_parallel_rank(rank):
    '''

def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())
    # from [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]
    # return 0 or 1? 返回的是相对顺序，还是绝对rank? TODO -> okay
    # 是相对rank，[0, group_world_size-1]
    # 返回当前进程的 rank。
    # rank 是赋值给一个分布式进程组组内的每个进程的唯一识别。一般而言，rank 均为从 0 到 group_world_size-1 的整数。

def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
    # 返回的是0-3的相对rank? ->是的
    # 当前进程在其所在的，管道-模型并行-进程组，的相对rank

def is_pipeline_first_stage():
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == 0
    # rank == 0 or not
    # relative rank from [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]

def is_pipeline_last_stage():
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1) 
    # rank == 3 or not
    # relative rank from [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]

def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank() # 当前gpu的全局rank
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size
    # (0...15//2)*2 TODO 什么含义?

def get_pipeline_model_parallel_first_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15] 取一个，
    # 对应的是当前caller所在的 管道并行-组，再从中取
    # 第0个，例如0，1，2，3

def get_pipeline_model_parallel_last_rank():
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1 # 3
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15] 取一个，
    # 对应的是当前caller所在的 管道并行-组，再从中取
    # 最后一个，例如12, 13, 14, 15

def get_pipeline_model_parallel_next_rank(): # （环）后一个，返回的是global rank
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]
    # 如果当前的gpu是第world_size-1个，则world_size % world_size=0，返回第0个
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]
    # 这里，world_size=4

def get_pipeline_model_parallel_prev_rank(): # (环)前一个，返回的是global rank
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size] 
    # 如果当前的是第0个，那么-1/world_size=word_size-1
    # e.g., -1/100 = 99!
    # [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]

def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())
    # 2, [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]
    # 返回数据并行,当前进程组内的进程数。

def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())
    # from [[0, 2], [1, 3], [4, 6], [5, 7], [8, 10], [9, 11], [12, 14], [13, 15]]
    # 返回的是relative rank, 0 or 1

def destroy_model_parallel():
    """Set the groups to none. - used for testing only TODO """ 
    # TODO
    # tensor, pipeline, data三个大方向！
    # 其中tensor, pipeline属于模型的并行化；
    # data属于数据的并行化：

    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

    # appended by xianchaow@nvidia.com
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None

