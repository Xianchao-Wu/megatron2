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

from .initialize import get_tensor_model_parallel_group # 当前gpu所在的并行群组group
from .initialize import get_tensor_model_parallel_world_size # 当前并行群组（集群）的gpu的数量
from .initialize import get_tensor_model_parallel_rank # 当前gpu在其所在的并行群组中的rank
from .utils import split_tensor_along_last_dim # 沿着最后一个维度切割tensor


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group.
    将多个进程中的数据按照指定的映射函数进行运算得到最后的结果存在一个进程中，例如求和归约，
    将四个不同进程的数据归约求和后保存到第一个进程。（group归约后，保存到当前进程，当前gpu）。
    All_reduce，实现方法类似reduce+scatter，即reduce之后的结果，重新发回到每个进程，使得每个gpu都有最新的data.
    """

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size()==1:
        return input_

    # All-reduce. 就是这么简单？！直接调用已有的all_reduce方法！
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice. 切：按照最后一列"""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _gather(input_):
    """Gather tensors and concatenate along the last dimension. 拼凑，拼接"""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)] # 在当前gpu，声明这么大的list，感觉内存很有压力啊！TODO
    tensor_list[rank] = input_ # input_是一个gpu的负担量，这里把tensor_list的rank位置的值，设定为input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group()) # 把多个进程的数据，拼凑在一起！

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous() #按照最后一列拼接，然后设置整个output是contiguous的！

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region. 【NO！类似scatter分发】"""
    # “复制-全归约”，前向类似broadcast，后向类似于all-reduce

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_ # 直接就是自己

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output) # all reduce所有gpus上的结果，归约，并让所有gpu都保持最新


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region. 全归约"""
    # “全归约-复制", 前向全归约，后向直接传递

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck (?chunk?) to the rank."""
    # 切割分发到各个gpu（前向切割分发，后向拼凑串接）
    # “切割-拼凑”

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_) # 切割成多个部分

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output) # 拼凑回来


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""
    # 从各个gpu中提取并拼凑（前向拼凑，后向切割）
    # “拼凑-切割”

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_) # 前向是拼凑

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output) # 后向是切割


# -----------------
# Helper functions.
# -----------------

def copy_to_tensor_model_parallel_region(input_): # 复制-全归约
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_): # 全归约-复制
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_): # 切割-拼凑
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_): # 拼凑-切割
    return _GatherFromModelParallelRegion.apply(input_)
