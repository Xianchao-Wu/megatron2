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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank # """Return my rank for the tensor model parallel group.""" 
# 张量-并行群组中的当前gpu的rank
# alike [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]中得到的0或者1

from .initialize import get_tensor_model_parallel_world_size # 
# """Return world size for the tensor model parallel group.""" 
# 张量-并行群组中的gpu的数量 (例如，2，类似于对一个张量进行上下切割成两份)
# 2. alike from [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]

from .mappings import copy_to_tensor_model_parallel_region # “forward复制-backward全归约” - 纵刀流的f (column parallel linear layer)
from .mappings import gather_from_tensor_model_parallel_region # “forward拼凑-back切割” - 纵刀流的g (column parallel linear layer)

from .mappings import scatter_to_tensor_model_parallel_region # "forward切割-backward拼凑" - 横刀流的f (row parallel linear layer)
from .mappings import reduce_from_tensor_model_parallel_region # “forward全归约-backward复制” - 横刀流的g (row parallel linear layer)

from .random import get_cuda_rng_tracker # rng=random number generator 跟踪仪
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from megatron import get_args


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {'tensor_model_parallel': False,
                                      'partition_dim': -1,
                                      'partition_stride': 1}


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute) # python内置函数，用于判断对象是否包含对应的属性
    # Set the attributes.
    setattr(tensor, 'tensor_model_parallel', is_parallel) # python自带的函数，用于设置属性值，该属性不一定是存在的！
    setattr(tensor, 'partition_dim', dim)
    setattr(tensor, 'partition_stride', stride) # 步幅


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute]) # 只有当tensor没有一个属性
        #的时候，才赋值缺省值； "tensor_model_parallel", "partition_dim", and "partition_stride".


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute): # 只有当source_tensor中有属性的时候，才进行复制
            setattr(destination_tensor, attribute,
                    getattr(source_tensor, attribute))
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(tensor=weight, # torch.Size([29056, 1024])
                                         is_parallel=True, # 属性1
                                         dim=partition_dim, # 属性2
                                         stride=stride) # 属性3

    with get_cuda_rng_tracker().fork(): # rng = random number generator
        init_method(weight) # megatron/model/utils.py's init_method_normal's init_() method


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(tensor=weight,
                                         is_parallel=True, # 属性1
                                         dim=partition_dim, # 属性2
                                         stride=stride) # 属性3

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    #rank = get_model_parallel_rank() # TODO
    rank = get_tensor_model_parallel_rank() # TODO
    world_size = get_tensor_model_parallel_world_size() # 当前的gpu所在的"张量-并行群组"group中的gpu的数量
    # alike 2 for groups: [0, 1], [2,3], [4,5], [6,7], [8,9], [10, 11], [12, 13], [14, 15]
    my_weight_list = weight_list[rank::world_size] # 起点为rank，步长为world_size，一直到weight_list的尽头

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings # vocab size
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size() 
        # 当前gpu所在的"张量-并行群组"中gpu的数量
        # e.g., self.tensor_model_parallel_size=2
        # Divide the weight matrix along the vocabulary dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size( 
                # 当前rank的gpu所覆盖的子词表index: [index_first, index_last)
                self.num_embeddings, get_tensor_model_parallel_rank(), 
                # 当前gpu在其所在的并行群组中的rank
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index # 一个gpu上负责的单词的数量

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty( # from torch.nn.parameter import Parameter
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype)) # 当前GPU
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        #import pdb; pdb.set_trace()
        # input_ 取值和单词在词表中的绝对序号有关系
        if self.tensor_model_parallel_size > 1: # 当前并行群组中gpu的数量
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index) # | bit-based OR
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index # TODO 为什么这里要-self.vocab_start_index?
            masked_input[input_mask] = 0
            # example, [1,2,3,4] = vocab; current gpu's [1,2]
            # input_=[3,2,4,1,2] （单词的id序号）
            # input_mask=[True, False, True, False, False]
            # masked_input = [3,2,4,1,2] - 1 = [2,1,3,0,1]
            # masked_input = [0,1,0,0,1]??? -> 没有关系的！参看output_parallel[input_mask, :]=0.0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight, # self.weight.shape=torch.size([29056, 1024]), so it is embedding weight matrix!
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1: # e.g., tensor_model_parallel_size=2
            output_parallel[input_mask, :] = 0.0 # 这里重要，[3,2,4,1,2]中的3和4对应的embeddings被设置为了0.0
            # 所以, input_.clone() - self.vocab_start_index就无关紧要了！
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel) # 前向全归约，后向复制
        # 当前token的index如果不在当前gpu上，则其embedding vector为全0 -> 则通过all_reduce的方式，加和回来！
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. Here, A is parallelized along
    its second dimension (A is a matrix) as A = [A_1, ..., A_p].

    解释：A是矩阵，从h维度，变换到4h维度. 相当于说上面公式中的X的最后一个维度是h，A是h*4h的矩阵
    而在代码中，是A按照h*(4h/p)的方式被并行化的：即所谓的A按照其第二个维度来被并行化！

    Arguments:
        input_size: first dimension of matrix A. [参数A的整体的行数] cx
        output_size: second dimension of matrix A. [参数A的整体的列数] ca
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i (注意输入X，不切割！）[rx, cx] * [cx, cai] -> [rx, cai]
                       rx=row number of x, cx=column number of x, cai=column number of A_i
                       这样的话，Y_i (i=1,...,p)的shape就是[rx, cai]
                       如果要gather的话，应该如何处理？
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization. [TODO] 何意?
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with (打成一片，与...融合) other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size # h=hidden_size of transformer
        self.output_size = output_size # 4*h
        self.gather_output = gather_output # True (纵刀流线性层独立使用) or False (作为MLP的第一个线性层使用)
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size() # 当前并行群组中gpu的个数
        self.output_size_per_partition = divide(output_size, world_size) # 每个“划分”上的output的维度大小, 4h/p
        # 即，对A的列的个数，按照gpu的数量，进行切分，然后得到的就是，每个gpu上的列的个数

        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose. 注意，如下的weight，是定义的A_i^T
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization: # args中是"--use-cpu-initialization" (使用CPU初始化参数的值)
            self.weight = Parameter(torch.empty(self.output_size_per_partition, # cai = column of A_i=row of A_i^T = 4h/p
                                                self.input_size, # rx=row of A/X=column of X^T = h
                                                dtype=args.params_dtype)) # float32 or float16
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty( # e.g., torch.Size([3072, 1024])
                self.output_size_per_partition, self.input_size, # cai=column of A_i=row of A_i^T=4h/p, rx=row of A=column of A^T=h
                device=torch.cuda.current_device(), dtype=args.params_dtype)) # device=当前的GPU NOTE this is important!
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)
            
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype)) # TODO this is very important! this code only for current GPU's output.size!
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(), # device=当前的gpu # TODO very important! weight of linear layer is at current GPU's memory!!!
                    dtype=args.params_dtype)) # torch.Size([3072])
            self.bias.tensor_model_parallel = True # TODO for what?
            self.bias.partition_dim = 0
            self.bias.stride = stride # 1, TODO for what?
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)


    def forward(self, input_):
        #import pdb; pdb.set_trace() # column parallel linear
        # Set up backprop all-reduce. 输入的是整体h，输出的是4h/p被按照gpu分割之后的！
        # “forward复制-backward全归约” - 纵刀流的f (column parallel linear layer)
        input_parallel = copy_to_tensor_model_parallel_region(input_) 
        # 前向复制identity，后向全归约all-reduce（论文中的Figure 3.a中的f）
        # 原因：X在多个GPU上被简单复制（没有进行任何切割），类似X -> f -> [X, X, ..., X]
        # 这样的话，反向的时候，就是多个gpu上的X的整体，
        # 通过all_reduce合并到一起（例如相加然后平均，或者直接element-wise相加 -> 这里是使用element-wise相加）
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None # 当不“忽略bias”的时候，带上self.bias
        output_parallel = F.linear(input_parallel, self.weight, bias) 
        # import torch.nn.functional as F
        # X * A_i^T = (b, s, h) * (h, 4h/p) = (b, s, 4h/p) = output_parallel

        if self.gather_output: # 作为[[独立的]]'column parallel linear layer'所需要的g函数，即forward使用all-gather：
            # All-gather across the partitions.
            # “forward拼凑-back切割” - 纵刀流的g (column parallel linear layer)
            output = gather_from_tensor_model_parallel_region(output_parallel) 
            # 前向拼凑，后向切割
            # 前向把各个GPU上的Y_i，进行拼凑，得到Y；即，从(b, s, 4h/p)拼凑到(b, s, 4h)的过程。
            # 后向的时候，把Y按照GPU的数量，进行切割，并送回到各个GPU。即，从(b, s, 4h)切割到(b, s, 4h/p)的过程。
        else:
            output = output_parallel 
            # 当前的'column parallel linear layer'作为GPU并行Transformer中的MLP的一部分的时候的用法：
            # TODO 注意，如果不gather，则实际在这里输出的是4h/p，是一个gpu上的结果！而不是整体的4h!
            # 这个部分非常重要，有助于理解selfattention中的每个tensor的维度！
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism (参数按照行，切分并行). 横刀流

    The linear layer is defined as Y = XA + b （X=输入tensor, A=可训练参数，权重矩阵, b=bias；Y=输出tensor）. 
    A is parallelized along its first dimension and X along its second (final, possibly not second!!!) dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p] (按照最后一列切割的)
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A. [A的行数] 4h
        output_size: second dimension of matrix A. [A的列数] h
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again. [输入已经按照GPUs进行切割完毕了，不再次切割]
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers (? stride=1含义? TODO).
        keep_master_weight_for_test: This was added for testing (TODO okay->mpu/tests/test_layers.py for test only!) and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with (熔合) other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size # 例如，4*hidden_size of transformer = 4*h (MLP的第二个linear layer)
        self.output_size = output_size # 例如，hidden_size of transformer = h
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size() # 当前"张量-并行群组"中gpu的数量, e.g. 2
        self.input_size_per_partition = divide(input_size, world_size) # 类似于从A到A_1, A_2, ..., A_p; 这里是从4h到4h/p
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            # Parameter：将一个不可训练的类型tensor转换成可以训练的类型parameter
            # 并将这个parameter绑定到这个module里面，相当于变成了模型的一部分，成为了模型中可以根据训练进行变化的参数！
            self.weight = Parameter(torch.empty(self.output_size, # h
                                                self.input_size_per_partition, # 对输入，按照“划分”的个数进行了“切割” 4h/p
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test) # forward中没有用到此参数！
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition, # output_size=h, input_size_per_partition=4h/p
                device=torch.cuda.current_device(), dtype=args.params_dtype)) # TODO this is important -> use current GPU to store the parameter weight!
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size, # h, for bias
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_() # in place method
        else:
            self.register_parameter('bias', None)
            # 向我们建立的网络module添加parameter!


    def forward(self, input_):
        #import pdb; pdb.set_trace()
        # Set up backprop all-reduce. 输入的tensor是被分割到每个gpu的，输出的tensor是整体all-reduce之后的！
        if self.input_is_parallel: # Transformer's MLP使用这个部分:
            input_parallel = input_
        else: 
            # 作为[[独立的]]row parallel线性层，使用这个部分：
            # "forward切割-backward拼凑" - 横刀流的f (row parallel linear layer)
            input_parallel = scatter_to_tensor_model_parallel_region(input_) # 前向切割，后向拼凑
            # 如果有必要，先把输入inputs_=X按照?（应该是最后一列，即4h -> 4h/p）切割，按照gpu的数量。
            # 得到的是X_1, ..., X_p这样的，shape是?
            # (batch, seq, 4*hidden/p) 或者(seq, batch, 4*hidden/p)

        # Matrix multiply. [Y1*B1] to [Yi*Bi] in Figure 3(a):
        output_parallel = F.linear(input_parallel, self.weight) # 注意，输入是X_i，而且经历的weight是A_i
        # 从4h/p -> h的linear (是对tensor的最后一列进行变换的！至于mini-batch方面，则是data-parallel的问题，这里不考虑！)
        # 相当于从X_i=(b, s, 4h/p) * A_i^T=(4h/p, h) -> (b, s, h)这样的结果。
        # 可以看到X按照最后一个维度被切割成了p份（gpu的数量); 
        # 然后经过线性层，从4h/p维度，被映射成了h维度。所以，最后的输出是(b, s, h).

        # All-reduce across all the partitions. [g function in Figure 3(a)]
        # “forward全归约-backward复制” - 横刀流的g (row parallel linear layer)
        output_ = reduce_from_tensor_model_parallel_region(output_parallel) # 
        # 前向全归约，后向复制(Figure 3(a) right-hand-side)
        # 对每个gpu上的(b, s, h)进行all_reduce，叠加（求平均），然后得到的是(b, s, h)，每个gpu上面都保持了最新的结果。

        if not self.skip_bias_add: # add:
            output = output_ + self.bias if self.bias is not None else output_ # 把bias加到output_上
            output_bias = None
        else: # not add:
            output = output_
            output_bias = self.bias
        return output, output_bias

