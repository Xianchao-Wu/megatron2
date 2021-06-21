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

"""Transformer."""

import math
import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import mpu
from .module import MegatronModule
from megatron.checkpointing import get_checkpoint_version
from megatron.model import import_layernorm
from megatron.model.fused_softmax import FusedScaleMaskSoftmax # 融合的=fused
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import openai_gelu, erf_gelu

# flags required to enable [jit fusion kernels, TODO]
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size [隐层维度]
     n: number of attention heads [注意力头的个数]
     p: number of model parallel partitions [模型并行的“划分”的个数]
     np: n/p [注意力头的个数/“模型并行”的划分数, 结果np表示的就是一个“划分”上可以有几个"注意力头"]
     hp: h/p [隐层维度/“模型并行”的划分数，hp表示的是一个“划分”上可以有多少hidden size dimension]
     hn: h/n [隐层维度/注意力头的个数=每个注意力头可以对应多少隐层维度，或者，一个head被表示成多少维度]
     b: batch size [批处理size]
     s: sequence length [序列长度]
     l: number of layers [层数 blocks? multi-head self-attention + point-wise feed-forward network?]
    Transformer takes input of size [s, b, h] (序列长度, 批大小, 隐层维度) and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] (批大小，一个划分上注意力头的个数，序列长度，序列长度) and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s]（批大小，一个划分上注意力头的个数，序列长度，序列长度）.
               masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
"""

class ParallelMLP(MegatronModule):
    """MLP (alike position-wise feed-forward linear layer in original transformer).

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    参考论文中的Figure 3(a)
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()

        # [first linear layer] Project to 4h: (alike position-wise feed-forward layer in transformer)
        self.dense_h_to_4h = mpu.ColumnParallelLinear( # TODO-okay: use mpu's 列并行-linear-layer
            args.hidden_size, # input.size，最大的隐层维度（按照gpu个数分割前的）= Transformer hidden size
            4 * args.hidden_size, # output.size, 4*最大的隐层维度 = 4 * "Transformer hidden size"
            gather_output=False, # TODO-okay why not True? 不需要gather, 即不需要对XA1, ..., XAp进行“前向拼接，后向切割”处理！
            init_method=init_method, # 初始化方法
            skip_bias_add=True)

        self.bias_gelu_fusion = args.bias_gelu_fusion # default=True, enable or disable "bias and gelu fusion"
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # [second linear layer] Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear( # TODO-okay use mpu's 行并行-linear-layer
            4 * args.hidden_size, # 4 * "Transformer hidden size" 最大的隐层维度
            args.hidden_size, # 最大的隐层维度, 'Transformer hidden size'
            input_is_parallel=True,
            init_method=output_layer_init_method, # 输出层的初始化方法
            skip_bias_add=True)
         

    def forward(self, hidden_states):

        # [s, b, 4hp] (序列长度，批大小，4*一个划分上的隐层维度hp)? TODO-okay
        # 这里应该是一个整体的隐层维度大小（所有gpu的整体）[s, b, 4*h] -> NO, is [s, b, 4h/p] for one gpu!
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel) # TODO
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h] (序列长度，皮大小，隐层整体维度？！居然不是hp? TODO-okay) -> 应该不是hp; 应该是4hp -> h
        # RowParallelLinear : 负责对整体的hidden size，按照gpu进行切割处理
        output, output_bias = self.dense_4h_to_h(intermediate_parallel) # is from [s, b, 4hp] to [s, b, h] (入分，出和)
        return output, output_bias


class ParallelSelfAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(self, attention_mask_func, init_method,
                 output_layer_init_method, layer_number): # layer_number=本self attention layer在整体transformer中的第几层(>=1)
        super(ParallelSelfAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        if not mpu.initialize._TENSOR_MODEL_PARALLEL_GROUP:
            tensor_model_parallel_size = 1
            mpu.initialize_model_parallel(tensor_model_parallel_size)

        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling 
        # bool：apply (or not) scale Q * K^T by 1 / layer-number

        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(args.hidden_size, # transformer hidden size; 每个gpu划分的隐层维度大小
                                                    world_size) # h/p
        self.hidden_size_per_attention_head = mpu.divide(
            args.hidden_size, args.num_attention_heads) # h/n
        self.num_attention_heads_per_partition = mpu.divide(
            args.num_attention_heads, world_size) # n/p

        # Strided linear layer. (strided何意？步幅TODO)
        self.query_key_value = mpu.ColumnParallelLinear( # alike q'=Qq, k'=Kk, and v'=Vv
            args.hidden_size, # h
            3 * args.hidden_size, # 3h
            gather_output=False,
            init_method=init_method) # h->3h的列并行Linear网络（细节可以参考ParallelMLP中的h->4h的情况）
        # real case is alike: h -> 3h/p = 3hp

        coeff = None # 系数 coefficient
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head) # sqrt(h/n)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16,
            args.scaled_upper_triang_masked_softmax_fusion,
            args.scaled_masked_softmax_fusion,
            self.attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

        # Output. real case is alike h/p=hp -> h
        self.dense = mpu.RowParallelLinear(
            args.hidden_size, # h
            args.hidden_size, # h
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True) # h->h的行并行Linear

    def _transpose_last_dim(self, mixed_layer, num_splits, num_splits_first):
        """ num_splits=3 for q, k, v;
            num_splits_first=True/False, true for [s,b,num_splits*np*hn] and false for [s,b,np*hn*num_splits]
            3 * n/p * h/n = 3h/p, is only for one gpu!
        """
        input_shape = mixed_layer.size();
        if num_splits_first:
            """[s, b, num_splits * np * hn] 
            -->(view) [s, b, num_splits, np, hn] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (num_splits, self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head) # 后面三个维度分别是(num_splits=3, np, hn)

            mixed_layer = mixed_layer.view(*intermediate_shape) # (s, b, 3, np, hn)
            mixed_layer = mixed_layer.transpose(-2, -3).contiguous() # (s, b, np, 3, hn)
        else:
            """[s, b, np * hn * num_splits] 
            -->(view) [s, b, np, hn, num_splits] 
            -->(tranpose) [s, b, np, num_splits, hn] 
            -->(view) [s, b, np * num_splits * hn] """

            intermediate_shape = input_shape[:-1] +\
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head, num_splits) # 后面三个维度分别是(np, hn, 3)

            mixed_layer = mixed_layer.view(*intermediate_shape) # (s, b, np, hn, 3)
            mixed_layer = mixed_layer.transpose(-1, -2).contiguous() # (s, b, np, 3, hn)
        mixed_layer = mixed_layer.view(*input_shape) # 这样，无论num_splits_first是True/False，结果
        # 都一样了，都是(s, b, np*3*hn)
        # 参照：最初的本方法的输入的时候是(s,b,3*np*hn)，或者(s,b,np*hn*3) --> (s,b,np*3*hn)
        
        return mixed_layer

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)], 
        # np=每个gpu上head的数量, hn=每个head的隐层维度，np*3*hn=3h/p是一个gpu上的！
        # 不应该在这里的啊！这里的应该是3h作为输出的维度 TODO-okay（这里是“总调度”，不是各个gpu上的并行）
        # n/p * 3 * h/n = 3h/p finally (is actually for one gpu's hidden dimension!)
        # 纵刀流：三个linear layer都是纵刀流
        mixed_x_layer, _ = self.query_key_value(hidden_states) # from h to 3h，这里还是整体的变换（非也，是h -> 3h/p）
        # TODO-okay 等待确认，实际输出的是3h/p，也就是说，是一个gpu上的！[确认]

        checkpoint_version = get_checkpoint_version()
        if checkpoint_version is not None:
           if checkpoint_version == 0:
               # [s, b, (3 * np * hn)] --> [s, b, (np * 3 * hn)]
               mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, True)
           elif checkpoint_version == 1.0:
               # [s, b, (np * hn * 3)] --> [s, b, (np * 3 * hn)]
               mixed_x_layer = self._transpose_last_dim(mixed_x_layer, 3, False)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition, # 每个划分中的 注意力head 的数量 = n/p = np
             3 * self.hidden_size_per_attention_head) # 每个 attention head的对应的hidden size = 3 * h/n = 3 * hn
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]， 每个gpu上的head的个数，之后是每个head对应的隐层的维度
        # 也就是说，head的数量，应该>= GPU的数量！
        (query_layer,
         key_layer,
         value_layer) = mpu.split_tensor_along_last_dim(mixed_x_layer, 3) # 按照最后一列维度，三等分

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if layer_past is not None:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer),
                                   key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer),
                                     value_layer), dim=0)
        if get_key_value:
            present = (key_layer, value_layer)


        # ===================================
        # Raw attention scores. [b, np, s, s] 完成的是Q*K^T的运算！
        # ===================================
        
        # [b, np, sq, sk]
        output_size = (query_layer.size(1), 
                       query_layer.size(2), 
                       query_layer.size(0), # sq = sequench length of q
                       key_layer.size(0)) # sk = sequence length of k
        
        # [sq, b, np, hn] -> [sq, b * np, hn] # batch * num_head_per_gpu -> 相乘之后，类似于组成了新的batch!
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1) # [sq, b*np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1) # [sk, b*np, hn]

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0]*output_size[1], 
            output_size[2], 
            output_size[3],
            dtype=query_layer.dtype, 
            device=torch.cuda.current_device())

        # Raw attention scores. matmul_result's shape=[b*np, sq, sk]
        matmul_result = torch.baddbmm(matmul_result, 
            query_layer.transpose(0, 1),   # from [sq, b*np, hn] to [b*np, sq, hn]
            key_layer.transpose(0,1).transpose(1, 2),  # from [sk, b*np, hn] to [b*np, sk, hn] and to  [b*np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))
        # 得到的结果是: [b*np, sq, hn] * [b*np, hn, sk] -> [b*np, sq, sk]

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)


        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if get_key_value:
            with torch.no_grad():
                if layer_past is not None:
                    attention_mask = attention_mask[
                        ...,
                        attention_scores.size(3) - 1,
                        :attention_scores.size(3)].unsqueeze(2)
                else:
                    attention_mask = attention_mask[
                        ...,
                        :attention_scores.size(3),
                        :attention_scores.size(3)]


        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.# TODO where? to confirm?
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs) 


        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1), 
                       value_layer.size(2), 
                       query_layer.size(0), 
                       value_layer.size(3)) 

        # change view [sk, b * np, hn] 
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)
        
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
        
        # matmul: [b*np, sq, sk] * [b*np, sk, hn] -> [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0,1)) 
        # TODO-okay important , X*V^T
        # where X = softmax(Q*K^T/scalar_for_scale)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp=np*hn=n/p*h/n=h/p]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)


        # =================
        # Output. [sq, b, h], sq=sequence length of q, b=batch size, h=hidden size
        # =================

        # 也就是说，这里扔给self.dense的时候的, context_layer是已经被切割之后的了！所以其最后一个维度是hp=h/p，而不是h!
        output, bias = self.dense(context_layer) # h-> h 的线性映射，RowParallelLinear
        # 真实情况是h/p作为输入的，大概明白了

        if get_key_value:
            output = [output, present]

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training) :
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add # 返回的是方法


@torch.jit.script
def bias_dropout_add_fused_train(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x, bias, residual, prob) :
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    return bias_dropout_add(x, bias, residual, prob, False)


class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformore layer takes input with size [b, s, h] and returns an
    output of the same size.
    b=batch size; 
    s=seq length;
    h=hidden size.
    """

    def __init__(self, attention_mask_func, init_method, 
                 output_layer_init_method, layer_number):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        # Layernorm on the input data.
        LayerNorm = import_layernorm(args.fp32_residual_connection) # from megatron.model import import_layernorm
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # Self attention.
        self.attention = ParallelSelfAttention(attention_mask_func, init_method,
                                               output_layer_init_method,
                                               layer_number)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion

        # Layernorm on the input data.
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon)

        # MLP, h->4h->h
        self.mlp = ParallelMLP(init_method,
                               output_layer_init_method)

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):
        # hidden_states: [b, s, h]
        # x -> [self.layernorm] -> x1 -> self.attention -> x2 -> bias_dropout_add_residual -> x3 -> self.mlp -> x4
        # -> bias_dropout_add_residual -> x5
        # 基本符合，一个transformer block: 先是self-attention，之后是h->4h->h的MLP这样的两个子块
        # 其中还有基于layer.norm和residual的变换和残差求和。

        # Layer norm at the begining of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output, attention_bias = \
            self.attention(layernorm_output,
                           attention_mask,
                           layer_past=layer_past,
                           get_key_value=get_key_value)

        if get_key_value:
            attention_output, presents = attention_output
    
        # Residual connection.
        if self.apply_residual_connection_post_layernorm: # layernorm之后是残差链接(bert的原来的做法)
            residual = layernorm_output # 用的是layernorm之后的结果作为x
        else:
            residual = hidden_states # 原来的输入x

        # jit scripting for a nn.module (with dropout) is not 
        # trigerring the fusion kernel. For now, we use two 
        # different nn.functional routines to account for varying
        # dropout semantics during training and inference phases.
        if self.bias_dropout_fusion:
            if self.training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(self.training)

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        #re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            output = bias_dropout_add_func(
                mlp_output,
                mlp_bias.expand_as(residual),
                residual,
                self.hidden_dropout)

        if get_key_value:
            output = [output, presents]

        return output


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, attention_mask_func,
                 init_method, output_layer_init_method):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.fp32_residual_connection = args.fp32_residual_connection

        # Store activation checkpoiting flag.
        self.checkpoint_activations = args.checkpoint_activations
        self.checkpoint_num_layers = args.checkpoint_num_layers

        # Number of layers.
        assert args.num_layers % mpu.get_pipeline_model_parallel_world_size() == 0, \
            'num_layers must be divisible by pipeline_model_parallel_size'
        self.num_layers = args.num_layers // mpu.get_pipeline_model_parallel_world_size()

        # Transformer layers.
        def build_layer(layer_number):
            return ParallelTransformerLayer(
                attention_mask_func, init_method,
                output_layer_init_method, layer_number)
        offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if mpu.is_pipeline_last_stage():
            # Final layer norm before output.
            LayerNorm = import_layernorm(args.fp32_residual_connection) # from megatron.model import import_layernorm
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask):
        """Forward method with activation checkpointing."""
        def custom(start, end):
            def custom_forward(*inputs):
                x_ = inputs[0]
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        # Make sure memory is freed/被释放.
        mpu.reset_checkpointed_activations_memory_buffer()
        l = 0
        while l < self.num_layers:
            hidden_states = mpu.checkpoint(
                custom(l, l + self.checkpoint_num_layers),
                hidden_states, attention_mask)
            l += self.checkpoint_num_layers

        return hidden_states

    def forward(self, hidden_states, attention_mask, layer_past=None,
                get_key_value=False):

        # Checks.
        if layer_past is not None:
            assert get_key_value, \
                'for not None values in layer_past, ' \
                'expected get_key_value to be set'
        if get_key_value:
            assert not self.checkpoint_activations, \
                'get_key_value does not work with ' \
                'activation checkpointing'

        if mpu.is_pipeline_first_stage():
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            # If the input flag for fp32 residual connection is set, convert for float.
            if self.fp32_residual_connection:
                hidden_states = hidden_states.transpose(0, 1).contiguous().float() 
            # possibly from fp16 to fp32
            # Otherwise, leave it as is.
            else:
                hidden_states = hidden_states.transpose(0, 1).contiguous()

        if self.checkpoint_activations: 
            # 另外一种模型并行的方法 (Chen, T., Xu, B., Zhang, C., and Guestrin, C. Training
            # deep nets with sublinear memory cost. CoRR,
            # abs/1604.06174, 2016. URL http://arxiv:org/abs/1604:06174.)
            hidden_states = self._checkpointed_forward(hidden_states,
                                                       attention_mask)
        else:
            if get_key_value:
                presents = []
            for index in range(self.num_layers):
                layer = self._get_layer(index) # return self.layers[layer_number]
                past = None
                if layer_past is not None:
                    past = layer_past[index]

                ### 调用ParallelTransformerLayer的forward函数：###
                hidden_states = layer(hidden_states,
                                      attention_mask,
                                      layer_past=past,
                                      get_key_value=get_key_value)
                if get_key_value:
                    hidden_states, present = hidden_states
                    presents.append(present)
        
        # Final layer norm.
        if mpu.is_pipeline_last_stage(): 
            # 即当前gpu的rank = world_size - 1! 也就是说当前的gpu是"gpu并行群组"的最后一个
            # defined in megatron/mpu/initialize.py
            # Reverting data format change [s b h] --> [b s h].
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            output = self.final_layernorm(hidden_states)
        else:
            output = hidden_states
        if get_key_value:
            output = [output, presents]

        return output
