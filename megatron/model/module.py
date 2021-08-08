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

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from megatron import get_args
from megatron import mpu


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining (pipeline并行化)."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        self.share_word_embeddings = share_word_embeddings # True=共享; False=不共享


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """Use this function to override the state dict for
        saving checkpoints.
        Arguments:
            destination : (None)
            prefix : ('')
            keep_vars : (False)
        """
        return self.state_dict(destination, prefix, keep_vars)
        # 用于保存到文件的状态词典(包含训练好的参数)


    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage():
            return self.language_model.embedding.word_embeddings.weight
            # TODO 没有看到self.language_model啊！哪里定义了？
        if mpu.is_pipeline_last_stage():
            if not self.share_word_embeddings:
                raise Exception('word_embeddings_weight() called for last '
                                'stage, but share_word_embeddings is false')
            return self.word_embeddings.weight
        raise Exception('word_embeddings_weight() should be '
                        'called for first and last stage only')


    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        if not self.share_word_embeddings:
            raise Exception('initialize_word_embeddings() was called but '
                            'share_word_embeddings is false')
        # Parameters are shared between the word embeddings layer, and the
        # heads (TODO what is head?) at the end of the model. In a pipelined setup with more than
        # one stage (TODO 状态是啥?), the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        #import pdb; pdb.set_trace()
        if mpu.is_pipeline_last_stage():
            if not mpu.is_pipeline_first_stage():
                self._word_embeddings_for_head_key = 'word_embeddings_for_head'
                # If first and last stages are different, set word_embeddings
                # weights to 0 here, then copy first stage's weights using
                # all_reduce below.
                self.word_embeddings = mpu.VocabParallelEmbedding(
                    args.padded_vocab_size, args.hidden_size,
                    init_method=init_method_normal(args.init_method_std))
                self.word_embeddings.weight.data.fill_(0)
                self.word_embeddings.weight.shared = True
        # Ensure that first and last stages have the same initial parameter
        # values.
        # TODO 背后的考量是什么？
        # 已知的是：first_stage()和last_stage()构成了当前进程所在的 embedding group!

        #if torch.distributed.get_rank() == 0:
        #    import pdb; pdb.set_trace()
        if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage(): # TODO important for, first stage and last stage -> all_reduce!
            torch.distributed.all_reduce(self.word_embeddings_weight().data,
                                         group=mpu.get_embedding_group())



def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""
    if not isinstance(val, (tuple, list)): # 递归调用的出口
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val] # 递归调用，一层展开
    if isinstance(val, tuple): # 最初的输入val是“元组”的时候：
        rtn = tuple(rtn)
    return rtn # 最初的输入val是list的时候。


def fp32_to_fp16(val):
    """Convert fp32 `val` to fp16"""
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES): 
            # _FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
            val = val.half()
        return val
    return conversion_helper(val, half_conversion) # 递归的把val中所有数值转换为fp16!


def fp16_to_fp32(val):
    """Convert fp16 `val` to fp32"""
    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _HALF_TYPES):
            # _HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
            val = val.float()
        return val
    return conversion_helper(val, float_conversion) # 递归的把val中所有的数值转换为fp32



class FP16Module(MegatronModule):
    # 有被用到：
    # megatron/training.py:        model = FP16Module(model)

    def __init__(self, module):
        super(FP16Module, self).__init__()
        self.add_module('module', module.half())
        # nn.Module中的half()方法将模型中的float32转化为float16，
        # 实现的原理是遍历所有tensor，而float32和float16都是tensor的属性。
        # 也就是说，一行代码解决.

    def forward(self, *inputs, **kwargs):
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_fp16(inputs)
        outputs = self.module(*inputs, **kwargs) # loss.shape=[4, 512] and binary.head.shape=[4,2]
        if mpu.is_pipeline_last_stage():
            outputs = fp16_to_fp32(outputs)
        return outputs


    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        return self.module.state_dict_for_save_checkpoint(destination, prefix,
                                                          keep_vars)


    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
