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

from .initialize import get_tensor_model_parallel_group
from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .utils import VocabUtility


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target):
        # vocab_parallel_logits = prediction result
        # target = reference target

        # (Pdb) p vocab_parallel_logits/100 is better
        # tensor([[[ -111.7186,  -496.5901,   163.0737,  -881.6877],
        #          [   53.9002,   668.3737,   -59.6576,  -467.4980],
        #          [  636.9127,  -714.0801, -1083.1244,  -554.7242]],
        #         [[  971.6850,  -515.0092,  1425.5266,   798.6854],
        #          [-2527.3352,  1477.7843,  -169.6236,  -991.8574],
        #          [-1456.9084,   256.2917,  -403.0470,   419.5274]]], device='cuda:0',
        #          grad_fn=<_ScatterToModelParallelRegionBackward>)
        # (Pdb) p target
        # tensor([[0, 0, 3],
        #         [0, 0, 0]], device='cuda:0')


        # 最大值：Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0] # 自己gpu上的; take the max value of each length=29056 vector! e.g., from shape of [4, 512, 29056] to [4, 512]
        # (Pdb) p logits_max
        # tensor([[ 1.630737,  6.683737,  6.369127],
        #         [14.255266, 14.777843,  4.195274]], device='cuda:0')

        # no change of logits_max if single gpu:
        torch.distributed.all_reduce(logits_max, # operator=max value of gpus
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=get_tensor_model_parallel_group())

        # 都减去最大值，相对顺序不变：Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1)) # [4, 512, 29056]
        # tensor([[[ -2.747923,  -6.596638,     0.0000, -10.447615],
        #          [ -6.144735,     0.0000,  -7.280313, -11.358717],
        #          [    0.0000, -13.509927, -17.200371, -11.916370]],
        #         [[ -4.538416, -19.405358,     0.0000,  -6.268412],
        #          [-40.051196,     0.0000, -16.474080, -24.696416],
        #          [-18.764358,  -1.632356,  -8.225743,     0.0000]]], device='cuda:0',
        #                                 grad_fn=<AsStridedBackward>)


        # Get the partition's vocab indecies (indices?)
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size

        # 4
        partition_vocab_size = vocab_parallel_logits.size()[-1] # e.g., 29056 

        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked). e.g., target.shape=[4, 512] for the reference target
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # (Pdb) p target_mask
        # tensor([[False, False, False],
        #         [False, False, False]], device='cuda:0')

        masked_target = target.clone() - vocab_start_index

        masked_target[target_mask] = 0
        # same with target

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size) # torch.Size([221, 11]); also alike [4, 512, 29056] -> [2048, 29056]
        # same with vocab_parallel_logits currently (gpu=1)

        masked_target_1d = masked_target.view(-1) # torch.Size([221]); reference target, e.g., from [4, 512] to [2048]
        # tensor([0, 0, 3, 0, 0, 0], device='cuda:0')

        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                device=logits_2d.device) # tensor([0,...,220], device='cuda:0')
        # tensor([0, 1, 2, 3, 4, 5], device='cuda:0')
        
        # TODO 这里已经是只取reference index的predicted prob:
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d] # TODO torch.Size([221])
        # (Pdb) p predicted_logits_1d
        #tensor([ -274.7923,  -614.4735, -1191.6370,  -453.8416, -4005.1196, -1876.4358], device='cuda:0')

        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target) # torch.Size([13,17])
        # (Pdb) p predicted_logits/100 todo to modify by div 100
        #tensor([[ -274.7923,  -614.4735, -1191.6370],
        #        [ -453.8416, -4005.1196, -1876.4358]], device='cuda:0')

        predicted_logits[target_mask] = 0.0
        # target_mask=all False for gpu=1, so this is fine, no change to predicted_logits

        # 多gpu求和：All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(predicted_logits, # operator=sum up
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_tensor_model_parallel_group())

        # 多gpu求和: Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        # (Pdb) p exp_logits
        # tensor([[[0., 0., 1., 0.],
        #          [0., 1., 0., 0.],
        #          [1., 0., 0., 0.]],

        #         [[0., 0., 1., 0.],
        #          [0., 1., 0., 0.],
        #          [0., 0., 0., 1.]]], device='cuda:0', grad_fn=<AsStridedBackward>)

        # when scale=10.0:
        # (Pdb) p exp_logits
        # tensor([[[6.4061e-02, 1.3649e-03, 1.0000e+00, 2.9017e-05],
        #          [2.1447e-03, 1.0000e+00, 6.8897e-04, 1.1667e-05],
        #          [1.0000e+00, 1.3574e-06, 3.3882e-08, 6.6802e-06]],

        #         [[1.0690e-02, 3.7356e-09, 1.0000e+00, 1.8952e-03],
        #          [4.0363e-18, 1.0000e+00, 7.0048e-08, 1.8814e-11],
        #          [7.0916e-09, 1.9547e-01, 2.6767e-04, 1.0000e+00]]], device='cuda:0',
        #                                     grad_fn=<AsStridedBackward>)


        sum_exp_logits = exp_logits.sum(dim=-1) # torch.Size([13, 17])
        # when scale=1000.0:
        # tensor([[1., 1., 1.],
        #         [1., 1., 1.]], device='cuda:0')

        # when scale=10.0:
        # (Pdb) p sum_exp_logits
        # tensor([[1.0655, 1.0028, 1.0000],
        #         [1.0126, 1.0000, 1.1957]], device='cuda:0')


        torch.distributed.all_reduce(sum_exp_logits,
                                     op=torch.distributed.ReduceOp.SUM,
                                     group=get_tensor_model_parallel_group())

        # Loss = log(sum(exp(logits))) - predicted-logit.
        # TODO why 为什么是这两个相减得到loss? 有点奇怪

        # 简单解释：
        # (a,b,c) -> 传统的方法是类似于：先softmax，再-log
        # (-np.log(exp(a)/(exp(a)+exp(b)+exp(c))), 
        #  -np.log(exp(b)/(exp(a)+exp(b)+exp(c))), 
        #  -np.log(exp(c)/(exp(a)+exp(b)+exp(c))), )

        # 现在只是在a,b,c中先取一个最大值，然后让a,b,c都减去这个最大值（防止溢出）:
        # (a-m, b-m, c-m)
        # e^a/(e^a+e^b+e^c) = e^(a-m)/(e^(a-m) + e^(b-m) + e^(c-m))
        # 然后再-log，结果是一样的。
        # 分母部分就是e^(a-m) + ... + e^(c-m)这样的；即log(sum_exp_logits).
        # 而分子就是-predicted_logits
        loss = torch.log(sum_exp_logits) - predicted_logits # torch.Size([13, 17])
        # 第一项log(1)之后，都是0了(只有scale=1000.0的时候，是全0；其他不一定！)：
        #

        # (Pdb) p loss
        # when scale=1000.0:
        # tensor([[ 274.7923,  614.4735, 1191.6370],
        #         [ 453.8416, 4005.1196, 1876.4358]], device='cuda:0')

        # when scale=10.0:
        # p loss:
        # tensor([[ 2.8113,  6.1476, 11.9164],
        #        [ 4.5509, 40.0512, 18.9431]], device='cuda:0')


        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1)) # [13, 17, 11]
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        # TODO, exp_logits=tensor([[[0...,1,0]]]), torch.Size([13, 17, 11])
        # target_mask=tensor([[False,...]], device='cuda:0), torch.Size([13, 17])
        # masked_target_1d=tensor([8,8,...], device='cuda:0'), torch.Size([221])

        return loss
        # shape=torch.Size([13,17]), tensor([[1.0992e+03, ...]], device='cuda:0')

    @staticmethod
    def backward(ctx, grad_output):

        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors

        # All the inputs have softmax as their gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0],
                                 device=grad_2d.device)
        grad_2d[arange_1d, masked_target_1d] -= (
            1.0 - target_mask.view(-1).float())

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """Helper function for the cross entropy."""
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target)

    # (Pdb) p vocab_parallel_logits[0][0]
    # tensor([-111.7186, -496.5901,  163.0737, -881.6877,   53.9002,  668.3737,
    #             -59.6576, -467.4980, -215.2529,  883.9616, -758.4169],
    #                    device='cuda:0', grad_fn=<SelectBackward>)

    # (Pdb) p target
    # tensor([[ 8,  8,  3,  3,  7,  4,  8,  6,  4,  6,  3, 10,  7,  4,  8,  6,  0],
    #    ])
    # device='cuda:0')

