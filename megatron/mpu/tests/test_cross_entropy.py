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

import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../..'))
sys.path.append(os.path.join(script_dir, '../..'))

from commons import set_random_seed
from commons import IdentityLayer
from commons import print_separator
from commons import initialize_distributed
from mpu.cross_entropy import vocab_parallel_cross_entropy
import mpu
import torch.nn.functional as F
import torch
import random
#import sys
#sys.path.append("../..")


def torch_cross_entropy(batch_size, seq_length, vocab_size,
                        logits_scale, seed): # 13, 17, 11, 1000.0, 1234
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda()
    logits = identity() # torch.Size([13, 17, 11])
    target = torch.cuda.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size) # torch.Size([13, 17])
    # (Pdb) p logits[0][0]
    # tensor([  685.4946,  1255.2649,  -143.4313,   188.9241, -1389.1897,   -86.4917,
    #             -491.8454,  -492.7151,  -102.4064,   881.8136,   115.9647],
    #                    device='cuda:0', grad_fn=<SelectBackward>)
    
    # (Pdb) p target
    # tensor([[ 8,  8,  3,  3,  7,  4,  8,  6,  4,  6,  3, 10,  7,  4,  8,  6,  0],
    #     [ 7,  6,  4, 10,  8,  3,  2,  6,  1,  3,  3,  3,  9,  6,  0,  3,  6],
    #     [ 8,  4,  9,  4, 10,  0,  9,  1,  2,  4,  7,  5,  4,  6,  6,  4,  0],
    # ...                     device='cuda:0')


    loss = F.cross_entropy(logits.view(-1, logits.size()[-1]), # torch.Size([221,11])
                           target.view(-1), # torch.Size([221])
                           reduction='none').view_as(target).mean()
    # (Pdb) p loss
    # can be changed due to random (when restarting pdb debug):
    # tensor(1480.4912, device='cuda:0', grad_fn=<MeanBackward0>)
    
    # example:
    # tensor([[ -111.7186,  -496.5901,   163.0737,  -881.6877],
    #        [   53.9002,   668.3737,   -59.6576,  -467.4980],
    #        [  636.9127,  -714.0801, -1083.1244,  -554.7242],
    #        [  971.6850,  -515.0092,  1425.5266,   798.6854],
    #        [-2527.3352,  1477.7843,  -169.6236,  -991.8574],
    #        [-1456.9084,   256.2917,  -403.0470,   419.5274]], device='cuda:0',
    #        grad_fn=<ViewBackward>)
    # (Pdb) p target.view(-1)
    # tensor([0, 0, 3, 0, 0, 0], device='cuda:0')

    # cross entropy 
    # -np.log_e(exp(logits[0][target[0]])/(exp(logits[0][0])+...+exp(logits[0][3])))
    # = -np.log_e(exp(-111.7186)/exp(logits[0]).sum()) = 274.7923
    # 依此类推，可以得到其他的值：
    # tensor([ 274.7923,  614.4735, 1191.6370,  453.8416, 4005.1196, 1876.4358],
    #       device='cuda:0', grad_fn=<NllLossBackward>)
    # so, mean=
    # 1402.7167

    # ---end of scale=1000.0 case---

    # --- start scale=10.0---
    #(Pdb) p logits
    #Parameter containing:
    #    tensor([[[ -1.1172,  -4.9659,   1.6307,  -8.8169],
    #             [  0.5390,   6.6837,  -0.5966,  -4.6750],
    #             [  6.3691,  -7.1408, -10.8312,  -5.5472]],
    #            [[  9.7169,  -5.1501,  14.2553,   7.9869],
    #             [-25.2734,  14.7778,  -1.6962,  -9.9186],
    #             [-14.5691,   2.5629,  -4.0305,   4.1953]]], device='cuda:0',
    #             requires_grad=True)
    #(Pdb) p F.cross_entropy(logits.view(-1, 4), target.view(-1), reduction='none')
    #   tensor([ 2.8113,  6.1476, 11.9164,  4.5509, 40.0512, 18.9431], device='cuda:0',
    #                   grad_fn=<NllLossBackward>)
    #    (Pdb) p target
    #    tensor([[0, 0, 3],
    #            [0, 0, 0]], device='cuda:0')

    # loss=(Pdb) p F.cross_entropy(logits.view(-1, 4), target.view(-1), reduction='none')
    #tensor([ 2.8113,  6.1476, 11.9164,  4.5509, 40.0512, 18.9431], device='cuda:0',
    #               grad_fn=<NllLossBackward>)
    # --- end of scale=10.0---


    loss.backward()
    return loss, identity.weight.grad # loss=scalar value, grad=torch.Size([13, 17, 11])


def mpu_cross_entropy(batch_size, seq_length, vocab_size,
                      logits_scale, seed): # 13, 17, 11, 1000.0, 1234
    set_random_seed(seed)
    identity = IdentityLayer((batch_size, seq_length, vocab_size),
                             scale=logits_scale).cuda()
    logits = identity() # use forward function of 'IdentityLayer'
    # TODO this is the difference:
    logits_parallel = mpu.scatter_to_tensor_model_parallel_region(logits) # 切割-拼凑
    # target generated here (randomly) is same with 'torch_cross_entropy''s target! since used same random seeds:
    target = torch.cuda.LongTensor(
        size=(batch_size, seq_length)).random_(0, vocab_size)
    # TODO this is the difference:
    loss = vocab_parallel_cross_entropy(logits_parallel, target).mean() # take mean -> tensor(1702.7167, device='cuda:0')
    loss.backward()
    return loss, identity.weight.grad # loss=scalar value, grad.shape=torch.Size([13, 17, 11])


def test_cross_entropy(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cross entropy with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 2 # 13
    seq_length = 3 # 17
    vocab_size_per_partition = 4 #11
    logits_scale = 1 #1000.0 TODO 1000.0 is too big for exp and log
    vocab_size = vocab_size_per_partition * tensor_model_parallel_size
    seed = 1234
    # use torch_cross_entropy: method 1
    loss_torch, grad_torch = torch_cross_entropy(batch_size, seq_length,
                                                 vocab_size, logits_scale,
                                                 seed)
    # loss_torch=tensor(1480.4912, device='cuda:0', grad_fn=<MeanBackward0>)
    # grad_torch.shape=torch.Size([13, 17, 11]) 

    # use mpu_cross_entropy: method 1 (distributed)
    loss_mpu, grad_mpu = mpu_cross_entropy(batch_size, seq_length,
                                           vocab_size, logits_scale,
                                           seed) # method 2

    error = loss_torch.sub_(loss_mpu).abs().max()
    print('   max error in loss on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = grad_torch.sub_(grad_mpu).abs().max()
    print('   max error in grad on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    #mpu.destroy_tensor_model_parallel()
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test cross entropy')
        test_cross_entropy(tensor_model_parallel_size)
        tensor_model_parallel_size *= 2

