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
from mpu import layers
from model import transformer
from commons import set_random_seed
from commons import print_separator
from commons import initialize_distributed
import mpu
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch
import random
from megatron import get_args
import traceback


def test_parallel_embedding(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing parallel embedding with model parallel size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    batch_size = 17
    seq_length = 23
    vocab_size = 48
    hidden_size = 16
    seed = 1236

    #set_random_seed(123)
    set_random_seed(seed)
    input_data = torch.LongTensor( # [17, 23], 随机构造输入序列
        size=(batch_size, seq_length)).random_(0, vocab_size).cuda()
    loss_weight = torch.randn([batch_size, seq_length, hidden_size]).cuda()

    set_random_seed(seed)
    embedding_original = torch.nn.Embedding(vocab_size, hidden_size).cuda()

    output = embedding_original(input_data) # (17,23) -> (17, 23, 16)
    loss_original = torch.mul(output, loss_weight).sum() # element-wise production; TODO 'mul' for what?
    loss_original.backward()

    #set_random_seed(seed)
    #embedding_parallel = layers.VocabParallelEmbedding( # TODO  should be ParallelEmbedding(), not VocabParallelEmbedding()!
    #    vocab_size, hidden_size, init_method=init.normal_).cuda()
    #output = embedding_parallel(input_data)
    #loss_parallel = torch.mul(output, loss_weight).sum()
    #loss_parallel.backward()

    set_random_seed(seed)
    embedding_vocab_parallel = layers.VocabParallelEmbedding(
        vocab_size, hidden_size, init_method=init.normal_).cuda()
    output = embedding_vocab_parallel(input_data) # this output is different with line 57's output...! TODO
    loss_vocab_parallel = torch.mul(output, loss_weight).sum()
    loss_vocab_parallel.backward()

    # no loss_parallel now, use only the vocab embedding vector layer for testing:
    #torch.distributed.barrier()
    #error = loss_parallel.sub(loss_original).abs()
    #print('   error in loss (parallel) on global rank {}: {}'.format(
    #    torch.distributed.get_rank(), error))
    #assert error < 1.0e-12, 'error: {}'.format(error)

    torch.distributed.barrier()
    import pdb; pdb.set_trace()
    error = loss_vocab_parallel.sub(loss_original).abs()
    print('   error in loss (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   hidden_size // tensor_model_parallel_size,
                                   1)[mpu.get_tensor_model_parallel_rank()]
    error = embedding_parallel.weight.grad.sub(weight_grad_orig).abs().max()
    print('   error in grad (parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    weight_grad_orig = torch.split(embedding_original.weight.grad,
                                   vocab_size // tensor_model_parallel_size,
                                   0)[mpu.get_tensor_model_parallel_rank()]
    error = embedding_vocab_parallel.weight.grad.sub(
        weight_grad_orig).abs().max()
    print('   error in grad (vocab parallel) on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-12, 'error: {}'.format(error)

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_initialize_affine_weight(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing initialize_affine_weight with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size

    # ---------------
    # Column parallel
    # ---------------
    print('--column parallel--')
    weight = torch.empty(output_size_coeff, input_size) # all zero: [17, 13]
    set_random_seed(seed)
    #layers._initialize_affine_weight(weight, output_size, input_size,
    # 使用normal_来初始化weight! NOTE 
    layers._initialize_affine_weight_cpu(weight, output_size, input_size,

                                     output_size_coeff, 0,
                                     torch.nn.init.normal_) # values in 'weight' now
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size) # all 0
    torch.nn.init.normal_(master_weight) # 使用了相同的random seed! master_weight = weight!! great! use normal distribution mean=0.0, std=1.0 to fill the input tensor
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, output_size_coeff,
                            dim=0)[rank].contiguous().clone()

    # Compare.
    # 1. weight <- layers._initialize_affine_weight_cpu and normal_
    # 2. my_weight <- master_weight <- normal_
    # 通过1和2的结果的比较，就可以测试出来1中使用的函数了
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   column parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # ------------
    # Row parallel
    # ------------
    print('--row parallel--')
    weight = torch.empty(output_size, input_size_coeff)
    set_random_seed(seed)
    mpu.layers._initialize_affine_weight_cpu(weight, output_size, input_size,
                                         input_size_coeff, 1,
                                         torch.nn.init.normal_)
    # Target.
    set_random_seed(seed)
    master_weight = torch.empty(output_size, input_size)
    torch.nn.init.normal_(master_weight)
    rank = mpu.get_tensor_model_parallel_rank()
    my_weight = torch.split(master_weight, input_size_coeff,
                            dim=1)[rank].contiguous().clone()

    # Compare.
    error = weight.sub(my_weight).abs().max()
    torch.distributed.barrier()
    print('   row parallel max error (should be zero) on global rank '
          '{}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer2D(torch.nn.Module):
    def __init__(self, m, n): # m=7, n=13
        super(IdentityLayer2D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n)) # torch.Size([7, 13])
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight # NOTE 直接返回weight的含义？和输入无关的网络？

def test_column_parallel_linear(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing ColumnParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = mpu.ColumnParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()

    # Forward
    input_ = identity_layer() # [7=batch.size, 13=input.size]
    output = linear_layer(input_) # [7=batch.size, 17=output.size]
    loss = torch.mul(output[0], loss_weight).sum()

    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight # [7=batch.size, 17=output.size]
    X = identity_layer.weight # X=input x, A=weight
    A = linear_layer.master_weight.cuda() # [17=output.size, 13=input.size] TODO no master_weight? -> need to set arguments.py's cpu-init parameter to be default true! then ok.
    dLdA = torch.matmul(dLdY.t(), X) # [17,7] * [7, 13] -> [17, 13] TODO stange...
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1) # [17]
    dLdX = torch.matmul(dLdY, A) # [7, 17] * [17, 13] -> [7, 13]

    rank = mpu.get_tensor_model_parallel_rank()
    # simulate dLoss/dweight-A = weight's gradient
    my_dLdA = torch.split(dLdA, output_size_coeff,
                          dim=0)[rank].contiguous().clone()

    # NOTE 1, weight A's gradient comparison:
    #(Pdb) p my_dLdA[0]
    #tensor([-0.4328, -0.0613, -0.9401, -0.8437, -0.2938,  1.3285, -1.1106, -0.1666,
    #             1.0681,  0.0227,  0.4735, -1.7876,  0.1343], device='cuda:0',
    #                    grad_fn=<SelectBackward>)
    #(Pdb) p linear_layer.weight.grad[0]
    #tensor([-0.4328, -0.0613, -0.9401, -0.8437, -0.2938,  1.3285, -1.1106, -0.1666,
    #             1.0681,  0.0227,  0.4735, -1.7876,  0.1343], device='cuda:0')

    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # simulate dLoss/dbias = bias's gradient
    my_dLdb = torch.split(dLdb, output_size_coeff,
                          dim=0)[rank].contiguous().clone()

    # NOTE 2, bias b's gradient comparison:
    #(Pdb) p my_dLdb
    #tensor([-1.4082,  2.0154,  3.9275,  2.7516, -0.5990, -0.0847,  3.5791, -3.7494,
    #             4.2867, -4.6110,  2.3795, -3.9009,  0.0198, -5.3021, -1.4378, -4.5350,
    #                      3.0481], device='cuda:0')
    #(Pdb) p linear_layer.bias.grad
    #tensor([-1.4082,  2.0154,  3.9275,  2.7516, -0.5990, -0.0847,  3.5791, -3.7494,
    #             4.2867, -4.6110,  2.3795, -3.9009,  0.0198, -5.3021, -1.4378, -4.5350,
    #                      3.0481], device='cuda:0')

    error = my_dLdb.sub(linear_layer.bias.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # NOTE 3, input X's gradient comparison:
    #(Pdb) p dLdX[0]
    #tensor([ 0.9437, -0.8477, -1.0318,  0.7537, -1.6996,  0.3988,  0.5533,  0.3265,
    #             0.7638, -1.4018, -0.7501,  0.7525,  0.1920], device='cuda:0')
    #(Pdb) p identity_layer.weight.grad.shape
    #torch.Size([7, 13])
    #(Pdb) p identity_layer.weight.grad[0]
    #tensor([ 0.9437, -0.8477, -1.0318,  0.7537, -1.6996,  0.3988,  0.5533,  0.3265,
    #             0.7638, -1.4018, -0.7501,  0.7525,  0.1920], device='cuda:0')

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def test_row_parallel_linear(tensor_model_parallel_size):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    if torch.distributed.get_rank() == 0:
        print('> testing RowParallelLinear with model parallel '
              'size: {}'.format(tensor_model_parallel_size))
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)
    input_size_coeff = 13
    input_size = input_size_coeff * tensor_model_parallel_size
    output_size_coeff = 17
    output_size = output_size_coeff * tensor_model_parallel_size
    batch_size = 7

    # Network
    identity_layer = IdentityLayer2D(batch_size, input_size).cuda()
    linear_layer = mpu.RowParallelLinear(
        input_size, output_size, keep_master_weight_for_test=True).cuda()
    loss_weight = torch.randn([batch_size, output_size]).cuda()
    # Forward
    input_ = identity_layer()
    output = linear_layer(input_)
    loss = torch.mul(output[0], loss_weight).sum()
    # Backward
    loss.backward()

    # Values.
    dLdY = loss_weight
    X = identity_layer.weight
    A = linear_layer.master_weight.cuda()
    dLdA = torch.matmul(dLdY.t(), X) # [17, 7] * [7, 13] -> [17, 13]
    dLdb = torch.matmul(torch.ones(batch_size, 1).cuda().t(), dLdY).view(-1) # [17]
    dLdX = torch.matmul(dLdY, A) # [7, 17] * [17, 13] -> [7, 13]

    rank = mpu.get_tensor_model_parallel_rank()
    my_dLdA = torch.split(dLdA, input_size_coeff,
                          dim=1)[rank].contiguous().clone()
    error = my_dLdA.sub(linear_layer.weight.grad).abs().max()
    #(Pdb) p my_dLdA.sum(), linear_layer.weight.grad.sum()
    #(tensor(-1.3395, device='cuda:0', grad_fn=<SumBackward0>), tensor(-1.3395, device='cuda:0'))

    torch.distributed.barrier()
    print('   error in dLdA on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdb.sub(linear_layer.bias.grad).abs().max()
    #(Pdb) p dLdb.sum(), linear_layer.bias.grad.sum()
    #(tensor(-3.6205, device='cuda:0'), tensor(-3.6205, device='cuda:0'))

    torch.distributed.barrier()
    print('   error in dLdb on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    error = dLdX.sub(identity_layer.weight.grad).abs().max()
    #(Pdb) p dLdX.sum(), identity_layer.weight.grad.sum()
    #(tensor(-2.7139, device='cuda:0'), tensor(-2.7139, device='cuda:0'))

    torch.distributed.barrier()
    print('   error in dLdX on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


class IdentityLayer3D(torch.nn.Module):
    def __init__(self, m, n, k):
        super(IdentityLayer3D, self).__init__()
        self.weight = Parameter(torch.Tensor(m, n, k))
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self):
        return self.weight


def parallel_self_attention(tensor_model_parallel_size, num_att_heads_per_partition,
                            hidden_size_per_att_head, dropout_prob, batch_size,
                            sequence_length):
    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    #attention_layer = mpu.BertParallelSelfAttention(hidden_size, num_att_heads,
    #import transformer
    args = get_args()
    from megatron.model.bert_model import bert_attention_mask_func
    attention_mask_func = bert_attention_mask_func # TODO

    from megatron.model.utils import init_method_normal, scaled_init_method_normal
    init_method = init_method_normal(args.init_method_std)
    scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)
    layer_number = 1 # TODO what is the meaning of layer_number?
    # mpu.initialize._TENSOR_MODEL_PARALLEL_GROUP is okay until here! NOTE
    #attention_layer = transformer.ParallelSelfAttention(hidden_size, num_att_heads,
    #                                                dropout_prob).cuda()
    attention_layer = transformer.ParallelSelfAttention(attention_mask_func,
            init_method, scaled_init_method, layer_number).cuda()

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    # TODO this attention_mask is not correct!
    #attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    attention_mask = torch.randn([13, 12, 5, 5])#.cuda()
    attention_mask = attention_mask > 0
    attention_mask = attention_mask.type(torch.ByteTensor)
    # Forward
    input_ = identity_layer()
    output = attention_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer


def test_parallel_self_attention(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelSelfAttention with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 12 # 3
    hidden_size_per_att_head = 64 # 7
    dropout_prob = 0.0  # has to be zero
    batch_size = 5
    sequence_length = 13

    rank_1, hideen_size_1, tensor_model_parallel_size_1, loss_1, \
        attention_layer_1, identity_layer_1 = parallel_self_attention(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        attention_layer, identity_layer = parallel_self_attention(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, dropout_prob, batch_size, sequence_length)
    assert hideen_size_1 == hidden_size

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    my_lin_grad_list = torch.split(
        attention_layer_1.query_key_value.weight.grad,
        hidden_size // tensor_model_parallel_size, 0)[rank::tensor_model_parallel_size]
    my_lin_grad = torch.cat(my_lin_grad_list, dim=0)
    error = my_lin_grad.sub(
        attention_layer.query_key_value.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   weight gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-6

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


def parallel_transformer(tensor_model_parallel_size, num_att_heads_per_partition,
                         hidden_size_per_att_head, batch_size, sequence_length):

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed = 12345
    set_random_seed(seed)

    num_att_heads = num_att_heads_per_partition * \
        torch.distributed.get_world_size()
    hidden_size = hidden_size_per_att_head * num_att_heads
    intermediate_size = 4 * hidden_size

    # Network
    identity_layer = IdentityLayer3D(batch_size, sequence_length,
                                     hidden_size).cuda()
    transformer_layer = mpu.BertParallelTransformerLayer(
        hidden_size, intermediate_size, num_att_heads, 0.0, 0.0,
        torch.nn.functional.relu, 1.0e-5).cuda()

    loss_weight = torch.randn([batch_size, sequence_length, hidden_size]).cuda()
    attention_mask = torch.randn([batch_size, 1, 1, sequence_length]).cuda()
    # Forward
    input_ = identity_layer()
    output = transformer_layer(input_, attention_mask)
    loss = torch.mul(output, loss_weight).sum()
    # Backward
    loss.backward()

    rank = mpu.get_tensor_model_parallel_rank()
    mpu.destroy_model_parallel()
    return rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer


def test_parallel_transformer_layer(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing ParallelTransformerLayer with model parallel '
              'size: {}'.format(tensor_model_parallel_size))

    num_att_heads_per_partition = 3
    hidden_size_per_att_head = 7
    batch_size = 5
    sequence_length = 13

    rank_1, hidden_size_1, tensor_model_parallel_size_1, loss_1, \
        transformer_layer_1, identity_layer_1 = parallel_transformer(
            1, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    rank, hidden_size, tensor_model_parallel_size, loss, \
        transformer_layer, identity_layer = parallel_transformer(
            tensor_model_parallel_size, num_att_heads_per_partition,
            hidden_size_per_att_head, batch_size, sequence_length)

    error = loss_1.sub(loss).abs().max()
    torch.distributed.barrier()
    print('   loss error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    error = identity_layer_1.weight.grad.sub(
        identity_layer.weight.grad).abs().max()
    torch.distributed.barrier()
    print('   input gradient error on global rank {}: {}'.format(
        torch.distributed.get_rank(), error))
    assert error < 5.0e-5, 'error: {}'.format(error)

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(' >> passed the test :-)')


if __name__ == '__main__':
    # 固定随机数种子: so that same input will ensure same output:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #args = get_args()

    from megatron.global_vars import set_global_variables
    set_global_variables(extra_args_provider=None,
            args_defaults={},
            ignore_unknown_args=False)

    initialize_distributed()
    world_size = torch.distributed.get_world_size()
    
    if True:
        print_separator('1.test initialize affine weight')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_initialize_affine_weight(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    try:
        print_separator('2.test parallel embedding')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_parallel_embedding(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    except Exception:
        tb = traceback.format_exc()
        print('Error', tb)
    finally:
        mpu.destroy_model_parallel()

    try:
        print_separator('3.test column-parallel linear')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_column_parallel_linear(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    except Exception:
        tb = traceback.format_exc()
        print('Error', tb)
    finally:
        mpu.destroy_model_parallel()

    try:
        print_separator('4.test row-parallel linear')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_row_parallel_linear(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    except Exception:
        tb = traceback.format_exc()
        print('Error', tb)
    finally:
        mpu.destroy_model_parallel()

    try:
        print_separator('5.test parallel self-attention')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_parallel_self_attention(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    except Exception:
        tb = traceback.format_exc()
        print('Error', tb)
    finally:
        mpu.destroy_model_parallel()
    
    try:
        print_separator('6.test parallel transformer')
        tensor_model_parallel_size = 1
        while tensor_model_parallel_size <= world_size:
            test_parallel_transformer_layer(tensor_model_parallel_size)
            tensor_model_parallel_size *= 2
    except Exception:
        tb = traceback.format_exc()
        print('Error', tb)
    finally:
        mpu.destroy_model_parallel()

