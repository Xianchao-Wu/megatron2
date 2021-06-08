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

from commons import print_separator
from commons import initialize_distributed
import mpu
import torch
#import sys
#sys.path.append("../..")


def test_set_cuda_rng_state(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing set_rng_state with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    size = 123
    seed = 1234
    torch.cuda.manual_seed(1234)
    tensor = torch.cuda.FloatTensor(size) # all zero with torch.Size([123])

    # Get the state (of curreng gpu)
    rng_state = torch.cuda.get_rng_state() # torch.Size([816]), returns the random number generator state of the current GPU, as a ByteTensor
    rng_state_copy = rng_state.clone()

    # Do some stuff.
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_1 = tensor.clone() # [123]

    assert rng_state.sub(rng_state_copy).max() == 0 # tensor(0, dtype=torch.uint8)
    assert torch.cuda.get_rng_state().sub(rng_state_copy).max() > 0 # 20, TODO ?? 即新get的generator state和之前的rng_state_copy应该是不一样的！

    # State should be different. 两次调用torch.cuda.get_rng_state()得到的结果应该是不一样的, 即两次的用来产生随机数的generator's state是不一样的-》：
    new_rng_state = torch.cuda.get_rng_state()
    max_diff = new_rng_state.sub(rng_state).max()
    print('   max diff in rng state (should be non-zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), max_diff))
    assert max_diff > 0

    # Reset the rng state and do the same stuff.
    mpu.random._set_cuda_rng_state(rng_state) # 设置为rng_state这个指定的state
    for _ in range(5):
        torch.randn(size, out=tensor)
    mpu.random._set_cuda_rng_state(rng_state) # 设置为rng_state这个指定的state
    for _ in range(5):
        torch.randn(size, out=tensor)
    result_2 = tensor.clone()

    # Results should be the same (since they used the same generator state), 大概明白了，对一个gpu的generator state重新设置一个state value之后，期望其产出的random numbers和其他gpu上使用相同的generator state得到的random numbers是一样的！
    error = result_2.sub(result_1).abs().max()
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Input state should have remained intact-完整的.
    error = rng_state.sub(rng_state_copy).max()
    print('   max error in rng state (should be zero) on global rank {}: {}'.
          format(torch.distributed.get_rank(), error))
    assert error == 0

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_cuda_rng_tracker(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing cuda rng tracker with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    seed_1 = 1234
    seed_2 = 4321
    size = [12, 21]
    tensor = torch.cuda.FloatTensor(size)

    # Set to seed_1 and generate two tensors.
    torch.cuda.manual_seed(seed_1)
    torch.randn(size, out=tensor)
    target_11 = tensor.clone()
    torch.randn(size, out=tensor)
    target_12 = tensor.clone()

    # Set to seed_2 and generate two tensors.
    torch.cuda.manual_seed(seed_2)
    torch.randn(size, out=tensor)
    target_21 = tensor.clone()
    torch.randn(size, out=tensor)
    target_22 = tensor.clone()

    # Now if we interleave-交错 seed_1 and seed_2,
    # we should still get the same tensors
    torch.cuda.manual_seed(seed_1)
    mpu.get_cuda_rng_tracker().add('test', seed_2)

    torch.randn(size, out=tensor)
    result_11 = tensor.clone()

    with mpu.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_21 = tensor.clone()

    torch.randn(size, out=tensor)
    result_12 = tensor.clone()

    with mpu.get_cuda_rng_tracker().fork('test'):
        torch.randn(size, out=tensor)
        result_22 = tensor.clone()

    # result_11/12 : seed_1; vs. result_21/22 : seed_2
    diff = result_11.sub(result_21).abs().max()
    diff = min(diff, result_12.sub(result_22).abs().max())
    print('   max diff in generated tensors (should be non-zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), diff))
    assert diff > 1.0e-6

    # result_11 and target_11 are all from seed_1
    # result_12 and target_12 are all from seed_1
    error = max(result_11.sub(target_11).abs().max(),
                result_12.sub(target_12).abs().max())

    # result_21 and target_21 are all from seed_2
    # result_22 and target_22 are all from seed_2
    error = max(error, result_21.sub(target_21).abs().max())
    error = max(error, result_22.sub(target_22).abs().max())
    print('   max error in generated tensors (should be zero) on '
          'global rank {}: {}'.format(torch.distributed.get_rank(), error))
    assert error < 1.0e-6

    # Reset the tracker
    mpu.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


def test_model_parallel_cuda_manual_seed(tensor_model_parallel_size):

    if torch.distributed.get_rank() == 0:
        print('> testing model parallel cuda manual seed with size {} ...'.
              format(tensor_model_parallel_size))

    mpu.initialize_model_parallel(tensor_model_parallel_size)
    tensor_model_parallel_size = mpu.get_tensor_model_parallel_world_size()

    mpu.model_parallel_cuda_manual_seed(12345)
    assert torch.cuda.initial_seed() == 12345
    with mpu.get_cuda_rng_tracker().fork():
        # 在这里面，是torch.cuda.initial_seed()=15063
        # 出了with之外，是torch.cuda.initial_seed()=12345
        assert torch.cuda.initial_seed() == (12345 + 2718 +
                                             mpu.get_tensor_model_parallel_rank())

    # Reset the tracker
    mpu.get_cuda_rng_tracker().reset()

    # Reset groups
    mpu.destroy_model_parallel()

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print('>> passed the test :-)')


if __name__ == '__main__':

    initialize_distributed()
    world_size = torch.distributed.get_world_size()

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test set rng state')
        test_set_cuda_rng_state(tensor_model_parallel_size) # TODO
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test cuda rng tracker')
        test_cuda_rng_tracker(tensor_model_parallel_size) # TODO
        tensor_model_parallel_size *= 2

    tensor_model_parallel_size = 1
    while tensor_model_parallel_size <= world_size:
        print_separator('test model parallel cuda manual seed')
        test_model_parallel_cuda_manual_seed(tensor_model_parallel_size) # TODO
        tensor_model_parallel_size *= 2

