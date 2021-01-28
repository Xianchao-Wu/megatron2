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

from .package_info import (
    __description__,
    __contact_names__,
    __url__,
    __download_url__,
    __keywords__,
    __license__,
    __package_name__,
    __version__,
)

from .global_vars import get_args
from .global_vars import get_current_global_batch_size
from .global_vars import get_num_microbatches
from .global_vars import update_num_microbatches
from .global_vars import get_tokenizer
from .global_vars import get_tensorboard_writer
from .global_vars import get_adlr_autoresume
from .global_vars import get_timers
from .initialize  import initialize_megatron

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0. (first gpu)"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
            # flush参数主要是刷新， 默认flush = False，不刷新，如上面例子，
            # print到f中的内容先存到内存中，当文件对象关闭时才把内容输出到 123.txt 中；
            # 而当flush = True时它会立即把内容刷新存到 123.txt 中。
    else:
        print(message, flush=True)

def is_last_rank():
    """check if current GPU is the gpu with the max rank (final indexed GPU)"""
    return torch.distributed.get_rank() == (
        torch.distributed.get_world_size() - 1)

def print_rank_last(message):
    """If distributed is initialized, print only on last rank. (last gpu)"""
    if torch.distributed.is_initialized():
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)
