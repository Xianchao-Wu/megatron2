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

import pathlib
import subprocess
import os
from torch.utils import cpp_extension

# Setting this param to a list has a problem of generating
# different compilation commands (with diferent order of architectures)
# and leading to recompilation of fused kernels.
# set it to empty string to avoid recompilation
# and assign arch flags explicity in extra_cuda_cflags below
os.environ["TORCH_CUDA_ARCH_LIST"] = ""

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)# raw_output = 'nvcc: NVIDIA (R) Cuda compiler driver\nCopyright (c) 2005-2019 NVIDIA Corporation\nBuilt on Fri_Feb__8_19:08:26_Pacific_Standard_Time_2019\nCuda compilation tools, release 10.1, V10.1.105\n' <- '/usr/local/cuda/bin/nvcc -V
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0] # release = ['10', '1,'] cuda10.1 in win10; or ['11', '1'] in dgx-1 10.19.60.52 machine! great!
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor # 'nvcc:...", '10', '1' -> win10; and 'nvcc:...', '11', '1' -> linux

def create_build_dir(buildpath):
    try:
        os.mkdir(buildpath)
    except OSError:
        if not os.path.isdir(buildpath):
            print(f"Creation of the build directory {buildpath} failed")

def load_scaled_upper_triang_masked_softmax_fusion_kernel():

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'

    create_build_dir(buildpath)

    scaled_upper_triang_masked_softmax_cuda = cpp_extension.load(
        name='scaled_upper_triang_masked_softmax_cuda',
        sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
                 srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu'],
        build_directory=buildpath,
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '-U__CUDA_NO_HALF_OPERATORS__',
                           '-U__CUDA_NO_HALF_CONVERSIONS__',
                           '--expt-relaxed-constexpr',
                           '--expt-extended-lambda',
                           '--use_fast_math'] + cc_flag)

def load_scaled_masked_softmax_fusion_kernel():
    # TODO for what?
    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME) # CUDA_HOME='/usr/local/cuda', '10'
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')
    # srcpath = WindowsPath('C:/Users/user/source/repos/megatron/megatron/megatron/fused_kernels')
    srcpath = pathlib.Path(__file__).parent.absolute() # __file__ = 'C:\\Users\\user\\source\\repos\\megatron\\megatron\\megatron\\fused_kernels\\__init__.py' -> '/workspace/megatron/megatron2/megatron/fused_kernels'
    buildpath = srcpath / 'build' # buildpath = WindowsPath('C:/Users/user/source/repos/megatron/megatron/megatron/fused_kernels/build')

    create_build_dir(buildpath)

    scaled_upper_triang_masked_softmax_cuda = cpp_extension.load(
        name='scaled_masked_softmax_cuda',
        sources=[srcpath / 'scaled_masked_softmax.cpp',
                 srcpath / 'scaled_masked_softmax_cuda.cu'],
        build_directory=buildpath,
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '-U__CUDA_NO_HALF_OPERATORS__',
                           '-U__CUDA_NO_HALF_CONVERSIONS__',
                           '--expt-relaxed-constexpr',
                           '--expt-extended-lambda',
                           '--use_fast_math'] + cc_flag) # 这部分和cuda编程相关了， TODO ninja总是搞不定... where cl是可以的


def load_fused_mix_prec_layer_norm_kernel():

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_major, _ = get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    if int(bare_metal_major) >= 11:
        cc_flag.append('-gencode')
        cc_flag.append('arch=compute_80,code=sm_80')

    srcpath = pathlib.Path(__file__).parent.absolute()
    buildpath = srcpath / 'build'

    create_build_dir(buildpath)

    fused_mix_prec_layer_norm_cuda = cpp_extension.load(
        name='fused_mix_prec_layer_norm_cuda',
        sources=[srcpath / 'layer_norm_cuda.cpp',
                 srcpath / 'layer_norm_cuda_kernel.cu'],
        build_directory=buildpath,
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3',
                           '-gencode', 'arch=compute_70,code=sm_70',
                           '-maxrregcount=50',
                           '--use_fast_math'] + cc_flag)
