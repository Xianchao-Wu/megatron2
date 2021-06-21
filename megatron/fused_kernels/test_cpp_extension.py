from torch.utils import cpp_extension
import pathlib


buildpath = '/workspace/megatron/megatron2/megatron/fused_kernels/build2' 
srcpath = '/workspace/megatron/megatron2/megatron/fused_kernels/' 
srcpath = pathlib.Path(buildpath).parent.absolute()
buildpath = srcpath/'build2'

cc_flag = []
cc_flag.append('-gencode')
cc_flag.append('arch=compute_80,code=sm_80')

scaled_upper_triang_masked_softmax_cuda = cpp_extension.load(
        name='scaled_masked_softmax_cuda',
        sources=[srcpath/'scaled_masked_softmax.cpp',
            srcpath/'scaled_masked_softmax_cuda.cu'],
        build_directory=buildpath,
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
            '-gencode', 'arch=compute_70,code=sm_70',
            '-U__CUDA_NO_HALF_OPERATORS__',
            '-U__CUDA_NO_HALF_CONVERSIONS__',
            '--expt-relaxed-constexpr',
            '--expt-extended-lambda',
            '--use_fast_math'] + cc_flag)
print('done')
