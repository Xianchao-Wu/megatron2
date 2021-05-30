import pathlib
from torch.utils import cpp_extension

def compile():

    srcpath = pathlib.Path(__file__).parent.absolute() 
    # __file__ = 'C:\\Users\\user\\source\\repos\\megatron\\megatron\\megatron\\fused_kernels\\__init__.py'

    buildpath = srcpath / 'build' 
    # buildpath = WindowsPath('C:/Users/user/source/repos/megatron/megatron/megatron/fused_kernels/build')

    #create_build_dir(buildpath)

    cc_flag = []

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
                           '--use_fast_math'] + cc_flag) # 这部分和cuda编程相关了， TODO

compile()
print('done')
