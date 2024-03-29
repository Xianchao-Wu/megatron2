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

"""Megatron arguments."""

import argparse
import os

import torch
from megatron import fused_kernels

def parse_args(extra_args_provider=None, defaults={},
               ignore_unknown_args=False):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='Megatron-LM Arguments',
                                     allow_abbrev=False)

    # Standard arguments.
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_initialization_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_validation_args(parser)
    parser = _add_data_args(parser)
    parser = _add_autoresume_args(parser)
    parser = _add_realm_args(parser)

    # Custom arguments.
    if extra_args_provider is not None:
        parser = extra_args_provider(parser)

    # Parse.
    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    # Distributed args.
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # Tensor model parallel size.
    args.tensor_model_parallel_size = min(
        args.tensor_model_parallel_size, args.world_size)
    assert args.world_size % args.tensor_model_parallel_size == 0, 'world size'\
        ' ({}) is not divisible by tensor model parallel size ({})'.format(
            args.world_size, args.tensor_model_parallel_size)

    # Pipeline model parallel size.
    args.pipeline_model_parallel_size = min(
        args.pipeline_model_parallel_size,
        (args.world_size // args.tensor_model_parallel_size))

    # Checks.
    model_parallel_size = args.pipeline_model_parallel_size * \
                          args.tensor_model_parallel_size
    assert args.world_size % model_parallel_size == 0, 'world size is not'\
        ' divisible by tensor parallel size ({}) times pipeline paralle ' \
        'size ({})'.format(args.world_size, args.tensor_model_parallel_size,
                           args.pipeline_model_parallel_size)
    args.data_parallel_size = args.world_size // model_parallel_size # TODO important! 例如有16个gpu，tensor_parallel_size=2, pipeline_parallel_size=8, so model_parallel_size=8; and then data_parallel_size=2 -> 一个完整的模型，占用了8个gpu；所以，16个gpu，一共有2个完整的models的copy；再往下就是data parallel size=2，即 data_parallel_size=number of data parallel groups? = number of complete models (2)
    if args.rank == 0:
        print('using world size: {}, data-parallel-size: {}, '
              'tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {} '.format(
                  args.world_size, args.data_parallel_size,
                  args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size), flush=True)

    # Deprecated arguments
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        # TODO 为什么是这样计算global_batch_size呢？
        # e.g., micro_batch_size=4, data_parallel_size=8
        # single-node的时候，类似于8 gpus, 每个gpu上的batch size=4? 道理：micro_batch_size=4，即一个模型片段上是4个sequences。另外，args.data_parallel_size = 完整的模型的个数，所以可以计算出来global_batch_size=4 * 2(e.g., 16 GPUs and each model is used by 8 gpus = 2 complete models)
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        args.params_dtype = torch.half
    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    # Consumed tokens.
    args.consumed_train_samples = 0 # 不是从命令行得到的值！ 新给args加了一个属性和值
    args.consumed_valid_samples = 0 # 取值无法从命令行获得！

    # Set input defaults.
    for key in defaults: # {'tokenizer_type': 'BertWordPieceLowerCase'}
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    # 问题：下面的iteration-based training和sample-based training是什么区别？
    # Iteration-based training.
    if args.train_iters: # 例如, train_iters=1,000,000
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training' # 倾斜升温，上升
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-samples'

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                     'max_position_embeddings']
    # e.g., num_layers=24, hidden_size=1024, num_attention_heads=16, max_position_embeddings=512 or 1024
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    assert args.hidden_size % args.num_attention_heads == 0 # 1024/16=64; or, 768/12=64
    if args.seq_length is not None:
        assert args.max_position_embeddings >= args.seq_length
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None

    # Mixed precision checks. [混合精度检查]
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16, \
            'residual connection in fp32 only supported when using fp16.'
    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work, you '\
            'need to enable checkpoint-activations'

    if args.scaled_masked_softmax_fusion:
        # TODO for what?
        if args.scaled_upper_triang_masked_softmax_fusion:
            fused_kernels.load_scaled_upper_triang_masked_softmax_fusion_kernel()
        else:
            fused_kernels.load_scaled_masked_softmax_fusion_kernel() # something wrong here: C:\Users\user\anaconda3\envs\pytorch36\lib\site-packages\torch\utils\cpp_extension.py:287: UserWarning: Error checking compiler version for cl: [WinError 2] The system cannot find the file specified;   warnings.warn('Error checking compiler version for {}: {}'.format(compiler, error)) INFO: Could not find files for the given pattern(s).
    else:
        # This argument will eventually go away, for now make sure it is off
        # if scaled_masked_softmax_fusion is off.
        args.scaled_upper_triang_masked_softmax_fusion = False

    # Load mixed precision fused layer norm.
    if args.fp32_residual_connection:
        fused_kernels.load_fused_mix_prec_layer_norm_kernel()

    _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('------------------------ arguments ------------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()): # 按照字母顺序打印
            print(arg, flush=True)
        print('-------------------- end of arguments ---------------------',
              flush=True)


def _check_arg_is_not_none(args, arg):
    assert getattr(args, arg) is not None, '{} argument is None'.format(arg)


def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--num-layers', type=int, default=12, #None,
                       help='Number of transformer layers.')
    group.add_argument('--hidden-size', type=int, default=768, #None,
                       help='Tansformer hidden size.')
    group.add_argument('--num-attention-heads', type=int, default=12, #None,
                       help='Number of transformer attention heads.')
    group.add_argument('--max-position-embeddings', type=int, default=128, #None,
                       help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--apply-residual-connection-post-layernorm', # TODO important!
                       action='store_true',
                       help='If set, use original BERT residula connection '
                       'ordering.')
    # BERT中是先residual connection再layer norm? -> 是Add & Norm，先residual add-up，再layer.norm!
    group.add_argument('--openai-gelu', action='store_true',
                       help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility' # 兼容性
                       'reasons.')
    group.add_argument('--onnx-safe', type=bool, required=False,
                       help='Use workarounds for known problems with Torch ONNX exporter')

    return parser


def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--attention-dropout', type=float, default=0.1,
                       help='Post attention dropout probability.')
    group.add_argument('--hidden-dropout', type=float, default=0.1,
                       help='Dropout probability for hidden state transformer.')
    group.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay coefficient for L2 regularization.')
    group.add_argument('--clip-grad', type=float, default=1.0,
                       help='Gradient clipping based on global L2 norm.')
    group.add_argument('--adam-beta1', type=float, default=0.9,
                       help='First coefficient for computing running averages of'
                       'gradient and its square')
    group.add_argument('--adam-beta2', type=float, default=0.999,
                       help='Second coefficient for computing running averages of'
                       'gradient and its square')
    group.add_argument('--adam-eps', type=float, default=1e-08,
                       help='Term added to the denominator to improve' # 分母上加adam-epsilon
                       'numerical stability')

    return parser


def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--micro-batch-size', type=int, default=2, #None,
                       help='Batch size per model instance (local batch size). ' # 每个模型（整体）的batch size
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.')
    # global batch size = 每个模型的batch size (micro-batch-size, 会根据pipeline-group-size再次切割?! TODO ) * data parallel size （TODO ） * 模型的个数
    group.add_argument('--batch-size', type=int, default=None,
                       help='Old batch size parameter, do not use. '
                       'Use --micro-batch-size instead')
    group.add_argument('--global-batch-size', type=int, default=4, #None,
                       help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')
    group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up (斜升) with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    group.add_argument('--checkpoint-activations', action='store_true', # TODO detailed logic?
                       help='Checkpoint activation to allow for training '
                       'with larger models, sequences, and batch sizes.')
    group.add_argument('--distribute-checkpointed-activations', # TODO
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1, # TODO
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-iters', type=int, default=10, #None,
                       help='Total number of iterations to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--train-samples', type=int, default=None,
                       help='Total number of samples to train over all '
                       'training runs. Note that either train-iters or '
                       'train-samples should be provided.')
    group.add_argument('--log-interval', type=int, default=5, #100,
                       help='Report loss and timing interval.')
    group.add_argument('--exit-interval', type=int, default=None,
                       help='Exit the program after the iteration is divisible '
                       'by this value.')
    group.add_argument('--exit-duration-in-mins', type=int, default=None,
                       help='Exit the program after this many minutes.')
    group.add_argument('--tensorboard-dir', type=str, default=None, # TODO
                       help='Write TensorBoard logs to this directory.')
    group.add_argument('--no-scaled-masked-softmax-fusion',
                       action='store_false', # 'store_true" for win10 only. TODO original is store_false for linux
                       help='Disable fusion of query_key_value scaling, '
                       'masking, and softmax.',
                       dest='scaled_masked_softmax_fusion')
    group.add_argument('--scaled-upper-triang-masked-softmax-fusion',
                       type=bool,
                       help='Use upper triangular version of fused '
                       'scale, mask, softmax fusion kernel (default for GPT). '
                       '- DEPRECATED') # TODO 废弃了，不过仍然在代码中被使用
    group.add_argument('--no-bias-gelu-fusion', action='store_false', # store_false也就是默认为True，一旦命令中有此参数，其值则变为False。
                       help='Disable bias and gelu fusion.',
                       dest='bias_gelu_fusion') # dest - 被添加到 parse_args() 所返回对象上的属性名
    group.add_argument('--no-bias-dropout-fusion', action='store_false',
                       help='Disable bias and dropout fusion.',
                       dest='bias_dropout_fusion')

    return parser


def _add_initialization_args(parser):
    group = parser.add_argument_group(title='initialization')

    group.add_argument('--seed', type=int, default=1234,
                       help='Random seed used for python, numpy, '
                       'pytorch, and cuda.')
    group.add_argument('--init-method-std', type=float, default=0.02,
                       help='Standard deviation of the zero mean normal '
                       'distribution used for weight initialization.')

    return parser


def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--lr', type=float, default=0.0001, #None,
                       help='Initial learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine'],
                       help='Learning rate decay function.')
    group.add_argument('--lr-decay-iters', type=int, default=2, #None,
                       help='number of iterations to decay learning rate over,'
                       ' If None defaults to `--train-iters`')
    group.add_argument('--lr-decay-samples', type=int, default=None,
                       help='number of samples to decay learning rate over,'
                       ' If None defaults to `--train-samples`')
    group.add_argument('--lr-warmup-fraction', type=float, default=0.01, #None,
                       help='fraction of lr-warmup-(iters/samples) to use '
                       'for warmup (as a float)')
    group.add_argument('--lr-warmup-iters', type=int, default=0,
                       help='number of iterations to linearly warmup '
                       'learning rate over.')
    group.add_argument('--lr-warmup-samples', type=int, default=0,
                       help='number of samples to linearly warmup '
                       'learning rate over.')
    group.add_argument('--warmup', type=int, default=None,
                       help='Old lr warmup argument, do not use. Use one of the '
                       '--lr-warmup-* arguments above')
    group.add_argument('--min-lr', type=float, default=1.0e-5, #0.0,
                       help='Minumum value for learning rate. The scheduler'
                       'clip values below this threshold.')
    group.add_argument('--override-lr-scheduler', action='store_true',
                       help='Reset the values of the scheduler (learning rate,'
                       'warmup iterations, minimum learning rate, maximum '
                       'number of iterations, and decay style from input '
                       'arguments and ignore values from checkpoints. Note'
                       'that all the above values will be reset.')
    group.add_argument('--use-checkpoint-lr-scheduler', action='store_true',
                       help='Use checkpoint to set the values of the scheduler '
                       '(learning rate, warmup iterations, minimum learning '
                       'rate, maximum number of iterations, and decay style '
                       'from checkpoint and ignore input arguments.')

    return parser


def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    def_checkpoint_path = r'C:\Users\user\source\repos\megatron\megatron\pretrained\bert_en_fsi'

    group.add_argument('--save', type=str, default=def_checkpoint_path, #None,
                       help='Output directory to save checkpoints to.')
    group.add_argument('--save-interval', type=int, default=5, #None,
                       help='Number of iterations between checkpoint saves.')
    group.add_argument('--no-save-optim', action='store_true',
                       help='Do not save current optimizer.')
    group.add_argument('--no-save-rng', action='store_true',
                       help='Do not save current rng state.')
    group.add_argument('--load', type=str, default=def_checkpoint_path, #None,
                       help='Directory containing a model checkpoint.')
    group.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer when loading checkpoint.')
    group.add_argument('--no-load-rng', action='store_true',
                       help='Do not load rng state when loading checkpoint.')
    group.add_argument('--finetune', action='store_true',
                       help='Load model for finetuning. Do not load optimizer '
                       'or rng state from checkpoint and set iteration to 0. '
                       'Assumed when loading a release checkpoint.')
    # rng state = random number generator's state
    return parser


def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--fp16', action='store_true',
                       help='Run model in fp16 mode.')
    group.add_argument('--loss-scale', type=float, default=None,
                       help='Static loss scaling, positive power of 2 '
                       'values can improve fp16 convergence. If None, dynamic'
                       'loss scaling is used.')
    group.add_argument('--initial-loss-scale', type=float, default=2**32,
                       help='Initial loss-scale for dynamic loss scaling.')
    group.add_argument('--min-loss-scale', type=float, default=1.0,
                       help='Minimum loss scale for dynamic loss scale.')
    group.add_argument('--loss-scale-window', type=float, default=1000,
                       help='Window over which to raise/lower dynamic scale.')
    group.add_argument('--hysteresis', type=int, default=2, # 滞后现象 TODO
                       help='hysteresis for dynamic loss scaling')
    group.add_argument('--fp32-residual-connection', action='store_true',
                       help='Move residual connections to fp32.')
    group.add_argument('--no-query-key-layer-scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    group.add_argument('--attention-softmax-in-fp32', action='store_true',
                       help='Run attention masking and softmax in fp32. '
                       'This flag is ignored unless '
                       '--no-query-key-layer-scaling is specified.')
    group.add_argument('--fp32-allreduce', action='store_true',
                       help='All-reduce in fp32')
    group.add_argument('--fp16-lm-cross-entropy', action='store_true',
                       help='Move the cross entropy unreduced loss calculation'
                       'for lm head to fp16.')

    return parser


def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--tensor-model-parallel-size', type=int, default=1,
                       help='Degree of tensor model parallelism.')
    group.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                       help='Degree of pipeline model parallelism.')
    group.add_argument('--model-parallel-size', type=int, default=None, # TODO do not use!
                       help='Old model parallel argument, do not use. Use '
                       '--tensor-model-parallel-size instead.')
    group.add_argument('--distributed-backend', default='nccl',
                       choices=['nccl', 'gloo'],
                       help='Which backend to use for distributed training.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--lazy-mpu-init', type=bool, required=False,
                       help='If set to True, initialize_megatron() skips DDP initialization'
                       ' and returns function to complete it instead.'
                       'Also turns on --use-cpu-initialization flag.'
                       'This is for external DDP manager.' )
    group.add_argument('--use-cpu-initialization', action='store_true', #'store_false', TODO, when real bert/gpt training, need to set "store_true"! i.e., default is gpu for initialization.
                       help='If set, affine parallel weights initialization uses CPU' )
    return parser


def _add_validation_args(parser):
    group = parser.add_argument_group(title='validation')

    group.add_argument('--eval-iters', type=int, default=5, #100,
                       help='Number of iterations to run for evaluation'
                       'validation/test for.')
    group.add_argument('--eval-interval', type=int, default=5, #1000,
                       help='Interval between running evaluation on '
                       'validation set.')

    return parser


def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')
    def_data_path=r'C:\Users\user\source\repos\megatron\megatron\fsi-en-bert-large-cased-small3-win10_text_sentence'
    #def_vocab_file=r'C:\Users\user\source\repos\megatron\megatron\pretrained\bert_en_fsi\bert-large-cased-vocab.txt'
    def_vocab_file=r'/workspace/megatron/megatron2/pretrained/bert-large-cased-vocab.txt'

    group.add_argument('--data-path', nargs='*', default=def_data_path, #None,
                       help='Path to the training dataset. Accepted format:'
                       '1) a single data path, 2) multiple datasets in the'
                       'form: dataset1-weight dataset1-path dataset2-weight '
                       'dataset2-path ...')
    group.add_argument('--split', type=str, default='949,50,1', #'969, 30, 1',
                       help='Comma-separated list of proportions for training,'
                       ' validation, and test split. For example the split '
                       '`90,5,5` will use 90%% of data for training, 5%% for '
                       'validation and 5%% for test.')
    group.add_argument('--vocab-file', type=str, default=def_vocab_file, #None,
                       help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file.')
    group.add_argument('--emoji-file', type=str, default=None,
                       help='Path to the emoji file for japanese tokenization.')
    group.add_argument('--mecab-dict-path', type=str, default=None,
                       help='Path to the mecab dict (ipadict/unidict) file for japanese tokenization (word breaker).')
    group.add_argument('--seq-length', type=int, default=128,#None,
                       help="Maximum sequence length to process.")
    group.add_argument('--mask-prob', type=float, default=0.15,
                       help='Probability of replacing a token with mask.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='Probability of producing a short sequence.')
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--num-workers', type=int, default=1, #2, TODO 1 is for debug only!
                       help="Dataloader number of workers.")
    group.add_argument('--tokenizer-type', type=str,
                       default='BertWordPieceCase', #'GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'BertWordPieceCaseJp',
                                'BertWordPieceCaseCh',
                                'GPT2BPETokenizer',
                                'GPT2BPETokenizerJp',
                                'GPT2BPETokenizerJpMecab'],
                       help='What type of tokenizer to use.')
    group.add_argument('--lowercase', action='store_true', 
                       help='is perform lowercase or not, default=keep case sensitive')
    group.add_argument('--data-impl', type=str, default='mmap', #'infer',
                       choices=['lazy', 'cached', 'mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--reset-position-ids', action='store_true',
                       help='Reset posistion ids after end-of-document token.')
    group.add_argument('--reset-attention-mask', action='store_true',
                       help='Reset self attention maske after '
                       'end-of-document token.')
    group.add_argument('--eod-mask-loss', action='store_true',
                       help='Mask loss for the end of document tokens.')

    return parser


def _add_autoresume_args(parser):
    group = parser.add_argument_group(title='autoresume')

    group.add_argument('--adlr-autoresume', action='store_true', # TODO what is adlr?
                       help='Enable autoresume on adlr cluster.')
    group.add_argument('--adlr-autoresume-interval', type=int, default=1000,
                       help='Intervals over which check for autoresume'
                       'termination signal')

    return parser


def _add_realm_args(parser): # REALM (2020.Feb.01): Retrieval-Augmented Language Model Pre-Training, a paper: https://arxiv.org/pdf/2002.08909.pdf
    group = parser.add_argument_group(title='realm')

    # network size
    group.add_argument('--ict-head-size', type=int, default=None,
                       help='Size of block embeddings to be used in ICT and REALM (paper default: 128)')

    # checkpointing
    group.add_argument('--ict-load', type=str, default=None,
                       help='Directory containing an ICTBertModel checkpoint')
    group.add_argument('--bert-load', type=str, default=None,
                       help='Directory containing an BertModel checkpoint (needed to start ICT and REALM)')

    # data
    group.add_argument('--titles-data-path', type=str, default=None,
                       help='Path to titles dataset used for ICT')
    group.add_argument('--query-in-block-prob', type=float, default=0.1,
                       help='Probability of keeping query in block for ICT dataset')
    group.add_argument('--use-one-sent-docs', action='store_true',
                       help='Whether to use one sentence documents in ICT')

    # training
    group.add_argument('--report-topk-accuracies', nargs='+', default=[],
                       help="Which top-k accuracies to report (e.g. '1 5 20')")

    # faiss index, Faiss is a library for efficient similarity search and clustering of dense vectors. https://github.com/facebookresearch/faiss
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--block-data-path', type=str, default=None,
                       help='Where to save/load BlockData to/from')

    # indexer
    group.add_argument('--indexer-batch-size', type=int, default=128,
                       help='How large of batches to use when doing indexing jobs')
    group.add_argument('--indexer-log-interval', type=int, default=1000,
                       help='After how many batches should the indexer report progress')
    return parser
