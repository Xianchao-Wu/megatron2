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

try:
    from apex.optimizers import FusedAdam as Adam # TODO no apex in win10
except:
    from torch.optim import Adam

from megatron import get_args
from megatron.model import import_layernorm

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import FP16OptimizerWithFP16Params, FP32Optimizer


def _get_params_for_weight_decay_optimization(module):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """
    args = get_args()
    LayerNorm = import_layernorm(args.fp32_residual_connection) # apex.normalization.fused_layer_norm.FusedLayerNorm
    import pdb; pdb.set_trace()
    weight_decay_params = {'params': []} # weight_decay is alike L2-regularization, to make weight to be smaller! TODO
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    import pdb; pdb.set_trace()
    for module_ in module.modules(): # TODO to check the details of weight decay!
        if isinstance(module_, LayerNorm):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])
    import pdb; pdb.set_trace()
    return weight_decay_params, no_weight_decay_params
    # 102 elements; 200 elements

def get_megatron_optimizer(model):
    args = get_args()

    # Base optimizer.
    param_groups = _get_params_for_weight_decay_optimization(model)
    optimizer = Adam(param_groups,
                     lr=args.lr,
                     weight_decay=args.weight_decay, # 0.01, for L2-regularizaiton's lmabda coefficient
                     betas=(args.adam_beta1, args.adam_beta2), # 0.9, 0.999
                     eps=args.adam_eps) # 1e-08; TODO [问题：只能是FusedAdam or Adam吗？]

    if args.fp16: # for fp16 mixed-precision
        # Constant loss scale. [常数损失值比例/刻度]
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale. [动态损失值比例/刻度]
        else:
            grad_scaler = DynamicGradScaler( # here!
                initial_scale=args.initial_loss_scale, # 4,294,967,296
                min_scale=args.min_loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=args.loss_scale_window, # 1000
                hysteresis=args.hysteresis) # 2
        # Megatron optimizer.
        return FP16OptimizerWithFP16Params(optimizer, grad_scaler,
                                           args.clip_grad)

    # FP32.
    return FP32Optimizer(optimizer, args.clip_grad)
