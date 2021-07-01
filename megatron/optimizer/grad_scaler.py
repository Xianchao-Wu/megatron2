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

"""Megatron grad scaler."""

from abc import ABC
from abc import abstractmethod

import torch

# 梯度大小调节器
class MegatronGradScaler(ABC):

    def __init__(self, initial_scale):
        """Initialize scale value with the input initial scale."""
        assert initial_scale > 0.0 # 4,294,967,296
        self._scale = torch.cuda.FloatTensor([initial_scale])

    @property
    def scale(self):
        return self._scale

    @property
    def inv_scale(self):
        """ inv = reciprocal, from x to 1.0/x """
        return self._scale.double().reciprocal().float() # 倒数 x to 1.0/x

    @abstractmethod
    def update(self, found_inf):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass



class ConstantGradScaler(MegatronGradScaler):
    """ constrant gradient scaler"""

    def update(self, found_inf):
        pass

    def state_dict(self):
        return dict()

    def load_state_dict(self, state_dict):
        pass



class DynamicGradScaler(MegatronGradScaler):
    """dynamic gradient scaler"""

    def __init__(self, initial_scale, min_scale, # 1=initial_scale=4.3Billion; 2=min_scale=1.0
                 growth_factor, backoff_factor, # 3=growth_factor=2.0; 4=backoff_factor=0.5
                 growth_interval, hysteresis): # 5=growth_interval=1000; 6=hysteresis=2
        """"Grad scaler with dynamic scale that gets adjusted
        during training."""
        super(DynamicGradScaler, self).__init__(initial_scale) # param.1

        # Lower bound on the scale.
        assert min_scale > 0.0 # param.2
        assert min_scale <= initial_scale
        self.min_scale = torch.cuda.FloatTensor([min_scale])

        # Growth and backoff factors for the scale.
        assert growth_factor > 1.0 # param.3 生长因子
        self.growth_factor = torch.cuda.FloatTensor([growth_factor])

        assert backoff_factor < 1.0 # param.4 后退因子
        assert backoff_factor > 0.0
        self.backoff_factor = torch.cuda.FloatTensor([backoff_factor])

        # Interval over which if we don't see any inf/nan,
        # we will scale the grad scale by the growth factor.
        assert growth_interval > 0 # param.5 增长区间
        self.growth_interval = growth_interval
        # Number of inf/nans we should see before scaling down
        # the grad scale by the backoff factor.
        assert hysteresis > 0 # 迟滞现象；滞后作用
        self.hysteresis = hysteresis

        # Trackers.
        self._growth_tracker = 0
        self._hysteresis_tracker = self.hysteresis


    def update(self, found_inf):

        # If we have an inf/nan, growth tracker is set to 0
        # and hysterisis tracker is reduced by 1.
        if found_inf:
            self._growth_tracker = 0
            self._hysteresis_tracker -= 1
            # Now if we are out of hysteresis count, scale down the loss.
            if self._hysteresis_tracker <= 0:
                self._scale = torch.max(self._scale * self.backoff_factor,
                                        self.min_scale)
        else:
            # If there is no nan/inf, increment the growth tracker.
            self._growth_tracker += 1
            # If we have had enough consequitive (consecutive?) intervals with no nan/inf:
            if self._growth_tracker == self.growth_interval:
                # Reset the tracker and hysteresis trackers,
                self._growth_tracker = 0
                self._hysteresis_tracker = self.hysteresis
                # and scale up the loss scale.
                self._scale = self._scale * self.growth_factor


    def state_dict(self):
        state_dict = {}
        state_dict['scale'] = self._scale
        state_dict['growth_tracker'] = self._growth_tracker
        state_dict['hysteresis_tracker'] = self._hysteresis_tracker
        return state_dict


    def load_state_dict(self, state_dict):
        self._scale = state_dict['scale'].cuda(torch.cuda.current_device())
        self._growth_tracker = state_dict['growth_tracker']
        self._hysteresis_tracker = state_dict['hysteresis_tracker']
