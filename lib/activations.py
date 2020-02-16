#!/usr/bin/env python3
# Copyright 2019 Lee Sharkey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

r"""
Activation functions and their derivatives
------------------------------------------

This script contains the several activation functions that I would like to try
namely
  -  hard sigmoid (default)
  -  ReLU
  -  swish

It also contains implementations of their derivatives.
"""

# TODO Update the code more generally so that your use of custom activations
#  doesn't look so stupid, not using autograd.

def get_hard_sigmoid():
    return torch.nn.Hardtanh(min_val=0.0)

def get_hard_sigmoid_gradient():
    return lambda x: (x >= 0) & (x <= 1)

def get_relu():
    return torch.nn.ReLU(inplace=False)

def get_relu_gradient():
    return lambda x: (x > 0) #TODO test whether >= makes a differenct to stability

def get_swish():
    sigmoid = torch.nn.Sigmoid()
    return lambda x : x * sigmoid(x)

def get_swish_gradient():
    sigmoid = torch.nn.Sigmoid()
    swish = get_swish()
    return lambda x : swish(x) + sigmoid(x)*(1-swish(x))

def get_leaky_hard_sigmoid():
    hsig = torch.nn.Hardtanh(min_val=0.0)
    return lambda x : hsig(x) + 0.01*x





