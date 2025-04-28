"""
litetorch/nn/activation.py
This module defines various activation functions used in neural networks.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module
from litetorch.core.relu_function import ReLUFunction
from litetorch.core.sigmoid_function import SigmoidFunction
from litetorch.core.tanh_function import TanhFunction
from litetorch.core.softmax_function import SoftmaxFunction
from litetorch.core.leaky_relu_function import LeakyReLUFunction


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return ReLUFunction()(input)


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return SigmoidFunction()(input)


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return TanhFunction()(input)


class Softmax(Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        return SoftmaxFunction()(input, dim=self.dim)


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return LeakyReLUFunction()(input, negative_slope=self.negative_slope)
