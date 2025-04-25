"""
litetorch/nn/activation.py
This module defines various activation functions used in neural networks.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module


class ReLU(Module):
    """
    Applies the ReLU activation function element-wise.
    The ReLU function is defined as:
        f(x) = max(0, x)
    """
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.maximum(0, input.data), requires_grad=input.auto_grad)


class Sigmoid(Module):
    """
    Applies the Sigmoid activation function element-wise.
    The Sigmoid function is defined as:
        f(x) = 1 / (1 + exp(-x))
    """
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(1 / (1 + np.exp(-input.data)), requires_grad=input.auto_grad)


class Tanh(Module):
    """
    Applies the Tanh activation function element-wise.
    The Tanh function is defined as:
        f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.tanh(input.data), requires_grad=input.auto_grad)


class Softmax(Module):
    """
    Applies the Softmax activation function along the specified dimension.
    The Softmax function is defined as:
        f(x_i) = exp(x_i) / sum(exp(x_j))
    where the sum is over all j in the specified dimension.
    """
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        exp_input = np.exp(input.data - np.max(input.data, axis=self.dim, keepdims=True))
        return Tensor(exp_input / np.sum(exp_input, axis=self.dim, keepdims=True), requires_grad=input.auto_grad)


class LeakyReLU(Module):
    """
    Applies the Leaky ReLU activation function element-wise.
    The Leaky ReLU function is defined as:
        f(x) = x if x > 0 else α * x
    where α is a small constant (default: 0.01).
    """
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.where(input.data > 0, input.data, self.negative_slope * input.data), requires_grad=input.auto_grad)
