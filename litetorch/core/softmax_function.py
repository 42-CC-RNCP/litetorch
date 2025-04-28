"""
litetorch/core/softmax_function.py
This module defines the softmax activation function for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import softmax


class SoftmaxFunction(Function):
    """
    Applies the Softmax activation function along the specified dimension.
    The Softmax function is defined as:
        f(x_i) = exp(x_i) / sum(exp(x_j))
    where the sum is over all j in the specified dimension.
    """

    def forward(self, input: Tensor, dim: int = -1) -> Tensor:
        self.input = input
        self.dim = dim
        output = softmax(input.data, dim)
        return Tensor(output, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        grad_output = grad_outputs[0]
        grad_input = grad_output.data * (self.input.data * (1 - self.input.data)).sum(axis=self.dim, keepdims=True)
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
