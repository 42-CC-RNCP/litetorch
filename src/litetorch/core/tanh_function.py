"""
litetorch/core/tanh_function.py
This module defines the TanhFunction class, which implements the hyperbolic tangent activation function for tensors in the LiteTorch framework.
This function is commonly used in neural networks as an activation function.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import tanh


class TanhFunction(Function):
    """
    TanhFunction implements the hyperbolic tangent activation function for tensors.
    It takes a tensor as input and returns the hyperbolic tangent of that tensor.
    The operation is defined as:
        f(x) = tanh(x)
    where x is a tensor.
    """

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        output = tanh(input.data)
        return Tensor(output, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        grad_output = grad_outputs[0]
        tanh_input = self.input.data
        grad_input = grad_output.data * (1 - tanh_input ** 2)
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
