"""
litetorch/core/sigmoid_function.py
This module defines the SigmoidFunction class, which implements the sigmoid activation function for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import sigmoid


class SigmoidFunction(Function):
    """
    SigmoidFunction implements the sigmoid activation function for tensors.
    It takes a tensor as input and returns the sigmoid of that tensor.
    The operation is defined as:
        f(x) = 1 / (1 + exp(-x))
    where x is a tensor.
    """

    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        output = sigmoid(input.data)
        return Tensor(output, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        grad_output = grad_outputs[0]
        sigmoid_input = self.input.data
        grad_input = grad_output.data * sigmoid_input * (1 - sigmoid_input)
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
