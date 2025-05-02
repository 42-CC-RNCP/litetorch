"""
litetorch/core/leaky_relu_function.py
This module defines the LeakyReLUFunction class, which implements the Leaky ReLU (Rectified Linear Unit) activation function

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import leaky_relu


class LeakyReLUFunction(Function):
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
        self.input = input
        output = leaky_relu(input.data, self.negative_slope)
        return Tensor(output, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        grad_output = grad_outputs[0]
        grad_input = grad_output.data * np.where(self.input.data > 0, 1, self.negative_slope)
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
