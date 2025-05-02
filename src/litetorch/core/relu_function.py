"""
litetorch/core/relu_function.py
This module defines the ReLUFunction class, which implements the ReLU (Rectified Linear Unit) activation function

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import relu


class ReLUFunction(Function):
    def forward(self, input: Tensor) -> Tensor:
        self.input = input
        output = relu(input.data)
        return Tensor(output, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tuple[Tensor, ...]:
        grad_output = grad_outputs[0]
        grad_input = grad_output.data * (self.input.data > 0).astype(np.float32)
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
