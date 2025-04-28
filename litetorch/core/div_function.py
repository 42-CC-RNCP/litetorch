"""
litetorch/core/div_function.py
This module defines the DivFunction class, which implements the division operation for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor


class DivFunction(Function):
    """
    DivFunction implements the division operation for tensors.
    It takes two tensors as input and returns their quotient.
    The operation is defined as:
        f(a, b) = a / b
    where a and b are tensors.
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the division operation.

        Parameters:
        - a: First input tensor.
        - b: Second input tensor.

        Returns:
        - Output tensor after division.
        """
        return Tensor(a.data / b.data, requires_grad=a.auto_grad or b.auto_grad)

    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Backward pass of the division operation.

        Parameters:
        - grad_outputs: Gradients of the output tensor.

        Returns:
        - Gradients of the input tensors.
        """
        # dL/da = 1/b
        # dL/db = -a/(b^2)
        # The gradient of the division is 1/b for a and -a/(b^2) for b
        return grad_outputs[0] / self.inputs[1], -self.inputs[0] * grad_outputs[0] / (self.inputs[1] ** 2)
