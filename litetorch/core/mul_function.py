"""
litetorch/core/mul_function.py
This module defines the MulFunction class, which implements the multiplication operation for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor


class MulFunction(Function):
    """
    MulFunction implements the multiplication operation for tensors.
    It takes two tensors as input and returns their product.
    The operation is defined as:
        f(a, b) = a * b
    where a and b are tensors.
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the multiplication operation.

        Parameters:
        - a: First input tensor.
        - b: Second input tensor.

        Returns:
        - Output tensor after multiplication.
        """
        return Tensor(a.data * b.data, requires_grad=a.auto_grad or b.auto_grad)

    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Backward pass of the multiplication operation.

        Parameters:
        - grad_outputs: Gradients of the output tensor.

        Returns:
        - Gradients of the input tensors.
        """
        # dL/da = b
        # dL/db = a
        # The gradient of the product is the other input
        # This means that the gradient of the output with respect to each input is the other input.
        return grad_outputs[0] * self.inputs[1], grad_outputs[0] * self.inputs[0]
