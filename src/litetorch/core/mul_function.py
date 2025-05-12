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
from litetorch.utils.linear_algebra import reduce_broadcast_shape


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
        grad_output = grad_outputs[0].data
        a = self.inputs[0]
        b = self.inputs[1]

        grad_a = grad_output * b.data
        grad_b = grad_output * a.data

        # Ensure that the gradients are of the same shape as the inputs
        # Handle broadcasting for a
        if grad_a.shape != a.shape:
            grad_a = reduce_broadcast_shape(grad_a, a.shape)
        # Handle broadcasting for b
        if grad_b.shape != b.shape:
            grad_b = reduce_broadcast_shape(grad_b, b.shape)

        return Tensor(grad_a, requires_grad=False), Tensor(grad_b, requires_grad=False)
