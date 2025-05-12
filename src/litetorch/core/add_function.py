"""
litetorch/core/add_function.py
This module defines the AddFunction class, which implements the addition operation for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.tensor import Tensor
from litetorch.core.function import Function
from litetorch.utils.linear_algebra import reduce_broadcast_shape


class AddFunction(Function):
    """
    AddFunction implements the addition operation for tensors.
    It takes two tensors as input and returns their sum.
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the addition operation.

        Parameters:
        - a: First input tensor.
        - b: Second input tensor.

        Returns:
        - Output tensor after addition.
        """
        return Tensor(a.data + b.data, requires_grad=a.auto_grad or b.auto_grad)

    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Backward pass of the addition operation.

        Parameters:
        - grad_outputs: Gradients of the output tensor.

        Returns:
        - Gradients of the input tensors.
        """
        # dL/da = 1
        # dL/db = 1
        # The gradient of the sum is 1 for both inputs
        # This means that the gradient of the output with respect to each input is 1.
        grad_output = grad_outputs[0].data
        a, b = self.inputs
        a_shape = a.shape
        b_shape = b.shape

        # Gradient for a: same shape as a
        grad_a = grad_output
        grad_b = grad_output

        # Handle broadcasting for a
        if grad_a.shape != a_shape:
            grad_a = reduce_broadcast_shape(grad_a, a_shape)

        # Handle broadcasting for b
        if grad_b.shape != b_shape:
            grad_b = reduce_broadcast_shape(grad_b, b_shape)

        return Tensor(grad_a, requires_grad=False), Tensor(grad_b, requires_grad=False)

