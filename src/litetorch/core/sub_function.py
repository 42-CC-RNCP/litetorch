"""
litetorch/core/sub_function.py
This module defines the SubFunction class, which implements the subtraction operation for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.linear_algebra import reduce_broadcast_shape


class SubFunction(Function):
    """
    SubFunction implements the subtraction operation for tensors.
    It takes two tensors as input and returns their difference.
    The operation is defined as:
        f(a, b) = a - b
    where a and b are tensors.
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the subtraction operation.
        ...
        Returns:
            Output tensor after subtraction.
        """
        return Tensor(a.data - b.data, requires_grad=a.auto_grad or b.auto_grad)

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
        # In the case of subtraction, the gradient with respect to b is -1.
        grad_output = grad_outputs[0].data
        a = self.inputs[0]
        b = self.inputs[1]

        grad_a = grad_output
        grad_b = -grad_output

        if grad_a.shape != a.shape:
            grad_a = reduce_broadcast_shape(grad_a, a.shape)
        if grad_b.shape != b.shape:
            grad_b = reduce_broadcast_shape(grad_b, b.shape)

        return Tensor(grad_a, requires_grad=a.auto_grad), Tensor(grad_b, requires_grad=b.auto_grad)
