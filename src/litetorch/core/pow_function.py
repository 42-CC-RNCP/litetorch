"""
litetorch/core/pow_function.py
This module defines the PowFunction class, which implements the power operation for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.linear_algebra import reduce_broadcast_shape


class PowFunction(Function):
    """
    PowFunction implements the power operation for tensors.
    It takes a tensor and a scalar exponent as input and returns the tensor raised to that power.
    The operation is defined as:
        f(a, b) = a ** b
    where a is a tensor and b is a scalar.
    """

    def forward(self, a: Tensor, b: float) -> Tensor:
        """
        Forward pass of the power operation.

        Parameters:
        - a: Input tensor.
        - b: Scalar exponent.

        Returns:
        - Output tensor after raising a to the power of b.
        """
        return Tensor(a.data ** b, requires_grad=a.auto_grad)
    
    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Backward pass of the power operation.

        Parameters:
        - grad_outputs: Gradients of the output tensor.

        Returns:
        - Gradient of the input tensor with respect to the output.
        """
        grad_output = grad_outputs[0].data
        a = self.inputs[0]
        b = self.inputs[1]
        # Gradient of a^b with respect to a is b * a^(b-1)
        grad_a = grad_output * b * (a.data ** (b - 1))
        # b is a scalar, so we don't need to compute grad_b
        
        return (Tensor(grad_a, requires_grad=False),)
