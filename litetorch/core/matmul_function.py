"""
litetorch/core/matmul_function.py
This module defines the MatMulFunction class, which implements the matrix multiplication operation for tensors in the LiteTorch framework

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor


class MatMulFunction(Function):
    """
    MatMulFunction implements the matrix multiplication operation for tensors.
    It takes two tensors as input and returns their product.
    The operation is defined as:
        f(a, b) = a @ b
    where a and b are tensors.
    """

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Forward pass of the matrix multiplication operation.

        Parameters:
        - a: First input tensor.
        - b: Second input tensor.

        Returns:
        - Output tensor after matrix multiplication.
        """
        return Tensor(a.data @ b.data, requires_grad=a.auto_grad or b.auto_grad)

    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """
        Backward pass of the matrix multiplication operation.

        Parameters:
        - grad_outputs: Gradients of the output tensor.

        Returns:
        - Gradients of the input tensors.
        """
        # dL/da = grad_outputs @ b.T
        # dL/db = a.T @ grad_outputs
        # The gradient of the matrix multiplication is the other input transposed
        return grad_outputs[0] @ self.inputs[1].T, self.inputs[0].T @ grad_outputs[0]
