"""
litetorch/core/mse_function.py
This module defines the Mean Squared Error (MSE) function for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import mse


class MSEFunction(Function):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the Mean Squared Error (MSE) between the input and target tensors.

        Parameters:
        - input: The predicted values (output of the model).
        - target: The true values (ground truth).

        Returns:
        - A tensor containing the MSE loss value.
        """
        self.input = input
        self.target = target
        self.diff = mse(input.data, target.data)
        batch_size = input.data.size
        return Tensor(self.diff / batch_size, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tensor:
        grad_output = grad_outputs[0]
        batch_size = self.input.data.size
        grad_input = (2 * self.diff) / batch_size
        grad_input *= grad_output.data
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
