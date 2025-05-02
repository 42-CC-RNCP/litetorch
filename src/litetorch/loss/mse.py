"""
litetorch/loss/mse.py
This module defines the Mean Squared Error (MSE) loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

from litetorch.core.tensor import Tensor
from litetorch.loss.base import Loss
from litetorch.core.mse_function import MSEFunction


class MSELoss(Loss):
    def __init__(self) -> None:
        """
        Initialize the Mean Squared Error (MSE) loss function.
        """
        super().__init__()
        self._name = "MSELoss"

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return MSEFunction()(output, target)
