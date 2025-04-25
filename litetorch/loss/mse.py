"""
litetorch/loss/mse.py
This module defines the Mean Squared Error (MSE) loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.loss.base import Loss


class MSELoss(Loss):
    def __init__(self) -> None:
        """
        Initialize the Mean Squared Error (MSE) loss function.
        """
        super().__init__()
        self._name = "MSELoss"

    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the MSE loss given the model's output and the target values.

        Parameters:
        - output: The model's output (predictions).
        - target: The true target values.

        Returns:
        - The calculated MSE loss value.
        """
        return ((output - target) ** 2).mean()

    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Calculate the gradient of the MSE loss with respect to the model's output.

        Parameters:
        - output: The model's output (predictions).
        - target: The true target values.

        Returns:
        - The gradient of the MSE loss with respect to the model's output.
        """
        return 2 * (output - target) / output.size
