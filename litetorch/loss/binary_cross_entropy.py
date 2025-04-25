"""
litetorch/loss/binary_cross_entropy.py
This module defines the Binary Cross Entropy (BCE) loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.loss.base import Loss


class BinaryCrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        self._name = "BinaryCrossEntropyLoss"

    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        # Clip the output to prevent log(0)
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Clip the output to prevent division by zero
        output = np.clip(output, 1e-15, 1 - 1e-15)
        return (output - target) / (output * (1 - output) * output.size)
