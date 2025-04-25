"""
litetorch/loss/binary_cross_entropy.py
This module defines the Binary Cross Entropy (BCE) loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.loss.base import Loss
from litetorch.utils.function import softmax


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.epsilon = epsilon
        self._name = "BinaryCrossEntropyLoss"

    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        loss = -np.mean(target * np.log(output + self.epsilon) + (1 - target) * np.log(1 - output + 1e-15))
        return loss

    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        grad = (output - target) / (output * (1 - output) + self.epsilon)
        return grad
