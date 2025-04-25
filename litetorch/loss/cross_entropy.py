"""
litetorch/loss/cross_entropy.py
This module defines the Cross Entropy loss function for a neural network framework.
This loss function is commonly used for multi-class classification tasks.
It calculates the loss between the predicted probabilities and the true class labels.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.loss.base import Loss
from litetorch.utils.function import softmax


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        self._name = "CrossEntropyLoss"

    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        # Apply softmax to the output to get probabilities
        probabilities = softmax(output)

        # Clip probabilities to avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        # Calculate the Cross Entropy loss
        loss = -np.sum(target * np.log(probabilities)) / target.shape[0]

        return loss

    def backward(self, output: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Apply softmax to the output to get probabilities
        probabilities = softmax(output)

        # Clip probabilities to avoid log(0)
        probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)

        # Calculate the gradient of the Cross Entropy loss
        grad = (probabilities - target) / target.shape[0]

        return grad

    def __call__(self, output: Tensor, target: Tensor) -> float:
        return self.forward(output.data, target.data)
