"""
litetorch/loss/cross_entropy.py
This module defines the Cross Entropy loss function for a neural network framework.
This loss function is commonly used for multi-class classification tasks.
It calculates the loss between the predicted probabilities and the true class labels.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

from litetorch.core.tensor import Tensor
from litetorch.loss.base import Loss
from litetorch.core.cross_entropy_function import CrossEntropyFunction


class CrossEntropyLoss(Loss):
    def __init__(self) -> None:
        super().__init__()
        self._name = "CrossEntropyLoss"

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return CrossEntropyFunction()(output, target)
