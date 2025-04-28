"""
litetorch/loss/binary_cross_entropy.py
This module defines the Binary Cross Entropy (BCE) loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.loss.base import Loss
from litetorch.core.binary_cross_entropy_function import BinaryCrossEntropyFunction


class BinaryCrossEntropyLoss(Loss):
    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.epsilon = epsilon
        self._name = "BinaryCrossEntropyLoss"

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return BinaryCrossEntropyFunction(self.epsilon)(output, target)
