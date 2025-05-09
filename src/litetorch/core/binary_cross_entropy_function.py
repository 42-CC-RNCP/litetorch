"""
litetorch/loss/binary_cross_entropy.py
This module defines the Binary Cross Entropy loss function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-28
"""

import numpy as np
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import softmax


class BinaryCrossEntropyFunction(Function):
    """
    Applies the Binary Cross Entropy loss function.
    The Binary Cross Entropy loss is defined as:
        BCE(y, y_hat) = -1/N * Σ [y * log(y_hat) + (1 - y) * log(1 - y_hat)]
    where:
        - y: true labels
        - y_hat: predicted probabilities
        - N: number of samples
    """
    def __init__(self, epsilon: float = 1e-15) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        self.input = input
        self.target = target

        # Clip input to avoid log(0)
        input_data = np.clip(self.input.data, self.epsilon, 1 - self.epsilon)

        # Calculate the Binary Cross Entropy loss
        loss = -np.mean(
            target.data * np.log(input_data + self.epsilon) +
            (1 - target.data) * np.log(1 - input_data + self.epsilon)
        )

        return Tensor(loss, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tensor:
        grad_output = grad_outputs[0]
        batch_size = self.input.data.shape[0]

        # Calculate the gradient of the Binary Cross Entropy loss
        # grad_input = (self.input.data - self.target.data) / (self.input.data * (1 - self.input.data) * batch_size)
        # Need to make sure the denominator is not zero
        denominator = self.input.data * (1 - self.input.data) * batch_size
        denominator = np.clip(denominator, self.epsilon, np.inf)
        grad_input = (self.input.data - self.target.data) / denominator
        grad_input *= grad_output.data

        return Tensor(grad_input, requires_grad=self.input.auto_grad)
