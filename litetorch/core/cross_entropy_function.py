"""
litetorch/core/cross_entropy_function.py
This module defines the CrossEntropyFunction class, which implements the cross-entropy loss function for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import softmax


class CrossEntropyFunction(Function):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Calculate the Cross Entropy loss between the input and target tensors.

        Parameters:
        - input: The predicted values (output of the model).
        - target: The true values (ground truth).

        Returns:
        - A tensor containing the Cross Entropy loss value.
        """
        self.input = input
        self.target = target
        self.probabilities = softmax(input.data)

        # Clip probabilities to avoid log(0)
        self.probabilities = np.clip(self.probabilities, 1e-15, 1 - 1e-15)

        # Calculate the Cross Entropy loss
        loss = -np.sum(target.data * np.log(self.probabilities)) / target.shape[0]

        return Tensor(loss, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tensor) -> Tensor:
        grad_output = grad_outputs[0]
        batch_size = self.input.data.shape[0]

        # Calculate the gradient of the Cross Entropy loss
        grad_input = (self.probabilities - self.target.data) / batch_size
        grad_input *= grad_output.data

        return Tensor(grad_input, requires_grad=self.input.auto_grad)
