"""
litetorch/core/cross_entropy_function.py
This module defines the CrossEntropyFunction class, which implements the cross-entropy loss function for tensors in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from typing import Tuple
from litetorch.core.function import Function
from litetorch.core.tensor import Tensor
from litetorch.utils.function import softmax


class CrossEntropyFunction(Function):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        CrossEntropy with class index targets.

        - input: shape (N, C), raw logits
        - target: shape (N,), values in 0..C-1
        """
        self.input = input
        self.target = target

        # Stable softmax
        x = input.data
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        self.probabilities = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Log probability of the correct class
        batch_indices = np.arange(x.shape[0])
        target_indices = target.data.astype(np.int32)
        log_probs = -np.log(self.probabilities[batch_indices, target_indices])

        loss = np.mean(log_probs)
        return Tensor(loss, requires_grad=input.auto_grad)

    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tensor:
        grad_output = grad_outputs[0].data  # usually scalar 1

        grad_input = self.probabilities.copy()
        batch_indices = np.arange(self.input.data.shape[0])
        class_indices = self.target.data.astype(np.int32)

        # Subtract 1 from the correct class
        grad_input[batch_indices, class_indices] -= 1
        grad_input /= self.input.data.shape[0]  # average over batch

        grad_input *= grad_output  # in case ∂L/∂loss ≠ 1
        return Tensor(grad_input, requires_grad=self.input.auto_grad)
