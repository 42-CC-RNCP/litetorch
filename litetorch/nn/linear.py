"""
litetorch/nn/linear.py
This module defines the Linear class, which implements a linear (fully connected) layer in a neural network.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module


class Linear(Module):
    """
    Linear layer (fully connected layer) in a neural network.

    Parameters:
    - in_features: Number of input features.
    - out_features: Number of output features.
    - bias: If True, adds a learnable bias to the output. Default is True.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.input = None
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        # Xavier initialization for weights
        # The weights are initialized using a normal distribution with mean 0 and standard deviation
        # sqrt(2 / in_features) to help with convergence.
        # This is a common practice for initializing weights in deep learning to avoid vanishing/exploding gradients.
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True,
        )
        self.bias_tensor = (
            Tensor(np.zeros((1, out_features)), requires_grad=True) if bias else None
        )
        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias_tensor if bias else None
        self._name = "Linear"

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the linear layer.

        Parameters:
        - x: Input tensor of shape (batch_size, in_features).

        Returns:
        - Output tensor of shape (batch_size, out_features).
        """
        self.input = x
        if self.bias_tensor is not None:
            return x @ self.weight + self.bias_tensor
        else:
            return x @ self.weight

    def backward(self, grad_output: Tensor) -> Tensor:
        x = self.input.data

        # shape of grad_output should be (batch_size, out_features)
        dL_dy = grad_output.data

        # ∂L/∂W = x^T * ∂L/∂y
        dL_dW = x.T @ dL_dy
        self.weight.grad += dL_dW

        # ∂L/∂b = ∂L/∂y
        dL_db = np.sum(dL_dy, axis=0, keepdims=True) if self.bias else None
        self.bias_tensor.grad += dL_db

        # ∂L/∂x = ∂L/∂y * W^T
        dL_dx = dL_dy @ self.weight.data.T
        return Tensor(dL_dx, requires_grad=self.input.auto_grad)

    def __repr__(self) -> str:
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias})"
