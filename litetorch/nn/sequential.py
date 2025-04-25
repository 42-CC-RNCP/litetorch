"""
litetorch/nn/sequential.py
This module defines the Sequential class for a neural network framework.
The Sequential class allows users to create a neural network by stacking layers in a sequential manner.
It provides methods for adding layers, performing forward and backward passes, and updating parameters.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

from typing import List, Callable, Any
import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module
from litetorch.optim.base import Optimizer


class Sequential(Module):
    """
    A sequential container for stacking layers in a neural network.
    """

    def __init__(self, *layers: List[Module]) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform the forward pass through the network.
        """
        for layer in self.layers:
            # update the input tensor and pass it to the next layer
            x = layer(x)
        return x

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Perform the backward pass through the network.
        """
        for layer in reversed(self.layers):
            # update the gradient tensor and pass it to the previous layer
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self) -> List[Tensor]:
        """
        Get all parameters of the network.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" + ",\n".join([f"  {layer}" for layer in self.layers]) + "\n)"
