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
from litetorch.utils.save_load import SaveLoadMixin


class Sequential(Module, SaveLoadMixin):
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

    def parameters(self) -> List[Tensor]:
        """
        Get all parameters of the network.
        """
        params = []
        for layer in self.layers:
            layer : Module
            params.extend(layer.parameters())
        return params

    def get_config(self):
        configs = []
        for layer in self.layers:
            layer : Module
            configs.append({
                "type": layer.__class__.__name__,
                "config": layer.get_config()
            })
        return {"architecture": configs}

    def get_parameters(self):
        params = {}
        for i, layer in enumerate(self.layers):
            layer : Module
            params = layer.get_parameters()
            if params:
                params[f"layer_{i}"] = params
        return params

    def __repr__(self):
        return f"{self.__class__.__name__}(\n" + ",\n".join([f"  {layer}" for layer in self.layers]) + "\n)"
