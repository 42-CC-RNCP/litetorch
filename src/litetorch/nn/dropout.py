"""
litetorch/nn/dropout.py
This module defines the Dropout class, which implements a dropout layer in a neural network.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-21
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module
from litetorch.utils.registry import register_layer


@register_layer("Dropout")
class Dropout(Module):
    """
    Dropout layer in a neural network.

    Parameters:
    - p: Probability of an element to be zeroed. Default is 0.5.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self._name = "Dropout"

    def forward(self, input: Tensor) -> Tensor:
        if self.p < 0 or self.p > 1:
            raise ValueError(f"Dropout probability p must be between 0 and 1, but got {self.p}")
        if not self._training or self.p == 0:
            return input.copy()
        # During training, apply dropout
        # Generate a mask with the same shape as the input
        # The mask is a binary tensor where each element is 1 with probability (1 - p) and 0 with probability p
        # The mask is generated using a binomial distribution
        # The input is multiplied by the mask, and then divided by (1 - p) to maintain the expected value
        # This ensures that the output during training has the same expected value as the input without dropout
        mask = np.random.binomial(1, 1 - self.p, size=input.shape)
        mask = Tensor(mask, requires_grad=False)
        return input * mask / (1 - self.p)

    def get_config(self) -> dict:
        """
        Returns the configuration of the Dropout layer.
        """
        return {
            "type": self._name,
            "p": self.p
        }

    def get_parameters(self) -> dict:
        """
        Returns a dictionary of parameters for the Dropout layer.
        """
        return {
            "p": self.p
        }

    def set_parameters(self, params: dict) -> None:
        """
        Sets the parameters for the Dropout layer.
        """
        if "p" in params:
            self.p = params["p"]
        else:
            raise ValueError("Dropout layer requires 'p' parameter.")

    def __repr__(self) -> str:
        """
        Returns a string representation of the Dropout layer.
        """
        return f"{self._name}(p={self.p})"
