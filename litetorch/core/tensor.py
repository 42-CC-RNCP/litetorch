"""
litetorch/core/tensor.py
This module defines the Tensor class for basic tensor operations.
It's designed to be a lightweight alternative to PyTorch, focusing on core tensor functionalities.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from typing import List, Union


class Tensor:
    def __init__(self, data: Union[List, np.ndarray], requires_grad: bool = False):
        """
        Initialize a Tensor object.

        Parameters:
        - data: List or numpy array containing the tensor data.
        - requires_grad: Boolean indicating if gradients should be tracked.
        """

        self.auto_grad = requires_grad
        self.data = np.array(data) if isinstance(data, list) else data
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self._backward = lambda: None
        self._prev = set()
        self._op = ""

    def __repr__(self):
        """
        String representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.auto_grad})"
