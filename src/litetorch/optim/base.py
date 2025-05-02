"""
litetorch/optim/base.py
This module defines the base class for optimizers in a neural network framework.
The base class provides a common interface for all optimizers, including methods for
updating parameters and zeroing gradients.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

from abc import ABC, abstractmethod
from typing import List
from litetorch.core.tensor import Tensor


class Optimizer(ABC):
    def __init__(self, parameters: List[Tensor], lr: float = 0.01) -> None:
        """
        Initialize the optimizer with parameters and learning rate.

        Parameters:
        - parameters: List of parameters (Tensors) to optimize.
        - lr: Learning rate for the optimizer.
        """
        self.parameters = parameters
        self.lr = lr
        self._name = "Optimizer"

    @abstractmethod
    def step(self) -> None:
        """
        Perform a single optimization step. This method should be overridden by subclasses
        to implement the specific optimization algorithm.
        """
        pass

    def zero_grad(self) -> None:
        """
        Zero the gradients of all parameters. This is typically called before the backward pass
        to clear old gradients.
        """
        for param in self.parameters:
            if param.grad is not None:
                param.grad[:] = 0.0
