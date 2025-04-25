"""
litetorch/loss/base.py
This module defines the base class for loss functions in a neural network framework.
The base class provides a common interface for all loss functions, including methods for
calculating the loss and its gradient with respect to the model's output.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self) -> None:
        """
        Initialize the loss function. This method can be overridden by subclasses
        to perform any necessary setup.
        """
        self._name = "Loss"

    @abstractmethod
    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        """
        Calculate the loss given the model's output and the target values.

        Parameters:
        - output: The model's output (predictions).
        - target: The true target values.

        Returns:
        - The calculated loss value.
        """
        pass

    @abstractmethod
    def backward(self, output, target) -> float:
        """
        Calculate the gradient of the loss with respect to the model's output.

        Parameters:
        - output: The model's output (predictions).
        - target: The true target values.

        Returns:
        - The gradient of the loss with respect to the model's output.
        """
        pass

    def __call__(self, output: np.ndarray, target: np.ndarray) -> float:
        """
        Call the loss function to calculate the loss and its gradient.

        Parameters:
        - output: The model's output (predictions).
        - target: The true target values.

        Returns:
        - The calculated loss value.
        """
        return self.forward(output, target)

    def __str__(self) -> str:
        """
        Return a string representation of the loss function.

        Returns:
        - A string representing the loss function.
        """
        return self._name

    def __repr__(self) -> str:
        """
        Return a string representation of the loss function for debugging.

        Returns:
        - A string representing the loss function.
        """
        return f"{self.__class__.__name__}()"
