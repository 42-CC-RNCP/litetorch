"""
litetorch/utils/function.py
This module provides utility functions for tensor operations and mathematical computations.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute the softmax of an array along the specified axis.

    Parameters:
    - x: Input array.
    - axis: Axis along which to compute the softmax. Default is -1 (last axis).

    Returns:
    - Softmax of the input array.
    """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of an array.

    Parameters:
    - x: Input array.

    Returns:
    - Sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute the ReLU (Rectified Linear Unit) of an array.

    Parameters:
    - x: Input array.

    Returns:
    - ReLU of the input array.
    """
    return np.maximum(0, x)


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent of an array.

    Parameters:
    - x: Input array.

    Returns:
    - Hyperbolic tangent of the input array.
    """
    return np.tanh(x)


def mse_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Compute the Mean Squared Error (MSE) loss.

    Parameters:
    - predictions: Predicted values.
    - targets: True values.

    Returns:
    - MSE loss value.
    """
    return np.mean((predictions - targets) ** 2)


def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
    """
    Compute the Leaky ReLU of an array.

    Parameters:
    - x: Input array.
    - negative_slope: Slope for negative values. Default is 0.01.

    Returns:
    - Leaky ReLU of the input array.
    """
    return np.where(x > 0, x, x * negative_slope)

