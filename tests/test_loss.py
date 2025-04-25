"""
tests/test_loss.py
Unit tests for loss functions in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.loss.mse import MSELoss
from litetorch.loss.cross_entropy import CrossEntropyLoss
from litetorch.loss.binary_cross_entropy import BinaryCrossEntropyLoss


def test_mse_loss():
    """
    Test Mean Squared Error (MSE) loss function.
    """
    mse_loss = MSELoss()
    output = np.array([[0.5, 0.2], [0.1, 0.4]])
    target = np.array([[0.0, 0.0], [1.0, 1.0]])

    # Forward pass
    loss_value = mse_loss.forward(output, target)
    expected_loss = ((output - target) ** 2).mean()

    assert np.isclose(loss_value, expected_loss), f"MSE Loss value mismatch: {loss_value} != {expected_loss}"

    # Backward pass
    grad = mse_loss.backward(output, target)
    expected_grad = 2 * (output - target) / output.size

    assert np.allclose(grad, expected_grad), f"MSE Loss gradient mismatch: {grad} != {expected_grad}"


def test_cross_entropy_loss():
    """
    Test Cross Entropy loss function.
    """

    cross_entropy_loss = CrossEntropyLoss()
    output = np.array([[0.1, 0.9], [0.8, 0.2]])
    target = np.array([1, 0])

    # Forward pass
    loss_value = cross_entropy_loss.forward(output, target)
    expected_loss = -np.log(output[np.arange(len(target)), target]).mean()

    assert np.isclose(loss_value, expected_loss), f"Cross Entropy Loss value mismatch: {loss_value} != {expected_loss}"

    # Backward pass
    grad = cross_entropy_loss.backward(output, target)
    expected_grad = output.copy()
    expected_grad[np.arange(len(target)), target] -= 1
    expected_grad /= len(target)

    assert np.allclose(grad, expected_grad), f"Cross Entropy Loss gradient mismatch: {grad} != {expected_grad}"


def test_binary_cross_entropy_loss():
    """
    Test Binary Cross Entropy loss function.
    """

    binary_cross_entropy_loss = BinaryCrossEntropyLoss()
    output = np.array([[0.1], [0.9]])
    target = np.array([[0], [1]])

    # Forward pass
    loss_value = binary_cross_entropy_loss.forward(output, target)
    expected_loss = -np.mean(target * np.log(output) + (1 - target) * np.log(1 - output))

    assert np.isclose(loss_value, expected_loss), f"Binary Cross Entropy Loss value mismatch: {loss_value} != {expected_loss}"

    # Backward pass
    grad = binary_cross_entropy_loss.backward(output, target)
    expected_grad = (output - target) / (output * (1 - output))

    assert np.allclose(grad, expected_grad), f"Binary Cross Entropy Loss gradient mismatch: {grad} != {expected_grad}"
