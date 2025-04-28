"""
tests/test_loss.py
Unit tests for loss functions in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
import torch
import torch.nn.functional as F
from litetorch.core.tensor import Tensor
from litetorch.loss.mse import MSELoss
from litetorch.loss.cross_entropy import CrossEntropyLoss
from litetorch.loss.binary_cross_entropy import BinaryCrossEntropyLoss


def test_mse_loss():
    """
    Test Mean Squared Error (MSE) loss function.
    """
    mse_loss = MSELoss()
    output = Tensor(np.array([[0.5, 0.2], [0.1, 0.4]]), requires_grad=True)
    target = Tensor(np.array([[0.0, 0.0], [1.0, 1.0]]), requires_grad=True)

    # Forward pass
    loss_value = mse_loss.forward(output, target)
    expected_loss = ((output.data - target.data) ** 2).mean()

    assert np.isclose(loss_value.data, expected_loss), f"MSE Loss value mismatch: {loss_value.data} != {expected_loss}"

    # Backward pass
    loss_value.backward()
    grad = output.grad
    expected_grad = 2 * (output.data - target.data) / output.data.size

    assert np.allclose(grad, expected_grad), f"MSE Loss gradient mismatch: {grad} != {expected_grad}"


def test_cross_entropy_loss():
    """
    Test Cross Entropy loss function.
    """
    output = Tensor(np.array([[2.0, 1.0, 0.1]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), requires_grad=True)

    # Forward pass
    cross_entropy_loss = CrossEntropyLoss()
    loss_value = cross_entropy_loss.forward(output, target)
    # Apply softmax to the output to get probabilities
    probabilities = np.exp(output.data - np.max(output.data, axis=1, keepdims=True))
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    # Clip probabilities to avoid log(0)
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    # Calculate the Cross Entropy loss
    expected_loss = -np.sum(target.data * np.log(probabilities)) / target.data.shape[0]
    assert np.isclose(loss_value.data, expected_loss), f"Cross Entropy Loss value mismatch: {loss_value} != {expected_loss}"

    # Backward pass
    loss_value.backward()
    grad = output.grad
    expected_grad = (probabilities - target.data) / target.data.shape[0]

    assert np.allclose(grad, expected_grad), f"Cross Entropy Loss gradient mismatch: {grad} != {expected_grad}"


def test_cross_entropy_loss_with_pytorch():
    """
    Test Cross Entropy loss function.
    """
    output_np = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    target_np = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    output = Tensor(output_np, requires_grad=True)
    target = Tensor(target_np, requires_grad=True)

    # ----- litetorch -----
    cross_entropy_loss = CrossEntropyLoss()
    loss_lite = cross_entropy_loss.forward(output, target)
    loss_lite.backward()
    grad_lite = output.grad

    # ----- PyTorch -----
    output_torch = torch.tensor(output_np, requires_grad=True)
    target_torch = torch.tensor(target_np)
    loss_torch = F.cross_entropy(output_torch, target_torch)
    loss_torch.backward()
    grad_torch = output_torch.grad
    grad_torch = grad_torch.detach().numpy()
    loss_torch = loss_torch.detach().numpy()

    # Check loss values
    assert np.isclose(loss_lite.data, loss_torch), f"Cross Entropy Loss value mismatch: {loss_lite} != {loss_torch}"
    # Check gradients
    assert np.allclose(grad_lite, grad_torch), f"Cross Entropy Loss gradient mismatch: {grad_lite} != {grad_torch}"


def test_binary_cross_entropy_loss():
    """
    Test Binary Cross Entropy loss function.
    """

    bce_loss = BinaryCrossEntropyLoss()
    output_np = np.array([[0.1], [0.9]])
    target_np = np.array([[0], [1]])
    output = Tensor(output_np, requires_grad=True)
    target = Tensor(target_np, requires_grad=True)

    # Forward pass
    loss_value = bce_loss.forward(output, target)
    expected_loss = -np.mean(target_np * np.log(output_np) + (1 - target_np) * np.log(1 - output_np))

    assert np.isclose(loss_value.data, expected_loss), f"Binary Cross Entropy Loss value mismatch: {loss_value} != {expected_loss}"

    # Backward pass
    loss_value.backward()
    grad = output.grad
    expected_grad = (output.data - target.data) / (output.data * (1 - output.data) * output.shape[0])

    assert np.allclose(grad, expected_grad), f"Binary Cross Entropy Loss gradient mismatch: {grad} != {expected_grad}"
