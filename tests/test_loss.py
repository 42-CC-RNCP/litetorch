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
from litetorch.nn.loss import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from conftest import assert_no_nan_or_inf


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
    output = Tensor(np.array([[2.0, 1.0, 0.1]], dtype=np.float32), requires_grad=True)
    target = Tensor(np.array([0]), requires_grad=False)  # class index

    cross_entropy_loss = CrossEntropyLoss()
    loss_value = cross_entropy_loss(output, target)

    # 手動計算 softmax + NLL
    logits = output.data
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    expected_loss = -np.log(probs[0, 0])  # class index 0

    assert np.isclose(loss_value.data, expected_loss, atol=1e-6)

    loss_value.backward()
    expected_grad = probs.copy()
    expected_grad[0, 0] -= 1  # subtract 1 for correct class
    assert np.allclose(output.grad, expected_grad, atol=1e-6)


def test_cross_entropy_loss_with_pytorch():
    output_np = np.array([[2.0, 1.0, 0.1]], dtype=np.float32)
    target_np = np.array([0])  # class index
    output = Tensor(output_np, requires_grad=True)
    target = Tensor(target_np)

    cross_entropy_loss = CrossEntropyLoss()
    loss_lite = cross_entropy_loss(output, target)
    loss_lite.backward()
    grad_lite = output.grad

    # PyTorch
    output_torch = torch.tensor(output_np, requires_grad=True)
    target_torch = torch.tensor(target_np)
    loss_torch = F.cross_entropy(output_torch, target_torch)
    loss_torch.backward()
    grad_torch = output_torch.grad.detach().numpy()
    loss_torch_val = loss_torch.item()

    assert np.isclose(loss_lite.data, loss_torch_val, atol=1e-6)
    assert np.allclose(grad_lite, grad_torch, atol=1e-6)
    

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


def test_binary_cross_entropy_loss_edge_cases():
    """
    Test Binary Cross Entropy loss function with extreme predictions near 0 and 1.
    """

    bce_loss = BinaryCrossEntropyLoss()
    output_np = np.array([[0.0], [1.0], [1e-20], [1 - 1e-20]])
    target_np = np.array([[0], [1], [0], [1]])
    output = Tensor(output_np, requires_grad=True)
    target = Tensor(target_np, requires_grad=False)

    # Forward pass should not produce NaN or Inf
    loss_value = bce_loss.forward(output, target)
    assert_no_nan_or_inf(loss_value.data, "Loss")

    # Backward pass should also not produce NaN or Inf
    loss_value.backward()
    grad = output.grad
    assert_no_nan_or_inf(output.grad, "Grad")
