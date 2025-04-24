"""
litetorch/tests/test_linear.py
Unit tests for the Linear class in the nn module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.linear import Linear


def test_linear_init():
    """
    Test Linear layer initialization.
    """
    linear = Linear(10, 5)
    assert linear.in_features == 10, "Input features should be 10."
    assert linear.out_features == 5, "Output features should be 5."
    assert linear.bias is True, "Bias should be True by default."
    assert linear.weight.shape == (10, 5), "Weight shape should be (10, 5)."
    assert linear.bias_tensor.shape == (1, 5), "Bias tensor shape should be (1, 5)."
    assert linear._name == "Linear", "Module name should be 'Linear'."
    assert linear._parameters["weight"] is linear.weight, "Weight parameter should be the same as the weight tensor."
    assert linear._parameters["bias"] is linear.bias_tensor, "Bias parameter should be the same as the bias tensor."


def test_linear_forward_shape():
    """
    Test forward pass of Linear layer.
    """
    linear = Linear(10, 5)
    x = Tensor([[1.0] * 10])
    output = linear(x)
    assert output.shape == (1, 5), "Output shape should be (1, 5)."
    assert output.auto_grad is True, "Output should require gradients."
    assert linear._name == "Linear", "Module name should be 'Linear'."


def test_linear_forward_no_bias_shape():
    """
    Test forward pass of Linear layer without bias.
    """
    linear = Linear(10, 5, bias=False)
    x = Tensor([[1.0] * 10])
    output = linear(x)
    assert output.shape == (1, 5), "Output shape should be (1, 5)."
    assert output.auto_grad is True, "Output should require gradients."
    assert linear._name == "Linear", "Module name should be 'Linear'."
    assert linear.bias_tensor is None, "Bias tensor should be None when bias is False."
    assert linear._parameters["bias"] is None, "Bias parameter should be None when bias is False."
    assert linear.weight.shape == (10, 5), "Weight shape should be (10, 5)."
    assert linear._parameters["weight"] is linear.weight, "Weight parameter should be the same as the weight tensor."
    assert linear._parameters["bias"] is None, "Bias parameter should be None when bias is False."
    assert linear._name == "Linear", "Module name should be 'Linear'."


def test_linear_backward_shape():
    """
    Test backward pass of Linear layer.
    """
    linear = Linear(10, 5)
    x = Tensor([[1.0] * 10], requires_grad=True)
    output = linear(x)
    grad_output = Tensor([[1.0] * 5], requires_grad=True)
    linear.backward(grad_output)
    assert linear.weight.grad is not None, "Weight gradient should not be None."
    assert linear.bias_tensor.grad is not None, "Bias gradient should not be None."
    assert linear.weight.grad.shape == (10, 5), "Weight gradient shape should be (10, 5)."
    assert linear.bias_tensor.grad.shape == (1, 5), "Bias gradient shape should be (1, 5)."
    assert grad_output.shape == (1, 5), "Gradient output shape should be (1, 5)."
    assert grad_output.auto_grad is True, "Gradient output should require gradients."
    assert (grad_output.data == 1.0).all(), "Gradient output data should be all ones."

def test_linear_forward_correctness():
    """
    Test forward pass correctness of Linear layer.
    """
    W = np.ones((3, 2), dtype=np.float32)
    b = np.ones((1, 2), dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    linear = Linear(3, 2)
    linear.weight.data = W
    linear.bias_tensor.data = b

    output = linear(Tensor(x))
    expected_output = np.dot(x, W) + b
    assert np.allclose(output.data, expected_output), "Forward pass output is incorrect."
    assert output.auto_grad is True, "Output should require gradients."

def test_linear_forward_no_bias_correctness():
    """
    Test forward pass correctness of Linear layer without bias.
    """
    W = np.ones((3, 2), dtype=np.float32)
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    linear = Linear(3, 2, bias=False)
    linear.weight.data = W

    output = linear(Tensor(x))
    expected_output = np.dot(x, W)
    assert np.allclose(output.data, expected_output), "Forward pass output is incorrect."
    assert output.auto_grad is True, "Output should require gradients."
    assert linear.bias_tensor is None, "Bias tensor should be None when bias is False."
    assert linear._parameters["bias"] is None, "Bias parameter should be None when bias is False."

def test_linear_backward_correctness():
    """
    Test backward pass correctness of Linear layer.
    dL/dW = x.T @ dL_dy
    dL/db = sum(dL_dy)
    dL/dx = dL_dy @ W.T
    """
    W = np.ones((3, 2), dtype=np.float32)
    b = np.zeros((1, 2), dtype=np.float32)
    x_data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    grad_out_data = np.array([[1.0, 1.0]], dtype=np.float32)

    x = Tensor(x_data, requires_grad=True)
    grad_out = Tensor(grad_out_data)

    linear = Linear(3, 2)
    linear.weight.data[:] = W
    linear.bias_tensor.data[:] = b

    linear.forward(x)
    dx = linear.backward(grad_out)

    expected_dW = x_data.T @ grad_out_data   # shape: (3,2)
    expected_db = grad_out_data              # shape: (1,2)
    expected_dx = grad_out_data @ W.T        # shape: (1,3)

    assert np.allclose(linear.weight.grad, expected_dW), "dW mismatch"
    assert np.allclose(linear.bias_tensor.grad, expected_db), "db mismatch"
    assert np.allclose(dx.data, expected_dx), "dx mismatch"
