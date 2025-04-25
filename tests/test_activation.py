"""
litetorch/tests/test_activation.py
Unit tests for the activation functions in the nn module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.activation import *


def test_relu_forward():
    """
    Test forward pass of ReLU activation function.
    """
    relu = ReLU()
    x = Tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
    output = relu(x)
    expected_output = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 4.0]])
    assert np.array_equal(output.data, expected_output), "ReLU output is incorrect."


def test_sigmoid_forward():
    """
    Test forward pass of Sigmoid activation function.
    """
    sigmoid = Sigmoid()
    x = Tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
    output = sigmoid(x)
    expected_output = 1 / (1 + np.exp(-x.data))
    assert np.allclose(output.data, expected_output), "Sigmoid output is incorrect."


def test_tanh_forward():
    """
    Test forward pass of Tanh activation function.
    """
    tanh = Tanh()
    x = Tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
    output = tanh(x)
    expected_output = np.tanh(x.data)
    assert np.allclose(output.data, expected_output), "Tanh output is incorrect."


def test_softmax_forward():
    """
    Test forward pass of Softmax activation function.
    """
    softmax = Softmax(dim=1)
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    output = softmax(x)
    expected_output = np.exp(x.data - np.max(x.data, axis=1, keepdims=True))
    expected_output /= np.sum(expected_output, axis=1, keepdims=True)
    assert np.allclose(output.data, expected_output), "Softmax output is incorrect."


def test_leaky_relu_forward():
    """
    Test forward pass of Leaky ReLU activation function.
    """
    leaky_relu = LeakyReLU(negative_slope=0.01)
    x = Tensor([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]])
    output = leaky_relu(x)
    expected_output = np.where(x.data > 0, x.data, 0.01 * x.data)
    assert np.array_equal(output.data, expected_output), "Leaky ReLU output is incorrect."
