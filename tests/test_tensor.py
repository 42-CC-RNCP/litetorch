"""
tests/test_tensor.py
Unit tests for the Tensor class in the core module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.core.tensor import Tensor


def test_tensor_init_from_list():
    """
    Test Tensor initialization from a list.
    """
    data = [[1, 2], [3, 4]]
    tensor = Tensor(data)
    assert isinstance(tensor.data, np.ndarray), "Tensor data should be a numpy array."
    assert tensor.shape == (2, 2), "Tensor shape should be (2, 2)."
    assert tensor.auto_grad is False, "Tensor should not track gradients by default."
    assert np.array_equal(
        tensor.data, np.array(data)
    ), "Tensor data should match the input list."


def test_tensor_init_from_numpy():
    """
    Test Tensor initialization from a numpy array.
    """
    data = np.array([[1, 2], [3, 4]])
    tensor = Tensor(data)
    assert isinstance(tensor.data, np.ndarray), "Tensor data should be a numpy array."
    assert tensor.shape == (2, 2), "Tensor shape should be (2, 2)."
    assert np.array_equal(
        tensor.data, data
    ), "Tensor data should match the input numpy array."


def test_tensor_init_from_list_with_autograd():
    """
    Test Tensor initialization from a list with requires_grad set to True.
    """
    data = [[1, 2], [3, 4]]
    tensor = Tensor(data, requires_grad=True)
    assert tensor.auto_grad is True, "Tensor should track gradients."
    assert tensor.grad.shape == (2, 2), "Tensor gradient shape should match data shape."
    assert (tensor.grad == 0).all(), "Tensor gradient should be initialized to zeros."


def test_tensor_init_from_numpy_with_autograd():
    """
    Test Tensor initialization from a numpy array with requires_grad set to True.
    """
    data = np.array([[1, 2], [3, 4]])
    tensor = Tensor(data, requires_grad=True)
    assert tensor.auto_grad is True, "Tensor should track gradients."
    assert tensor.grad.shape == (2, 2), "Tensor gradient shape should match data shape."
    assert (tensor.grad == 0).all(), "Tensor gradient should be initialized to zeros."


def test_tensor_repr():
    """
    Test the string representation of the Tensor object.
    """
    data = [[1, 2], [3, 4]]
    tensor = Tensor(data)
    expected_repr = f"Tensor(data={tensor.data}, shape={tensor.shape}, requires_grad={tensor.auto_grad})"
    assert (
        repr(tensor) == expected_repr
    ), "Tensor representation does not match expected format."
