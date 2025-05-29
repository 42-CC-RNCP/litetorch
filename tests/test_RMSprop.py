"""
tests/test_RMSprop.py
Unit tests for the RMSprop optimizer in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

import numpy as np
from litetorch.optim.RMSprop import RMSprop
from litetorch.core.tensor import Tensor


def test_basic_rmsprop():
    """
    Test basic functionality of the RMSprop optimizer.
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    
    optimizer = RMSprop([param], lr=0.01, rho=0.9, eps=1e-8, weight_decay=0.0)
    optimizer.step()
    
    # Expected calculation:
    # grad = [[0.1, 0.2]]
    # cache = 0.1 * 0 + 0.9 * grad^2 = [[0.001, 0.004]]
    # updated_param = param - 0.01 * grad / (sqrt(cache) + eps)
    # sqrt(cache) = [[0.03162, 0.06325]]
    # update = [[0.01 * 0.1 / 0.03162, 0.01 * 0.2 / 0.06325]] ≈ [[0.03162, 0.03162]]
    expected_param = np.array([[1.0 - 0.03162, 2.0 - 0.03162]], dtype=np.float32)
    
    assert np.allclose(param.data, expected_param, atol=1e-5), "RMSprop step did not update the parameter correctly."


def test_rmsprop_with_weight_decay():
    """
    Test RMSprop optimizer with weight decay.
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    
    optimizer = RMSprop([param], lr=0.01, rho=0.9, eps=1e-8, weight_decay=0.01)
    optimizer.step()
    
    # grad = [[0.11, 0.22]]
    # cache = [[0.00121, 0.00484]]
    # sqrt = [[0.03478, 0.06957]]
    # update = [[0.031623, 0.031623]]
    expected_param = np.array([[0.968377, 1.968377]], dtype=np.float32)
    
    assert np.allclose(param.data, expected_param, atol=1e-5), "RMSprop with weight decay did not update the parameter correctly."


def test_rmsprop_with_cache():
    """
    Test RMSprop optimizer with cache initialization and update.
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    
    optimizer = RMSprop([param], lr=0.01)
    optimizer.step()
    
    assert param in optimizer.cache, "Cache was not initialized for the parameter."
    assert optimizer.cache[param].shape == param.data.shape, "Cache shape does not match parameter shape."
    
    # Do another step to ensure cache updates
    optimizer.step()
    assert np.all(optimizer.cache[param] >= 0), "Cache should contain non-negative values."


def test_rmsprop_with_rho():
    """
    Test RMSprop optimizer with different rho values.
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    
    optimizer = RMSprop([param], lr=0.01, rho=0.5, eps=1e-8)
    optimizer.step()
    
    # cache = 0.5 * 0 + 0.5 * grad^2 = [[0.005, 0.02]]
    # sqrt = [[0.0707, 0.1414]]
    # update = [[0.01 * 0.1 / 0.0707, 0.01 * 0.2 / 0.1414]] ≈ [[0.01414, 0.01414]]
    expected_param = np.array([[1.0 - 0.01414, 2.0 - 0.01414]], dtype=np.float32)
    
    assert np.allclose(param.data, expected_param, atol=1e-5), "RMSprop with rho did not update the parameter correctly."
