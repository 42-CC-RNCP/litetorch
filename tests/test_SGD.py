"""
tests/test_SGD.py
Unit tests for the SGD optimizer in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.optim.SGD import SGD
from litetorch.core.tensor import Tensor


def test_basic_sgd():
    """
    Test basic functionality of the SGD optimizer.
    """
    # Create a simple tensor with requires_grad=True
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])

    # Initialize the SGD optimizer
    optimizer = SGD([param], lr=0.01)

    # Perform a step
    optimizer.step()

    # Check if the parameter is updated correctly
    expected_param = np.array([[0.999, 1.998]], dtype=np.float32)
    assert np.allclose(param.data, expected_param), "SGD step did not update the parameter correctly."


def test_sgd_with_momentum():
    """
    Test SGD optimizer with momentum.
    """
    from litetorch.optim.SGD import SGD
    from litetorch.core.tensor import Tensor
    import numpy as np

    # Create a tensor with gradient
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])

    # Initialize the optimizer with momentum
    optimizer = SGD([param], lr=0.01, momentum=0.9)

    # First step — no history, acts like vanilla SGD
    optimizer.step()

    # Expected result: same as SGD without momentum
    expected = np.array([[1.0 - 0.01 * 0.1, 2.0 - 0.01 * 0.2]])

    assert np.allclose(param.data, expected), f"First step mismatch. Got: {param.data}, Expected: {expected}"

    # Simulate second step with same gradient
    param.grad = np.array([[0.1, 0.2]])
    optimizer.step()

    # Now momentum_buffer = 0.1 from last time
    # New buffer = 0.9 * 0.1 + (1 - 0) * 0.1 = 0.09 + 0.1 = 0.19
    # param = old - lr * 0.19 = ...
    expected = np.array([
        [expected[0, 0] - 0.01 * 0.19, expected[0, 1] - 0.01 * 0.19 * 2]
    ])

    assert np.allclose(param.data, expected, atol=1e-6), f"Second step mismatch. Got: {param.data}, Expected: {expected}"


def test_sgd_with_weight_decay():
    """
    Test SGD optimizer with weight decay.
    """
    # Create a simple tensor with requires_grad=True
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])

    # Initialize the SGD optimizer with weight decay
    optimizer = SGD([param], lr=0.01, weight_decay=0.01)
    optimizer.step()

    # Check if the parameter is updated correctly
    expected_param = np.array([[1.0 - 0.01 * (0.1 + 0.01 * 1.0),
                                2.0 - 0.01 * (0.2 + 0.01 * 2.0)]])
    assert np.allclose(param.data, expected_param, atol=1e-6), (
        f"SGD with weight decay failed.\nExpected: {expected_param}, Got: {param.data}"
    )


def test_sgd_with_dampening():
    """
    Test SGD optimizer with dampening (takes effect from second step).
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    optimizer = SGD([param], lr=0.01, momentum=0.9, dampening=0.5)

    # Step 1: No dampening applied yet
    optimizer.step()

    # Step 2: dampening applies
    param.grad = np.array([[0.1, 0.2]])  # same grad
    optimizer.step()

    # momentum_buffer after step 2:
    buffer = 0.9 * np.array([[0.1, 0.2]]) + 0.5 * np.array([[0.1, 0.2]])  # [[0.14, 0.28]]
    expected = np.array([[0.999, 1.998]]) - 0.01 * buffer                 # [[0.9976, 1.9952]]

    assert np.allclose(param.data, expected, atol=1e-6), (
        f"SGD with dampening failed.\nExpected: {expected}, Got: {param.data}"
    )


def test_sgd_with_momentum_and_dampening():
    """
    Test SGD optimizer with both momentum and dampening.
    """
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])
    optimizer = SGD([param], lr=0.01, momentum=0.9, dampening=0.5)

    # Step 1
    optimizer.step()  # param = [0.999, 1.998]

    # Step 2
    param.grad = np.array([[0.1, 0.2]])
    optimizer.step()

    # Correct expected value
    buffer = 0.9 * np.array([[0.1, 0.2]]) + 0.5 * np.array([[0.1, 0.2]])  # [[0.14, 0.28]]
    expected = np.array([[0.999, 1.998]]) - 0.01 * buffer                 # [[0.9976, 1.9952]]

    assert np.allclose(param.data, expected, atol=1e-6), (
        f"SGD with momentum and dampening failed.\nExpected: {expected}, Got: {param.data}"
    )


def test_sgd_with_all_params():
    """
    Test SGD optimizer with multiple parameters.
    """
    # Create two simple tensors with requires_grad=True
    param1 = Tensor([[1.0, 2.0]], requires_grad=True)
    param2 = Tensor([[3.0, 4.0]], requires_grad=True)
    param1.grad = np.array([[0.1, 0.2]])
    param2.grad = np.array([[0.3, 0.4]])

    # Initialize the SGD optimizer
    optimizer = SGD([param1, param2], lr=0.01)

    # Perform a step
    optimizer.step()

    # Check if the parameters are updated correctly
    expected_param1 = [[1.0 - 0.01 * 0.1, 2.0 - 0.01 * 0.2]]
    expected_param2 = [[3.0 - 0.01 * 0.3, 4.0 - 0.01 * 0.4]]

    assert np.allclose(param1.data, expected_param1), (
        f"SGD step did not update param1 correctly.\nExpected: {expected_param1}, Got: {param1.data}"
    )
    assert np.allclose(param2.data, expected_param2), (
        f"SGD step did not update param2 correctly.\nExpected: {expected_param2}, Got: {param2.data}"
    )


def test_sgd_zero_grad():
    """
    Test zero_grad method of the SGD optimizer.
    """
    # Create a simple tensor with requires_grad=True
    param = Tensor([[1.0, 2.0]], requires_grad=True)
    param.grad = np.array([[0.1, 0.2]])

    # Initialize the SGD optimizer
    optimizer = SGD([param], lr=0.01)

    # Zero the gradients
    optimizer.zero_grad()

    # Check if the gradients are zeroed
    assert param.grad is not None, "Parameter grad should not be None."
    assert np.all(param.grad == 0), "Parameter grad should be zero after zero_grad."
