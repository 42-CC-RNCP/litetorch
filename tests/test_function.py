"""
litetorch/tests/test_function.py
Unit tests for all functions in the core module of the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-12
"""

import numpy as np
from litetorch.core.tensor import Tensor


def test_add_forward_broadcast():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=False)
    b = Tensor(np.array([[10.0, 20.0]]), requires_grad=False)  # (1,2)

    out = a + b
    expected = np.array([[11.0, 22.0], [13.0, 24.0]])

    assert np.allclose(out.data, expected), "Add forward with broadcast failed"


def test_add_backward_broadcast():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[10.0, 20.0]]), requires_grad=True)

    out = a + b  # shape = (2,2)
    out.backward(Tensor(np.ones_like(out.data)))

    # expected grad of a is just 1s
    expected_grad_a = np.ones((2, 2))
    # expected grad of b is sum over axis 0
    expected_grad_b = np.array([[2.0, 2.0]])  # because b was broadcast to 2 rows

    assert np.allclose(a.grad, expected_grad_a), "Add backward grad for a is incorrect"
    assert np.allclose(b.grad, expected_grad_b), "Add backward grad for b is incorrect"


def test_mul_backward_broadcast():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)  # (1,2)

    out = a * b  # shape = (2,2)
    out.backward(Tensor(np.ones_like(out.data)))  # gradient from loss = 1

    # grad_a = b.broadcasted → (2,2)
    expected_grad_a = np.array([[2.0, 3.0], [2.0, 3.0]])
    # grad_b = sum(a * 1) over axis=0
    expected_grad_b = np.array([[4.0, 6.0]])  # 1+3 and 2+4

    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)

def test_div_backward_broadcast():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)  # (1,2)

    out = a / b
    out.backward(Tensor(np.ones_like(out.data)))  # ∂L/∂out = 1

    expected_grad_a = np.array([[0.5, 1/3], [0.5, 1/3]])         # (2,2)
    expected_grad_b = np.array([[-1.0, -0.66666667]])            # (1,2) ← sum over axis 0

    assert np.allclose(a.grad, expected_grad_a, atol=1e-6)
    assert np.allclose(b.grad, expected_grad_b, atol=1e-6)
