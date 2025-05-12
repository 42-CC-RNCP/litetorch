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


def test_sub_backward_broadcast():
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)
    b = Tensor(np.array([[2.0, 3.0]]), requires_grad=True)  # (1,2)

    out = a - b
    out.backward(Tensor(np.ones_like(out.data)))  # ∂L/∂out = 1

    expected_grad_a = np.ones_like(a.data)               # shape (2,2)
    expected_grad_b = np.array([[-2.0, -2.0]])            # shape (1,2), sum of two -1s per column

    assert np.allclose(a.grad, expected_grad_a)
    assert np.allclose(b.grad, expected_grad_b)


def test_softmax():
    from litetorch.core.softmax_function import SoftmaxFunction

    a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    softmax = SoftmaxFunction(dim=1)
    out = softmax(a)

    exp = np.exp(np.array([[1.0, 2.0]]))
    expected = exp / np.sum(exp, axis=1, keepdims=True)
    assert np.allclose(out.data, expected, atol=1e-6), "Softmax forward failed"

    grad_from_loss = Tensor(np.array([[1.0, 0.0]]))  # ∂L/∂softmax = [1, 0]
    out.backward(grad_from_loss)

    s = expected[0]
    # Jv = s * (v - dot(v, s))
    v = np.array([1.0, 0.0])
    dot = np.dot(s, v)  # = s[0]
    grad_expected = s * (v - dot)

    assert np.allclose(a.grad, grad_expected.reshape(1, -1), atol=1e-6), f"Softmax backward incorrect.\nExpected: {grad_expected}\nGot: {a.grad}"

def test_tahn():
    from litetorch.core.tanh_function import TanhFunction

    a = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    tanh = TanhFunction()
    out = tanh(a)

    expected = np.tanh(a.data)
    assert np.allclose(out.data, expected, atol=1e-6), "Tanh forward failed"

    grad_from_loss = Tensor(np.array([[1.0, 0.0]]))  # ∂L/∂tanh = [1, 0]
    out.backward(grad_from_loss)

    # Jv = (1 - tanh^2(x)) * v
    v = np.array([1.0, 0.0])
    grad_expected = (1 - np.tanh(a.data) ** 2) * v

    assert np.allclose(a.grad, grad_expected.reshape(1, -1), atol=1e-6), f"Tanh backward incorrect.\nExpected: {grad_expected}\nGot: {a.grad}"


def test_bce_shape_handling():
    from litetorch.core.binary_cross_entropy_function import BinaryCrossEntropyFunction

    # Input and target with compatible but different shapes
    input = Tensor(np.array([[0.9], [0.1], [0.8]], dtype=np.float32), requires_grad=True)  # shape (3,1)
    target_flat = Tensor(np.array([1, 0, 1], dtype=np.float32))                            # shape (3,)
    target_column = Tensor(np.array([[1], [0], [1]], dtype=np.float32))                   # shape (3,1)

    # Should work even though shapes differ
    loss_fn = BinaryCrossEntropyFunction()
    loss_flat = loss_fn(input, target_flat)
    loss_column = loss_fn(input, target_column)

    # Forward values must be the same
    assert np.allclose(loss_flat.data, loss_column.data, atol=1e-6), "BCE loss mismatch on shape-normalized targets"

    # Backward should not fail
    loss_flat.backward()
    assert input.grad.shape == input.shape, "Gradient shape mismatch after BCE backward"

