"""
tests/test_squential.py
Unit tests for the Sequential class in the nn module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.sequential import Sequential
from litetorch.nn.linear import Linear
from litetorch.nn.activation import ReLU


def test_sequential_representation():
    """
    Test the string representation of the Sequential class.
    """
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 2)
    )
    expected_repr = "Sequential(\n  Linear(in_features=10, out_features=5, bias=True),\n  ReLU(),\n  Linear(in_features=5, out_features=2, bias=True)\n)"
    assert str(model) == expected_repr, f"Expected: {expected_repr}, but got: {str(model)}"


def test_sequential_shape():
    """
    Test the backward pass of the Sequential class.
    """
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 2)
    )

    # --- Forward pass ---
    x = Tensor(np.random.randn(4, 10), requires_grad=True)  # batch_size=4, input_dim=10
    output = model(x)

    assert output.shape == (4, 2), "Output shape mismatch after forward."

     # --- Dummy loss ---
    dummy_grad_output = Tensor(np.ones_like(output.data))

    # --- Backward pass ---
    grad_input = model.backward(dummy_grad_output)

    # --- Check shape consistency ---
    assert grad_input.shape == (4, 10), "Grad input shape mismatch after backward."


def test_sequential_parameters():
    """
    Test the parameters of the Sequential class.
    """

    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 2)
    )

    params = model.parameters()
    assert len(params) == 4, "Expected 4 parameters in the Sequential model."
    assert all(isinstance(param, Tensor) for param in params), "All parameters should be Tensors."
    assert all(param.auto_grad for param in params), "All parameters should require gradients."
