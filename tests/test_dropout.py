"""
tests/test_dropout.py
Unit tests for the Dropout class in the nn module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-25
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.sequential import Sequential
from litetorch.nn.linear import Linear
from litetorch.nn.activation import ReLU
from litetorch.nn.dropout import Dropout


def test_dropout_representation():
    """
    Test the string representation of the Dropout class.
    """
    dropout = Dropout(p=0.5)
    expected_repr = "Dropout(p=0.5)"
    assert str(dropout) == expected_repr, f"Expected: {expected_repr}, but got: {str(dropout)}"


def test_dropout_shape():
    """
    Test the forward pass of the Dropout class.
    """
    dropout = Dropout(p=0.5)
    x = Tensor(np.random.randn(4, 10), requires_grad=True)  # batch_size=4, input_dim=10
    output = dropout(x)

    assert output.shape == (4, 10), "Output shape mismatch after forward."


def test_dropout_training_mode_behavior():
    dropout = Dropout(p=0.5)
    x = Tensor(np.ones((1000, 10)), requires_grad=True)

    dropout.train()
    out = dropout(x).data

    dropout_rate = np.mean(out == 0)
    assert 0.45 < dropout_rate < 0.55, f"Dropout rate abnormal: {dropout_rate}"

    unique_values = np.unique(out[out != 0])
    assert np.allclose(unique_values, 2.0), f"Unexpected non-zero values: {unique_values}"


def test_dropout_eval_mode_behavior():
    dropout = Dropout(p=0.5)
    x = Tensor(np.random.randn(10, 10), requires_grad=True)

    dropout.eval()
    out = dropout(x)
    assert np.allclose(out.data, x.data), "Dropout in eval mode should return input unchanged"
