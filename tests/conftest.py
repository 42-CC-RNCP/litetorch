# tests/conftest.py
import sys
import os
import pytest
import numpy as np

# This file is used to configure pytest and set up the test environment.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def assert_no_nan_or_inf(x, name=""):
    """
    Assert that a numpy array does not contain NaN or Inf.
    """
    assert not np.isnan(x).any(), f"{name} contains NaN"
    assert not np.isinf(x).any(), f"{name} contains Inf"
