"""
tests/test_split.py
Unit tests for the data split functions in the data module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from litetorch.data.split import *


def test_train_val_split():
    """
    Test the train_val_split function.
    """
    data = np.arange(10)
    train_data, val_data = train_val_split(data, val_size=0.2, shuffle=False)

    assert len(train_data) == 8, "Training data size mismatch."
    assert len(val_data) == 2, "Validation data size mismatch."


def test_train_val_test_split():
    """
    Test the train_val_test_split function.
    """
    data = np.arange(10)
    train_data, val_data, test_data = train_val_test_split(data, val_size=0.2, test_size=0.2, shuffle=False)

    assert len(train_data) == 6, "Training data size mismatch."
    assert len(val_data) == 2, "Validation data size mismatch."
    assert len(test_data) == 2, "Test data size mismatch."


def test_kfold_split():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)

    folds = list(kfold_split(X, y, n_splits=5, shuffle=False))

    assert len(folds) == 5
    for X_train, y_train, X_valid, y_valid in folds:
        assert len(X_train) + len(X_valid) == 10
        assert X_valid.shape[1] == 2
        assert len(y_train) + len(y_valid) == 10
        assert y_valid.shape[0] == X_valid.shape[0]
        assert y_train.shape[0] == X_train.shape[0]


def test_stratified_kfold_split():
    X = np.arange(20).reshape(10, 2)
    y = np.array([0, 1] * 5)

    folds = list(stratified_kfold_split(X, y, n_splits=5, shuffle=False))

    assert len(folds) == 5
    for X_train, y_train, X_valid, y_valid in folds:
        assert len(X_train) + len(X_valid) == 10
        assert X_valid.shape[1] == 2
        assert len(y_train) + len(y_valid) == 10
        assert y_valid.shape[0] == X_valid.shape[0]
        assert y_train.shape[0] == X_train.shape[0]
        assert np.array_equal(np.unique(y_valid), np.array([0, 1]))
        assert np.array_equal(np.unique(y_train), np.array([0, 1]))
