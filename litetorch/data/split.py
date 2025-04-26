"""
litetorch/data/split.py
This module provides a function to split a dataset into training and validation sets.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

import numpy as np
from typing import Tuple, Generator


def train_val_split(
    data: np.ndarray,
    val_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split the dataset into training and validation sets.

    Parameters:
    - data: The dataset to be split.
    - val_size: The proportion of the dataset to include in the validation set.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: Seed for the random number generator.

    Returns:
    - A tuple containing the training and validation sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        np.random.shuffle(data)

    split_index = int(len(data) * (1 - val_size))
    train_data = data[:split_index]
    val_data = data[split_index:]

    return train_data, val_data


def train_val_test_split(
    data: np.ndarray,
    val_size: float = 0.2,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training, validation, and test sets.

    Parameters:
    - data: The dataset to be split.
    - val_size: The proportion of the dataset to include in the validation set.
    - test_size: The proportion of the dataset to include in the test set.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: Seed for the random number generator.

    Returns:
    - A tuple containing the training, validation, and test sets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        np.random.shuffle(data)

    val_index = int(len(data) * (1 - test_size - val_size))
    test_index = int(len(data) * (1 - test_size))

    # Ensure that the indices are within bounds
    val_index = min(val_index, len(data))
    test_index = min(test_index, len(data))

    # Split the data into train, val, and test sets
    train_data = data[:val_index]
    val_data = data[val_index:test_index]
    test_data = data[test_index:]

    return train_data, val_data, test_data


def kfold_split(
    X: np.ndarray,
    y: np.ndarray = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = None
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generate indices to split data into training and validation sets for K-Fold cross-validation.

    Parameters:
    - X: Features of the dataset.
    - y: Labels of the dataset (optional).
    - n_splits: Number of folds.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: Seed for the random number generator.

    Yields:
    - A tuple containing the training and validation indices for each fold.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    # balance the last fold if n_samples is not divisible by n_splits
    fold_sizes[:n_samples % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_indices = indices[start:stop]
        train_indices = np.concatenate((indices[:start], indices[stop:]))

        X_train, X_val = X[train_indices], X[val_indices]

        if y is not None:
            y_train, y_val = y[train_indices], y[val_indices]
            yield (X_train, y_train, X_val, y_val)
        else:
            yield (X_train, X_val, None, None)
        current = stop
