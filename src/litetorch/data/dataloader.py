"""
litetorch/data/dataloader.py
This module provides a DataLoader class for loading data from a dataset and batching, shuffling, and
iterating over the data.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-03
"""

import numpy as np
from typing import Generator
from litetorch.core.tensor import Tensor


class DataLoader:

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray = None,
                 shuffle: bool = False,
                 auto_batch: bool = True,
                 batch_size: int = 32,
                 drop_last: bool = False):
        self.X = X
        self.y = y
        self.auto_batch = auto_batch
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.n_samples = len(X)

        # if the dataset is small, set batch_size to full dataset size
        if self.n_samples <= 1000:
            self.batch_size = self.n_samples
        elif self.auto_batch:
            self.batch_size = min(self.batch_size, self.n_samples)

    def __iter__(self) -> Generator[tuple[Tensor, Tensor], None, None]:
        indices = list(range(self.n_samples))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, self.n_samples, self.batch_size):
            end = start + self.batch_size
            if end > self.n_samples:
                if self.drop_last:
                    break
                else:
                    end = self.n_samples
            yield Tensor(self.X[indices[start:end]], requires_grad=False), \
                Tensor(self.y[indices[start:end]], requires_grad=False) if self.y is not None else None


if __name__ == "__main__":
    # Example usage
    X = np.random.rand(2000, 10)
    y = np.random.randint(0, 2, size=(2000,))
    dataloader = DataLoader(X, y, shuffle=True, batch_size=32)
    for batch_X, batch_y in dataloader:
        print("Batch X shape:", batch_X.shape)
        print("Batch y shape:", batch_y.shape)
        break  # Remove this line to iterate through all batches
