"""
tests/test_trainer.py
This is more close to regression test for the trainer.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.training.trainer import Trainer
from litetorch.nn.sequential import Sequential
from litetorch.nn.linear import Linear
from litetorch.nn.activation import Sigmoid
from litetorch.nn.loss import BinaryCrossEntropyLoss
from litetorch.optim.SGD import SGD
from litetorch.data.dataloader import DataLoader


def test_trainer_with_toy_data():
    # 1. Toy dataset (4 samples, 2 features)
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)  # XOR-like for binary test

    train_loader = DataLoader(X, y, batch_size=4, shuffle=False)
    val_loader = DataLoader(X, y, batch_size=4, shuffle=False)

    # 2. Model: Linear → Sigmoid
    model = Sequential(
        Linear(2, 1),
        Sigmoid()
    )

    # 3. Loss and Optimizer
    loss_fn = BinaryCrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)

    # 4. Trainer
    trainer = Trainer(model, optimizer, loss_fn, train_loader, max_epochs=10, val_loader=val_loader)

    # 5. Capture initial weights and run training
    initial_weight = model.layers[0].weight.data.copy()
    trainer.train()
    final_weight = model.layers[0].weight.data

    # 6. Assert: Loss should decrease and weights should change
    assert not np.allclose(initial_weight, final_weight), "Weights did not update"
    assert trainer.train_losses[-1] < trainer.train_losses[0], "Loss did not decrease"
