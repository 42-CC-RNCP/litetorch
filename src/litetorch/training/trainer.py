"""
litetorch/training/trainer.py
The Trainer class is responsible for managing the training process of a model.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-03
"""

import numpy as np
from typing import Callable, Optional, List
from litetorch.nn.module import Module
from litetorch.core.tensor import Tensor
from litetorch.optim.base import Optimizer
from litetorch.nn.loss import Loss
from litetorch.data.dataloader import DataLoader


class Trainer:
    def __init__(self,
                 model: Module,
                 optimizer: Optimizer,
                 loss_fn: Loss,
                 train_loader: DataLoader,
                 max_epochs: int = 10,
                 val_loader: Optional[DataLoader] = None,
                 early_stopping: Optional[int] = None,
                 clip_grad: Optional[float] = None,
                 callbacks: Optional[List[Callable]] = None,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping
        self.clip_grad = clip_grad
        # TODO: Implement callback functionality
        self.callbacks = callbacks if callbacks is not None else []
        
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.no_improv_epochs = 0
        
    def train(self):
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            # Train for one epoch
            train_loss = self._train_one_epoch()
            self.train_losses.append(train_loss)
            val_loss = self._validate_one_epoch() if self.val_loader else None
            print(f"Epoch {epoch}/{self.max_epochs} - Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Epoch {epoch}/{self.max_epochs} - Val Loss: {val_loss:.4f}")
            else:
                print()
            if self.val_loader:
                self.val_losses.append(val_loss)
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.no_improv_epochs = 0
                else:
                    self.no_improv_epochs += 1
                    if self.early_stopping and self.no_improv_epochs >= self.early_stopping:
                        print(f"Early stopping at epoch {epoch}")
                        break
    
    def save_model(self, path: str = "saved_model.json") -> None:
        if hasattr(self.model, "save"):
            self.model.save(path)
        else:
            raise TypeError("Model does not support saving. Make sure it inherits from SaveLoadMixin.")
    
    def _train_one_epoch(self) -> float:
        epoch_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad:
                # TODO: Implement gradient clipping
                pass
            self.optimizer.step()
            epoch_loss += loss.data.item()
        # average training loss = total_loss / number of batches
        return epoch_loss / len(self.train_loader)
    
    def _validate_one_epoch(self) -> float:
        total_loss = 0.0
        for X_batch, y_batch in self.val_loader:
            # Forward pass
            y_pred = self.model(X_batch)
            loss = self.loss_fn(y_pred, y_batch)
            total_loss += loss.data.item()
        # average validation loss = total_loss / number of batches
        average_val_loss = total_loss / len(self.val_loader)
        return average_val_loss
