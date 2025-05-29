"""
litetorch/optim/RMSprop.py
This module defines the RMSprop optimizer for training neural networks.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

import numpy as np
from typing import List, Dict
from litetorch.core.tensor import Tensor
from litetorch.optim.base import Optimizer


class RMSprop(Optimizer):
    def __init__(self,
                 parameters: List[Tensor],
                 lr: float = 0.01,
                 rho: float = 0.9,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        """
        Initialize the RMSprop optimizer.

        Parameters:
        - parameters: List of parameters (Tensors) to optimize.
        - lr: Learning rate for the optimizer.
        - rho: Decay factor for the moving average of squared gradients.
        - eps: Small value to avoid division by zero.
        - weight_decay: L2 regularization term.
        """
        super().__init__(parameters, lr)
        self.rho = rho
        self.eps = eps
        self.weight_decay = weight_decay
        self._name = "RMSprop"
        self.cache: Dict[str, np.ndarray] = {}
        
    def step(self) -> None:
        """
        formula:
        1. If weight decay is used, apply it to the gradient:
            grad = grad + weight_decay * param
        2. Update cache: 
            cache = rho * cache + (1 - rho) * grad^2
        3. Update parameters:
            param = param - lr * grad / (sqrt(cache) + eps)
        """
        for param in self.parameters:
            if param.grad is None:
                continue
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad += self.weight_decay * param.data
            
            # Initialize cache if not already done
            if param not in self.cache:
                self.cache[param] = np.zeros_like(param.data)
            # Update cache with squared gradients
            self.cache[param] = (self.rho * self.cache[param] +
                                 (1 - self.rho) * np.square(param.grad))
            # Update parameters
            param.data -= self.lr * param.grad / (np.sqrt(self.cache[param]) + self.eps)
