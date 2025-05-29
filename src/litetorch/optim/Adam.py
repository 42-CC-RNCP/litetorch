"""
litetorch/optim/Adam.py
This module defines the Adam optimizer, an adaptive learning rate optimizer

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

import numpy as np
from typing import List, Dict, Tuple
from litetorch.core.tensor import Tensor
from litetorch.optim.base import Optimizer


class Adam(Optimizer):
    def __init__(self,
                 parameters: List[Tensor],
                 lr: float = 0.01,
                 rhos: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.rhos = rhos
        self.eps = eps
        self.weight_decay = weight_decay
        self.m : Dict[str, np.ndarray] = {}
        self.v : Dict[str, np.ndarray] = {}
        self.t = 0
        self._name = "Adam"
        
    def step(self) -> None:
        """
        Perform a single optimization step using the Adam algorithm.
        This method updates the parameters based on their gradients and the learning rate.
        
        The Adam algorithm combines momentum and adaptive learning rates.
        """
        self.t += 1
        for param in self.parameters:
            if param.grad is None:
                continue
            
            # Apply weight decay
            if self.weight_decay != 0:
                param.grad += self.weight_decay * param.data
            
            # Initialize m and v if not already done
            if param not in self.m:
                self.m[param] = np.zeros_like(param.data)
            if param not in self.v:
                self.v[param] = np.zeros_like(param.data)
            
            # Update m and v
            self.m[param] = (self.rhos[0] * self.m[param] + 
                             (1 - self.rhos[0]) * param.grad)
            self.v[param] = (self.rhos[1] * self.v[param] + 
                             (1 - self.rhos[1]) * np.square(param.grad))
            
            # Compute bias-corrected m and v
            m_hat = self.m[param] / (1 - self.rhos[0] ** self.t)
            v_hat = self.v[param] / (1 - self.rhos[1] ** self.t)
            
            # Update parameter
            param.data -= (self.lr * m_hat) / (np.sqrt(v_hat) + self.eps)
