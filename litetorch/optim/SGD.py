"""
litetorch/optim/SGD.py
This module defines the SGD (Stochastic Gradient Descent) optimizer for training neural networks.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from typing import List, Dict
from litetorch.core.tensor import Tensor
from litetorch.optim.base import Optimizer

class SGD(Optimizer):
    def __init__(self,
                 parameters: List[Tensor],
                 lr: float = 0.01,
                 momentum: float = 0.0,
                 dampening: float = 0.0,
                 weight_decay: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.momentum_buffer : Dict[str, np.ndarray] = {}
        self.dampening = dampening
        self.weight_decay = weight_decay
        self._name = "SGD"

    def step(self) -> None:
        """
        Perform a single optimization step using SGD.
        This method updates the parameters based on their gradients and the learning rate.
        weight_decay: L2 regularization term.

        formula:
        1. Apply weight decay:
           θ = θ - η * λ * θ
        2. Update momentum:
           v = μ * v + (1 - dampening) * ∇L(θ)
        3. Update parameters:
            θ = θ - η⋅∇L(θ)
        """
        for param in self.parameters:
            if param.grad is not None:
                # Apply weight decay
                if self.weight_decay != 0:
                    param.grad += self.weight_decay * param.data

                # Update momentum
                if self.momentum != 0:
                    if param not in self.momentum_buffer:
                        self.momentum_buffer[param] = np.copy(param.grad)
                    else:
                        self.momentum_buffer[param] = (
                            self.momentum * self.momentum_buffer[param] +
                            (1 - self.dampening) * param.grad
                        )
                    param.data -= self.lr * self.momentum_buffer[param]
                else:
                    param.data -= self.lr * param.grad
