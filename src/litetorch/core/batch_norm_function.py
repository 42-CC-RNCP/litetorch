"""
litetorch/core/batch_norm_function.py
This module defines the Batch Normalization function for a neural network framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-27
"""

from typing import Tuple
import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.core.function import Function


class BatchNormFunction(Function):
    def __init__(self, epsilon: float = 1e-5):
        super().__init__()
        self.eps = epsilon
        self.mu : np.ndarray = None
        self.var : np.ndarray = None
        self.std : np.ndarray = None
        self.x_hat : np.ndarray = None
        
    def forward(self, input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        """
        Forward pass for batch normalization.
        
        Args:
            input (Tensor): Input tensor to be normalized.
            weight (Tensor): Scale parameter.
            bias (Tensor): Shift parameter.
        
        Returns:
            Tensor: Normalized output tensor.
        """
        self.x = input
        self.weight = weight
        self.bias = bias
        input_data = input.data
        self.mu = input_data.mean(axis=0)
        self.var = input_data.var(axis=0)
        self.std = np.sqrt(self.var + self.eps)
        
        self.x_hat = (input_data - self.mu) / self.std
        output_data = weight.data * self.x_hat + bias.data
        
        return Tensor(output_data, requires_grad=True)
    
    def backward(self, *grad_outputs: Tuple[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        dy : np.ndarray = grad_outputs[0].data # shape: (N, D)
        dbias = dy.sum(axis=0)  # shape: (D,)
        dweight = (self.x_hat * dy).sum(axis=0)
        
        N = dy.shape[0]
        dx_hat = dy * self.weight.data  # shape: (N, D)
        dvar = np.sum(dx_hat * (self.x.data - self.mu) * -0.5 * (self.var + self.eps)**(-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / self.std, axis=0) + dvar * np.mean(-2 * (self.x.data - self.mu), axis=0)

        dx = (dx_hat / self.std) + (dvar * 2 * (self.x.data - self.mu) / N) + (dmu / N)
        return Tensor(dx, requires_grad=True), Tensor(dweight, requires_grad=False), Tensor(dbias, requires_grad=False)
