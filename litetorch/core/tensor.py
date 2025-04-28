"""
litetorch/core/tensor.py
This module defines the Tensor class for basic tensor operations.
It's designed to be a lightweight alternative to PyTorch, focusing on core tensor functionalities.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from typing import List, Union
from litetorch.core.add_function import AddFunction
from litetorch.core.sub_function import SubFunction
from litetorch.core.mul_function import MulFunction
from litetorch.core.div_function import DivFunction
from litetorch.core.matmul_function import MatMulFunction


class Tensor:
    def __init__(self, data: Union[List, np.ndarray], requires_grad: bool = False):
        """
        Initialize a Tensor object.

        Parameters:
        - data: List or numpy array containing the tensor data.
        - requires_grad: Boolean indicating if gradients should be tracked.
        """

        self.auto_grad = requires_grad
        self.data = np.array(data) if isinstance(data, list) else data
        self.shape = self.data.shape
        self.grad = np.zeros_like(self.data) if requires_grad else None

        self.creator = None  # Function that created this tensor
        self.creation_args = None  # Arguments used to create this tensor

    def __repr__(self):
        """
        String representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.auto_grad})"

    def __add__(self, other: 'Tensor'):
        """
        Addition operator overload.
        """
        if isinstance(other, Tensor):
            result = AddFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Tensor' and '{}'".format(type(other)))

    def __sub__(self, other: 'Tensor'):
        """
        Subtraction operator overload.
        """
        if isinstance(other, Tensor):
            result = SubFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Tensor' and '{}'".format(type(other)))

    def __mul__(self, other: 'Tensor'):
        """
        Multiplication operator overload.
        """
        if isinstance(other, Tensor):
            result = MulFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Tensor' and '{}'".format(type(other)))

    def __div__(self, other: 'Tensor'):
        """
        Division operator overload.
        """
        if isinstance(other, Tensor):
            result = DivFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Tensor' and '{}'".format(type(other)))


    def __matmul__(self, other: 'Tensor'):
        """
        Matrix multiplication operator overload.
        """
        if isinstance(other, Tensor):
            result = MatMulFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Tensor' and '{}'".format(type(other)))
