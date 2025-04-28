"""
litetorch/core/tensor.py
This module defines the Tensor class for basic tensor operations.
It's designed to be a lightweight alternative to PyTorch, focusing on core tensor functionalities.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

import numpy as np
from typing import List, Union, Tuple
from litetorch.core.function import Function


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

        self.creator : Function = None  # The function that created this tensor
        self.inputs : List[Tensor] = []  # Arguments used to create this tensor

    def backward(self, grad_output: 'Tensor' = None) -> None:
        if not self.auto_grad:
            raise RuntimeError("Gradients are not being tracked for this tensor. Set requires_grad=True to enable gradient tracking.")

        if grad_output is None:
            # if is scalar tensor, automatically generate a gradient of 1
            if self.data.size != 1:
                raise ValueError("Gradients must be provided for non-scalar outputs.")
            grad_output = Tensor(np.ones_like(self.data), requires_grad=False)

        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad_output.data

        if self.creator is not None:
            # Call the backward function of the creator
            grads = self.creator.backward(Tensor(self.grad, requires_grad=False))
            if not isinstance(grads, tuple):
                grads = (grads,)
            for input_tensor, grad in zip(self.creator.inputs, grads):
                if input_tensor.auto_grad:
                    input_tensor.backward(grad)

    def __add__(self, other: 'Tensor'):
        """
        Addition operator overload.
        """
        if isinstance(other, Tensor):
            from litetorch.core.add_function import AddFunction
            result = AddFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Tensor' and '{}'".format(type(other)))

    def __sub__(self, other: 'Tensor'):
        """
        Subtraction operator overload.
        """
        if isinstance(other, Tensor):
            from litetorch.core.sub_function import SubFunction
            result = SubFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for -: 'Tensor' and '{}'".format(type(other)))

    def __mul__(self, other: 'Tensor'):
        """
        Multiplication operator overload.
        """
        if isinstance(other, Tensor):
            from litetorch.core.mul_function import MulFunction
            result = MulFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Tensor' and '{}'".format(type(other)))

    def __truediv__(self, other: 'Tensor'):
        """
        Division operator overload.
        """
        if isinstance(other, Tensor):
            from litetorch.core.div_function import DivFunction
            result = DivFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for /: 'Tensor' and '{}'".format(type(other)))


    def __matmul__(self, other: 'Tensor'):
        """
        Matrix multiplication operator overload.
        """
        if isinstance(other, Tensor):
            from litetorch.core.matmul_function import MatMulFunction
            result = MatMulFunction()(self, other)
            return result
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Tensor' and '{}'".format(type(other)))

    def __repr__(self):
        """
        String representation of the Tensor object.
        """
        return f"Tensor(data={self.data}, shape={self.shape}, requires_grad={self.auto_grad})"
