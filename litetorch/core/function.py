"""
litetorch/core/function.py
This module defines the Function class for atomic operation nodes in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-26
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple


class Function(ABC):
    """
    Base class for all functions in the LiteTorch framework.
    This class provides a template for implementing custom operations.
    """

    @abstractmethod
    def forward(self, *inputs: Tuple[Tensor, ...]) -> Tensor:
        """
        Forward pass of the function.

        Parameters:
        - inputs: Tuple of input tensors.

        Returns:
        - Output tensor after applying the function.
        """
        pass

    @abstractmethod
    def backward(self, *grad_outputs: Tuple[Tensor, ...]) -> Tuple[Tensor, ...]:
        """
        Backward pass of the function.

        Parameters:
        - grad_outputs: Gradients of the output tensor. For Transformer, it will have more than one gradient output.

        Returns:
        - Gradients of the input tensors.
        """
        pass

    def __call__(self, *inputs: Tuple[Tensor]) -> Tensor:
        from litetorch.core.tensor import Tensor

        self.inputs = inputs
        outputs = self.forward(*inputs)

        if isinstance(outputs, Tensor):
            outputs.creator = self
            outputs.creation_args = list(inputs)
        elif isinstance(outputs, (tuple, list)):
            for output in outputs:
                if isinstance(output, Tensor):
                    output.creator = self
                    output.creation_args = list(inputs)
        else:
            raise TypeError("Output must be a Tensor or a tuple/list of Tensors.")
        return outputs

    def __repr__(self):
        """
        String representation of the Function object.
        """
        return f"{self.__class__.__name__}()"
