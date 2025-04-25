"""
litetorch/nn/module.py
This module defines the Module class, which serves as a base class for all neural network modules.
It provides methods for parameter management, forward and backward propagation, and module registration.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

from typing import List, Dict, Any
from litetorch.core.tensor import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
        self._name: str = self.__class__.__name__

    def add_module(self, name: str, module: 'Module') -> None:
        """
        Adds a submodule to the current module.

        Parameters:
        - name: Name of the submodule.
        - module: Instance of the Module to be added.
        """
        if not isinstance(module, Module):
            raise TypeError(f"Expected a Module instance, got {type(module)}")
        self._modules[name] = module
        module._name = name

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output: Tensor) -> Tensor:
        raise NotImplementedError("Backward method not implemented.")

    def parameters(self) -> List[Tensor]:
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def zero_grad(self) -> None:
        """
        Sets the gradients of all parameters to zero.
        """
        for param in self._parameters.values():
            if param.auto_grad:
                param.grad[:] = 0.0

    def __call__(self, x) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
