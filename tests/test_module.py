"""
tests/test_module.py
Unit tests for the Module class in the nn module.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-04-24
"""

from litetorch.nn.module import Module
from litetorch.core.tensor import Tensor


class DummyModule(Module):
    """
    A dummy module for testing purposes.
    Do nothing in forward and backward methods.
    """

    def __init__(self):
        super().__init__()
        self.w = Tensor([[1.0, 2.0]], requires_grad=True)
        self.b = Tensor([[0.0, 0.0]], requires_grad=True)
        self._parameters = {"w": self.w, "b": self.b}
        self._modules = {}
        self._name = "DummyModule"

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output

    def __repr__(self):
        return f"{self.__class__.__name__}()"


def test_module_init():
    """
    Test Module initialization.
    """
    module = Module()
    assert isinstance(
        module, Module
    ), "Module should be an instance of the Module class."
    assert (
        module._parameters == {}
    ), "Module parameters should be initialized as an empty dictionary."
    assert (
        module._modules == {}
    ), "Module submodules should be initialized as an empty dictionary."
    assert module._name == "Module", "Module name should be 'Module' by default."


def test_parameters():
    module = DummyModule()
    assert isinstance(module.parameters(), list), "Parameters should be a list."
    assert len(module.parameters()) == 2, "There should be two parameters."
    assert isinstance(
        module.parameters()[0], Tensor
    ), "First parameter should be a Tensor."
    assert isinstance(
        module.parameters()[1], Tensor
    ), "Second parameter should be a Tensor."


def test_add_module():
    module = DummyModule()
    submodule = DummyModule()
    module.add_module("submodule", submodule)
    assert (
        "submodule" in module._modules
    ), "Submodule should be added to the module's submodules."
    assert isinstance(
        module._modules["submodule"], DummyModule
    ), "Submodule should be an instance of DummyModule."
    assert module._modules["submodule"]._name == "submodule", "Submodule name should be 'submodule'."
    assert module._modules["submodule"]._modules == {}, "Submodule submodules should be initialized as an empty dictionary."

def test_zero_grad():
    module = DummyModule()
    # set the gradients to 1.0 for testing
    for param in module.parameters():
        param.grad[:] = 1.0
    module.zero_grad()
    for param in module.parameters():
        assert param.grad.sum() == 0, "Gradients should be zero after zero_grad call."

def test_add_module_and_forward():
    module = DummyModule()
    submodule = DummyModule()
    module.add_module("submodule", submodule)
    x = Tensor([[1.0, 2.0]])
    output = module(x)
    assert isinstance(output, Tensor), "Output should be a Tensor."
    assert output.shape == x.shape, "Output shape should match input shape."
    assert (
        output.data == x.data
    ).all(), "Output data should match input data in this dummy implementation."
