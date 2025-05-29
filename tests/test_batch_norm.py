"""
tests/test_batch_norm.py
Unit tests for the Batch Normalization function in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-27
"""

import numpy as np
import torch
import torch.nn as nn
from litetorch.core.tensor import Tensor
from litetorch.core.batch_norm_function import BatchNorm1DFunction



def test_batchnorm_forward_backward():
    np.random.seed(42)
    input_np = np.random.randn(4, 3).astype(np.float32)
    weight_np = np.ones(3, dtype=np.float32)
    bias_np = np.zeros(3, dtype=np.float32)

    # --- LiteTorch ---
    x = Tensor(input_np, requires_grad=True)
    weight = Tensor(weight_np, requires_grad=True)
    bias = Tensor(bias_np, requires_grad=True)

    bn = BatchNorm1DFunction()
    out = bn.forward(x, weight, bias)

    grad_out = Tensor(np.ones_like(out.data))
    dx, dweight, dbias = bn.backward(grad_out)

    # --- PyTorch Reference ---
    x_pt = torch.tensor(input_np, requires_grad=True)
    weight_pt = torch.tensor(weight_np, requires_grad=True)
    bias_pt = torch.tensor(bias_np, requires_grad=True)
    bn_pt = nn.BatchNorm1d(3, eps=1e-5, affine=True)

    with torch.no_grad():
        bn_pt.weight.copy_(weight_pt)
        bn_pt.bias.copy_(bias_pt)

    out_pt = bn_pt(x_pt)
    out_pt.sum().backward()

    # --- Assertions ---
    assert np.allclose(dx.data, x_pt.grad.numpy(), atol=1e-5), "dx mismatch"
    assert np.allclose(dweight.data, bn_pt.weight.grad.detach().numpy(), atol=1e-5), "dweight mismatch"
    assert np.allclose(dbias.data, bn_pt.bias.grad.detach().numpy(), atol=1e-5), "dbias mismatch"
    
    
def test_batchnorm_eval_mode():
    np.random.seed(123)
    input_np = np.random.randn(6, 3).astype(np.float32)

    running_mean = np.array([0.5, -0.2, 1.0], dtype=np.float32)
    running_var = np.array([0.25, 0.5, 2.0], dtype=np.float32)
    eps = 1e-5

    weight_np = np.array([1.0, 0.5, 2.0], dtype=np.float32)
    bias_np = np.array([0.0, 1.0, -1.0], dtype=np.float32)

    x = Tensor(input_np)
    norm = (input_np - running_mean) / np.sqrt(running_var + eps)
    expected_output_np = weight_np * norm + bias_np
    expected_output = Tensor(expected_output_np)

    x_pt = torch.tensor(input_np)
    bn_pt = nn.BatchNorm1d(3, eps=eps, affine=True)
    bn_pt.eval()  # 設為 inference 模式

    with torch.no_grad():
        bn_pt.running_mean.copy_(torch.tensor(running_mean))
        bn_pt.running_var.copy_(torch.tensor(running_var))
        bn_pt.weight.copy_(torch.tensor(weight_np))
        bn_pt.bias.copy_(torch.tensor(bias_np))

    out_pt = bn_pt(x_pt).detach().numpy()

    # --- Assertions ---
    assert np.allclose(out_pt, expected_output_np, atol=1e-5), "Eval output mismatch with manual computation"
