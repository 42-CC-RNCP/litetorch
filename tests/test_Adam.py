"""
tests/test_Adam.py
Unit tests for the Adam optimizer in the litetorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

import torch
import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.optim.Adam import Adam
from litetorch.nn.loss import MSELoss


def test_adam_step_matches_pytorch():
    np.random.seed(42)
    torch.manual_seed(42)

    data_np = np.random.randn(10).astype(np.float32)
    grad_np = np.random.randn(10).astype(np.float32)

    # --- LiteTorch ---
    param = Tensor(data_np.copy(), requires_grad=True)
    param.grad = grad_np.copy()
    opt = Adam(parameters=[param], lr=0.001)

    opt.step()

    # --- PyTorch Adam ---
    param_torch = torch.tensor(data_np, requires_grad=True)
    param_torch.grad = torch.tensor(grad_np)
    opt_pt = torch.optim.Adam([param_torch], lr=0.001, betas=(0.9, 0.999), eps=1e-8)
    opt_pt.step()

    assert np.allclose(param.data, param_torch.detach().numpy(), atol=1e-6)
    
    
def test_adam_convergence():
    x_np = np.random.randn(20, 5).astype(np.float32)
    y_np = np.random.randn(20, 1).astype(np.float32)

    x = Tensor(x_np)
    y = Tensor(y_np)

    w = Tensor(np.random.randn(5, 1).astype(np.float32), requires_grad=True)
    b = Tensor(np.zeros(1, dtype=np.float32), requires_grad=True)
    opt = Adam([w, b], lr=0.01)

    loss_fn = MSELoss()
    for _ in range(500):
        y_pred = x @ w + b
        loss = loss_fn(y_pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    
    # Check if loss is decreasing
    y_pred = x @ w + b
    final_loss = loss_fn(y_pred, y)
    assert final_loss.data < 1.0, "Adam did not converge to a reasonable loss value"