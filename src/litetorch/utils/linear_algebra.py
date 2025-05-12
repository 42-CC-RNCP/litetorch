import numpy as np
from typing import Tuple


def reduce_broadcast_shape(grad: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduce summed gradient along broadcasted dimensions.
    For example:
        grad: shape (4, 2)
        target_shape: (1, 2)
        → return: (1, 2), via summing over axis 0
    """
    if grad.shape == target_shape:
        return grad

    # Check if the shapes are compatible for broadcasting
    if len(grad.shape) < len(target_shape):
        raise ValueError(f"Grad shape {grad.shape} is not compatible with target shape {target_shape}")

    # Determine axes to sum over
    axes_to_sum = []
    for i in range(len(grad.shape)):
        if i >= len(target_shape) or target_shape[i] == 1 and grad.shape[i] != 1:
            axes_to_sum.append(i)

    # Reduce
    reduced_grad = np.sum(grad, axis=tuple(axes_to_sum), keepdims=True)

    # Reshape to exactly match target_shape
    return reduced_grad.reshape(target_shape)
