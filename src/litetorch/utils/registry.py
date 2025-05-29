"""
litetorch/utils/registry.py
# This module provides a registry for managing custom functions and modules in the LiteTorch framework.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-29
"""

LAYER_REGISTRY = {}

def register_layer(name):
    def decorator(cls):
        LAYER_REGISTRY[name] = cls
        return cls
    return decorator
