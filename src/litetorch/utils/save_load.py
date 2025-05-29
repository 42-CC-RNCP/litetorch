"""
litetorch/utils/save_load.py
Provides functionality to save and load model parameters and configurations.

format:

```json
{
  "architecture": [
    {"type": "Linear", "in_features": 30, "out_features": 64, "bias": true},
    {"type": "ReLU"},
    ...
  ],
  "parameters": {
    "layer_0": { "weight": [...], "bias": [...] },
    ...
  }
}
```

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-02
"""

import json
import numpy as np
from typing import List, Dict
from litetorch.utils.registry import LAYER_REGISTRY


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    else:
        return obj

class SaveLoadMixin:
    def save(self, filepath: str, with_params = True) -> None:
        arch = []
        params = {}

        for i, layer in enumerate(self.layers):
            arch.append(layer.get_config())
            raw_params = layer.get_parameters()
            serializable_params = to_serializable(raw_params)
            params[f"layer_{i}"] = serializable_params

        model_dict = {
            "architecture": arch,
            "parameters": {}
        }
        if with_params:
            model_dict["parameters"] = params
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=4)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> None:
        from litetorch.nn.sequential import Sequential
        # Make sure to import all layer classes to register them
        from litetorch.nn.linear import Linear
        from litetorch.nn.activation import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
        from litetorch.nn.dropout import Dropout
        from litetorch.nn.batchnorm import BatchNorm1d

        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        arch : List[Dict] = model_dict["architecture"]
        params : Dict = model_dict["parameters"]
        layers = []

        for layer_config in arch:
            layer_type = layer_config["type"]

            if layer_type not in LAYER_REGISTRY:
                raise ValueError(f"Layer type '{layer_type}' not registered. Available types: {list(LAYER_REGISTRY.keys())}")
            layer_class = LAYER_REGISTRY[layer_type]
            layer : Module = layer_class(**{k: v for k, v in layer_config.items() if k != "type"})
            layers.append(layer)

        model = Sequential(*layers)
        for i, layer in enumerate(model.layers):
            layer : Module
            if f"layer_{i}" in params:
                layer.set_parameters(params[f"layer_{i}"])

        print(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    from litetorch.nn.sequential import Sequential
    from litetorch.nn.linear import Linear
    from litetorch.nn.activation import *
    # Example usage
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1)
    )
    model.save("model.json")
    loaded_model = Sequential.load("model.json")
    print(loaded_model)
