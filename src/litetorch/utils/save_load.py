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


class SaveLoadMixin:
    def save(self, filepath: str) -> None:
        arch = []
        params = {}

        for i, layer in enumerate(self.layers):
            arch.append(layer.get_config())
            params[f"layer_{i}"] = layer.get_parameters()

        model_dict = {
            "architecture": arch,
            "parameters": params
        }
        with open(filepath, 'w') as f:
            json.dump(model_dict, f, indent=4)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> None:
        from litetorch.nn.activation import ReLU, Sigmoid, Tanh, Softmax, LeakyReLU
        from litetorch.nn.linear import Linear
        from litetorch.nn.sequential import Sequential

        with open(filepath, 'r') as f:
            model_dict = json.load(f)
        arch : List[Dict] = model_dict["architecture"]
        params : Dict = model_dict["parameters"]
        layers = []

        for layer_config in arch:
            layer_type = layer_config["type"]

            if layer_type == "Linear":
                layer = Linear(**{k: v for k, v in layer_config.items() if k != "type"})
            elif layer_type == "ReLU":
                layer = ReLU()
            elif layer_type == "Sigmoid":
                layer = Sigmoid()
            elif layer_type == "Tanh":
                layer = Tanh()
            elif layer_type == "Softmax":
                layer = Softmax()
            elif layer_type == "LeakyReLU":
                layer = LeakyReLU(**{k: v for k, v in layer_config.items() if k != "type"})
            else:
                raise ValueError(f"Unknown layer type: {layer_type}")
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
