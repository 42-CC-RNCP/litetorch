"""
tests/test_save_load.py
Unit tests for the save and load functions for models.

Author: Lea Yeh
Version: 0.0.1
Date: 2025-05-02
"""

import os
import tempfile
from litetorch.nn.sequential import Sequential
from litetorch.core.tensor import Tensor
from litetorch.nn.linear import Linear
from litetorch.nn.activation import *


def test_sequential_save_load():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1)
    )

    # Create a temporary directory for saving the model
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "model.json")
        # Save the model
        model.save(output_path)

        # Load the model
        loaded_model : Module = Sequential.load(output_path)

        # Check if the loaded model is the same as the original model
        assert isinstance(loaded_model, Sequential), "Loaded model should be an instance of Sequential."
        assert len(loaded_model.layers) == len(model.layers), "Loaded model should have the same number of layers."
        for original_layer, loaded_layer in zip(model.layers, loaded_model.layers):
            original_layer : Module
            loaded_layer : Module
            assert isinstance(loaded_layer, type(original_layer)), "Layer types should match."
            assert original_layer.get_config() == loaded_layer.get_config(), "Layer configurations should match."
            assert original_layer.get_parameters() == loaded_layer.get_parameters(), "Layer parameters should match."
