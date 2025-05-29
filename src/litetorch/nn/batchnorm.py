import numpy as np
from litetorch.core.tensor import Tensor
from litetorch.nn.module import Module
from litetorch.core.batch_norm_function import BatchNorm1DFunction
from litetorch.utils.registry import register_layer


@register_layer("BatchNorm1d")
class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Tensor(np.ones(num_features), requires_grad=True)
        self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        
        self._parameters["weight"] = self.weight
        self._parameters["bias"] = self.bias
        self._name = "BatchNorm1d"
        
        # TODO: Consider adding running mean and variance for inference mode
        
    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass for batch normalization.
        
        Args:
            input (Tensor): Input tensor to be normalized.
        
        Returns:
            Tensor: Normalized output tensor.
        """
        return BatchNorm1DFunction(self.eps)(input, self.weight, self.bias)
        
    def get_config(self) -> dict:
        return {
            "type": "BatchNorm1d",
            "num_features": self.num_features,
            "eps": self.eps
        }
        
    def set_parameters(self, params: dict) -> None:
        if "weight" in params:
            self.weight.data = params["weight"]
        if "bias" in params:
            self.bias.data = params["bias"]
    
    def get_parameters(self) -> dict:
        return {
            "weight": self.weight.data,
            "bias": self.bias.data
        }
    
    def __repr__(self) -> str:
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps})"


# Example usage:
if __name__ == "__main__":
    bn = BatchNorm1d(num_features=3)
    input_tensor = Tensor(np.random.randn(5, 3), requires_grad=True)
    output_tensor = bn.forward(input_tensor)
    print("Input Tensor:", input_tensor.data)
    print("Output Tensor:", output_tensor.data)
    print("Parameters:", bn.get_parameters())
    print("Config:", bn.get_config())
    print(bn)
