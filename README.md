# litetorch

## How to Implement A NN Framework From Scratch

I implemented a simple neural network framework from scratch using Python.
1. Implemented a `Tensor` class to represent multi-dimensional arrays.
2. Implemented a `Module` class to be the base class for all neural network modules.
3. Implemented a `Linear` class inherit from `Module` to represent a fully connected layer.
5. Implemented an `Optimizer` class inherit from `Module` to represent the optimizer.
4. Implemented a `Loss` class inherit from `Module` to represent the loss function.

For each class, I followed TDD (Test Driven Development) to implement the class. I wrote the test cases first and then implemented the class to pass the test cases.


### Project Structure

#### Core module

The whole calculation is base on the `Tensor` class, which is a multi-dimensional array. The `Tensor` class has the following attributes:
- `data`: the actual data of the tensor
- `grad`: the gradient of the tensor
- `auto_grad`: a boolean value indicating whether to calculate the gradient or not
- `shape`: the shape of the tensor
- `_backward`: a function that calculates the gradient of the tensor
- `_op`: the operation that generated the tensor
- `_prev`: the previous tensors that generated the tensor

The `Tensor` class has the following methods:
- `__mulmat__`: the matrix multiplication of two tensors
- `__add__`: the addition of two tensors
- `__sub__`: the subtraction of two tensors

After any operation,
1. The `Tensor` class will create a new tensor with the result of the operation.
2. Record the self and the other tensor in the `_prev` attribute of the new tensor.
3. The `_op` attribute of the new tensor will be set to the operation that generated the tensor.

#### Optimizer module
The `Optimizer` class is used to update the parameters of the model. It has the following methods:
- `step`: update the parameters of the model
- `zero_grad`: set the gradient of the parameters to zero

#### Loss module
The `Loss` class is used to calculate the loss of the model. It has the following methods:
- `forward`: calculate the loss of the model
- `backward`: calculate the gradient of the loss with respect to the parameters of the model

#### Model module
The `Model` class is used to define the model. It has the following methods:
- `forward`: define the forward pass of the model
- `backward`: define the backward pass of the model
- `parameters`: return the parameters of the model
- `zero_grad`: set the gradient of the parameters to zero

