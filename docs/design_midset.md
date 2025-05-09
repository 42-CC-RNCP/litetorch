# Design Mindset

This document notes the mindset I had while designing the neural network framework. It is not a design document, but rather a collection of thoughts and ideas that I had while designing the framework. It is not meant to be a complete or final design document, but rather a starting point for further discussion and refinement.

## The Principle Design of the Pytorch Framework

I noticed that the Pytorch framework is not covered all of the utilities for NN training. It is very explicit focus on low level nn framework and the user can build their own utilities on top of it.

1. Flexibility over Automation
    - PyTorch prefers to offer building blocks rather than automated pipelines. This gives you the flexibility to:
        - Use any data format
        - Apply custom logic (e.g., domain-specific label encoding, tokenization, transformations)
        - Compose your own pipeline from scratch
    - Contrast that with frameworks like scikit-learn or Keras, which provide higher-level abstractions and "magic" behaviors — useful for rapid prototyping but sometimes too rigid for research or non-standard use cases.
2. Transparency and Explicitness
3. Modularity by Design
    - PyTorch expects you to:
        - Use external tools like pandas, scikit-learn, torchtext, or huggingface/transformers for preprocessing
        - Integrate them with PyTorch’s Dataset and DataLoader API
    -This modularity makes it easy to swap out components, reuse your framework across projects, and support diverse workflows.


## PyTorch Autograd Design Mindset

1. Define-by-Run (Dynamic Computation Graph)
    - PyTorch uses a dynamic computation graph — also called define-by-run. That means the graph is built on-the-fly as you perform tensor operations in Python.
    - Why?
        - It mirrors native Python control flow (e.g., loops, if statements).
        - No need to "compile" or "freeze" the graph before using it.
        - You can debug easily with Python tools like print() or pdb.

```python
# Example of dynamic computation graph
x = torch.tensor(1.0, requires_grad=True)
y = x ** 2 + 2 * x
y.backward()
print(x.grad)  # Computed on-demand
```

2. Composability and Modularity
    - Each operation (like add, matmul, log, etc.) is a tiny modular function that knows how to:
        - Compute the forward pass
        - Compute its gradient (backward pass)
    - These operations are chained together via Function nodes into a graph structure.
    - This design means you can:
        - Build any custom function or operation
        - Insert your own autograd.Function with custom forward/backward logic

    