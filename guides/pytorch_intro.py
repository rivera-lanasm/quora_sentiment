import torch

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# The Torch Tensor and NumPy array will share their underlying memory 
# locations (if the Torch Tensor is on CPU), and changing one will change 
# the other.

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# docs
# https://pytorch.org/docs/stable/torch.html


# Central to all neural networks in PyTorch is the autograd package. 
# It is a define-by-run framework, which means that your backprop is defined by 
# how your code is run, and that every single iteration can be different.

# torch.Tensor is the central class of the package. If you set its attribute .requires_grad 
# as True, it starts to track all operations on it. When you finish your computation you can 
# call .backward() and have all the gradients computed automatically. The gradient for this tensor 
# will be accumulated into .grad attribute.

# Tensor and Function are interconnected and build up 
# an acyclic graph, that encodes a complete history of computation

# https://pytorch.org/docs/stable/autograd.html#function

# Generally speaking, torch.autograd is an engine for computing vector-Jacobian product

#? Neural networks can be constructed using the torch.nn package.


