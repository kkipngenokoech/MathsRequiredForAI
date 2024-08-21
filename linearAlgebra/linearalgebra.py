#! usr/bin/env python3
print('Hello, Linear Algebra!')
import torch

# SCALARS - single number (they are denoted by ordinary lowercase variables i.e x, y, z)
print(torch.tensor(4.0))
print(torch.tensor(4.0).shape)
x = torch.tensor(4.0)
y = torch.tensor(2.0)
print(x + y)
print(x**y)

# VECTORS - array of numbers - FIXED length array of scalars (they are denoted by lowercase boldface variables i.e x, y, z)

x = torch.arange(4)
print(x)
print(x[3])
print(len(x))
print(x.shape)

# MATRICES - 2D array of numbers (they are denoted by uppercase boldface variables i.e A, B, C)
X = torch.arange(12).reshape(3, 4)
print(x)
print(x.shape)
print(x.numel())
print(X)
X = torch.arange(16).reshape(4, 4)
print(X.T)
print(X == X.T)

# TENSORS - array with more than 2 axes (they are denoted by calligraphic boldface variables i.e X, Y, Z)
X = torch.arange(24, dtype=torch.int).reshape(2, 3, 4)
print(X)

x = torch.arange(48).reshape(2, 3, 4, 2)
print(x)

X = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
print(X)