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
print(x.sum(axis=0))

X = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)
print(X)

y = 2
print(X.sum())
print(X + y)

# ELEMENTWISE OPERATIONS
X = torch.tensor([1.0, 2, 4, 8])
Y = torch.tensor([2, 2, 2, 2])
print(X + Y)
print(X - Y)
print(X * Y)
print(X / Y)
print(X ** Y)
print(torch.exp(X))

# DOT PRODUCTS
print(torch.dot(X, Y.to(torch.float32)))

# MATRIX-VECTOR PRODUCTS
print("Matrix-Vector Products")
A = torch.arange(6).reshape(2, 3)
x = torch.arange(3)
print(A)
print(x)
print(A@x)
print(torch.mv(A, x))

# MATRIX-MATRIX MULTIPLICATION
print("Matrix-Matrix Multiplication")
B = torch.arange(6).reshape(3, 2)
print(torch.mm(A, B))
print(A@B)

# NORMS
print("Norms")
