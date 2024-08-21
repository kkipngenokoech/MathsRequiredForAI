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
