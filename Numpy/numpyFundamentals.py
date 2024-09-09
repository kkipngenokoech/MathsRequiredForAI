print("numpy fundamentals")
# creating arrays
import numpy as np
import torch
npArray = np.ones((3, 4), dtype=np.float32)
print(npArray, npArray.shape, npArray.size, npArray.dtype, type(npArray))
npArray2 = np.zeros((3, 4), dtype=np.float32)
print(npArray2, npArray2.shape, npArray2.size, npArray2.dtype, type(npArray2))
npArray3 = np.random.rand(3, 4)
print(npArray3, npArray3.shape, npArray3.size, npArray3.dtype, type(npArray3))
npArray4 = np.random.randn(3, 4)
print(npArray4, npArray4.shape, npArray4.size, npArray4.dtype, type(npArray4))

# converting ndarray to other types
npToTensor = torch.tensor(npArray)
print(npToTensor, npToTensor.shape, npToTensor.size(), npToTensor.dtype, type(npToTensor))

npToTensor = torch.from_numpy(npArray)
print(npToTensor, npToTensor.shape, npToTensor.size(), npToTensor.dtype, type(npToTensor))

# from tensor to ndarray
tensorToNp = npToTensor.numpy()
print(tensorToNp, tensorToNp.shape, tensorToNp.size, tensorToNp.dtype, type(tensorToNp))

##  SEEDING
np.random.seed(0)
X = np.random.randint(-1000, 1000, size=3000)
print(X)
print(np.mean(X), np.std(X), np.var(X))
print(torch.tensor(X))

Y = np.random.normal(0, 1, size=30)
print(Y)
print(torch.tensor(Y))

Z = np.random.rand(3, 4)
print(Z)

Q = np.ones((3, 4))
print(Q)
print(torch.tensor(Q))


A = np.array([[0,1, 1], [1,0,-1],[-2,1,0],[-1,1,1]])
print(A)
print(A.shape)
print(A.T)
print(A.T.shape)
print(A.T @ A)
print((A.T @ A).T ==  A.T @ A)

A = np.array([[0,1, 1], [1,0,-1],[-2,1,0],[-1,1,1]])
print(A.T @ A)