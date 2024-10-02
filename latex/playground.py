import numpy as np

# Define the augmented matrix [A|b]
A_b = np.array([
    [2, 1, 3, 4],
    [1, 1, 2, 6],
    [2, 3, 4, 2],
    [1, 4, 6, 1],
    [4, 1, 5, 1],
    [6, 1, 1, 4]
])

# Compute the rank of the matrix
rank = np.linalg.matrix_rank(A_b)

print("The rank of the augmented matrix [A|b] is:", rank)

# to find inverse of a matrix
A = np.array([[0, 1, -2], [1, 0, 2], [1, 1, -1]])  
print(A)

# inverse of A
A_inv = np.linalg.inv(A)    
print(A_inv)