import numpy as np

# Define a 4x4 matrix
matrix = np.array([[5, 1.2, 0.8, 0.6],
                   [1.2, 4, 0.5, 0.3],
                   [0.8, 0.5, 3, 0.2],
                   [0.6, 0.3, 0.2, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)