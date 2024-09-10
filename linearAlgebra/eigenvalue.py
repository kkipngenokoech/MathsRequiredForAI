import numpy as np

# Coefficients of the cubic equation: λ^3 - 9λ^2 + 22λ - 13 = 0
coefficients = [1, -9, 22, -13]

# Get the roots
roots = np.roots(coefficients)
print(roots)



# Define the matrix B
B = np.array([[4, 2, 1],
              [1, 3, 2],
              [0, 1, 2]])

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

# Output the results
print(eigenvalues)
