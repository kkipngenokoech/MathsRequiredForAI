import numpy as np

# Coefficients of the polynomial λ^3 - 9λ^2 + 18λ - 13 = 0
coefficients = [1, -9, 18, -13]

# Find the roots of the polynomial
roots = np.roots(coefficients)

# Print the roots
print("The roots of the polynomial are:", roots)