import numpy as np
from cvxopt import matrix, solvers

# Labels for the data points
y = np.array([1, 1, 1, -1, -1, -1], dtype=float)

# Kernel matrix (dot products of points)
K = [[ 0,  0,  0,  0,  0,  0],
 [ 0,  2,  6,  2,  0, 12],
 [ 0,  6, 20, 12, 2, 42],
 [ 0, 2,   12,20 ,6 ,30],
 [0 ,0 ,2 ,6 ,2 ,6],
 [0 ,12 ,42 ,30 ,6 ,90]]

# Construct the matrix form for cvxopt
# The objective function: maximize sum(alpha) - 1/2 * alpha^T * (y_i * y_j * K)
# To solve using cvxopt, we define it as a quadratic programming problem:
P = np.outer(y, y) * K  # G = y_i * y_j * K
q = -np.ones(6)

# Equality constraint: sum(alpha_i * y_i) = 0
A_eq = y.reshape(1, -1)
b_eq = np.array([0.0])

# Inequality constraints: alpha_i >= 0 (no explicit matrix needed, handled by cvxopt default)

# Convert everything to cvxopt matrix format
P = matrix(P)
q = matrix(q)
A_eq = matrix(A_eq)
b_eq = matrix(b_eq)
G_ineq = matrix(-np.eye(6))  # alpha_i >= 0
h_ineq = matrix(np.zeros(6))

# Solve the quadratic programming problem
sol = solvers.qp(P, q, G_ineq, h_ineq, A_eq, b_eq)

# Output the result (the alphas)
alphas = np.ravel(sol['x'])
print("Optimal alphas:", alphas)
