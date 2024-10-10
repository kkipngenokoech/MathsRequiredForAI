import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate the eigenvalues of a given matrix.
    
    Args:
    matrix (numpy.ndarray): A square matrix for which to calculate eigenvalues.
    
    Returns:
    numpy.ndarray: An array of eigenvalues.
    """
    # Check if the input is a square matrix
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be a square matrix")
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    
    return eigenvalues

# Example usage
if __name__ == "__main__":
    # Create a sample 3x3 matrix
    sample_matrix = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    
    print("Sample matrix:")
    print(sample_matrix)
    
    # Calculate eigenvalues
    eigenvalues = calculate_eigenvalues(sample_matrix)
    
    print("\nEigenvalues:")
    print(eigenvalues)
