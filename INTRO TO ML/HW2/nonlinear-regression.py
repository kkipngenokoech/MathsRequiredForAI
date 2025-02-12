import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv, pinv
from sklearn.model_selection import train_test_split

class SinusoidalRegressor:
    def __init__(self):
        self.k = None
        self.weights = None

    def phi(self, x):
        # The basis function for a general 2k
        
        # TODO: YOUR CODE GOES HERE

    def fit(self, X_train, Y_train, k):
        self.k = k
        # Construct the design matrix Phi for all data points in X_train
        # TODO: YOUR CODE GOES HERE
        # Solve for the weights using the normal equation with a pseudo-inverse
        # Make sure the shapes align: Phi.T @ Phi should be a square matrix and Phi.T @ Y_train should be a vector
        # TODO: YOUR CODE GOES HERE

    def predict(self, X):
        # Check if the model is fitted
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        # Apply the learned model

        return # TODO: YOUR CODE GOES HERE

    def rmse(self, X_val, Y_val):
        # Predict the values for X_val
        # TODO: YOUR CODE GOES HERE
        # Calculate the RMSE
        return # TODO: YOUR CODE GOES HERE

np.random.seed(61)
csv_file = 'nonlinear-regression-data.csv'
data = pd.read_csv(csv_file)
x = np.array(data['X'])
y = np.array(data['Noisy_y'])


### Evaluation Part 0 #################################################################################

# Split the data

# TODO: YOUR CODE GOES HERE


### Evaluation Part 1 and 2 #################################################################################

# Initialize the model
# TODO: YOUR CODE GOES HERE

# Vary k from 1 to 10 and obtain RMSE error on the training set and validation set

# TODO: YOUR CODE GOES HERE

# Plotting the training error versus k

# TODO: YOUR CODE GOES HERE

# Plotting the validation error versus k

# TODO: YOUR CODE GOES HERE

### Evaluation Part 4 #################################################################################
# You will reate separate plots for each k you can use plt.subplots function

# TODO: YOUR CODE GOES HERE