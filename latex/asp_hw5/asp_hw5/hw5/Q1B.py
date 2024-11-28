import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scipy import linalg
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate


class HiddenMarkovModel:
    def __init__(self, num_states, num_observations):
        """
        Initializes the HMM with random transition, emission, and initial state probabilities.
        
        Args:
            num_states (int): Number of hidden states.
            num_observations (int): Number of unique observations.
        """
        self.num_states = num_states
        self.num_observations = num_observations
        self.A = np.random.dirichlet(np.ones(num_states), num_states)  # Transition matrix
        self.B = np.random.dirichlet(np.ones(num_observations), num_states)  # Emission matrix
        self.pi = np.random.dirichlet(np.ones(num_states))  # Initial state distribution
        
        # Log-space versions to avoid underflow
        self.A_log = np.log(self.A + 1e-10)
        self.B_log = np.log(self.B + 1e-10)
        self.pi_log = np.log(self.pi + 1e-10)

    def forward_algorithm_log(self, O):
        T = len(O)
        N = self.num_states
        alpha_log = np.zeros((T, N))

        # Initialization
        alpha_log[0] = self.pi_log + self.B_log[:, O[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                alpha_log[t, j] = np.logaddexp.reduce(alpha_log[t - 1] + self.A_log[:, j]) + self.B_log[j, O[t]]

        return alpha_log

    def backward_algorithm_log(self, O):
        T = len(O)
        N = self.num_states
        beta_log = np.zeros((T, N))

        # Initialization
        beta_log[-1] = 0  # log(1) = 0

        # Recursion
        for t in range(T - 2, -1, -1):
            for i in range(N):
                beta_log[t, i] = np.logaddexp.reduce(
                    self.A_log[i] + self.B_log[:, O[t + 1]] + beta_log[t + 1]
                )

        return beta_log

    def baum_welch_log(self, O, max_iter=100, epsilon=1e-6):
        T = len(O)
        for iteration in range(max_iter):
            alpha_log = self.forward_algorithm_log(O)
            beta_log = self.backward_algorithm_log(O)

            # Compute gamma and xi in log-space
            gamma_log = alpha_log + beta_log - np.logaddexp.reduce(alpha_log[-1])
            xi_log = np.zeros((T - 1, self.num_states, self.num_states))

            for t in range(T - 1):
                xi_log[t] = self.A_log + self.B_log[:, O[t + 1]] + beta_log[t + 1] + alpha_log[t].reshape(-1, 1)
                xi_log[t] -= np.logaddexp.reduce(xi_log[t].flatten())

            # Update transition, emission, and initial probabilities
            self.A_log = np.logaddexp.reduce(xi_log, axis=0) - np.logaddexp.reduce(gamma_log[:-1], axis=0).reshape(-1, 1)
            self.B_log = np.zeros_like(self.B_log)
            for k in range(self.num_observations):
                mask = (O == k)
                self.B_log[:, k] = np.logaddexp.reduce(gamma_log[mask], axis=0) - np.logaddexp.reduce(gamma_log, axis=0)
            self.pi_log = gamma_log[0]

        return np.exp(self.A_log), np.exp(self.B_log), np.exp(self.pi_log)

    def viterbi_algorithm_log(self, O):
        T = len(O)
        N = self.num_states
        delta_log = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        delta_log[0] = self.pi_log + self.B_log[:, O[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                scores = delta_log[t - 1] + self.A_log[:, j]
                delta_log[t, j] = np.max(scores) + self.B_log[j, O[t]]
                psi[t, j] = np.argmax(scores)

        # Path backtracking
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta_log[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states


if __name__ == "__main__":
    # Load and preprocess the weather dataset
    weather_df = pd.read_csv('seattle-weather.csv', parse_dates=True, index_col='date')
    weather_df = weather_df[["temp_max", "temp_min", "weather"]].dropna()

    # Calculate the average temperature and encode the 'weather' column
    weather_df["temp_avg"] = (weather_df["temp_max"] + weather_df["temp_min"]) / 2
    weather_mapping = {label: idx for idx, label in enumerate(weather_df["weather"].unique())}
    reverse_weather_mapping = {v: k for k, v in weather_mapping.items()}
    weather_df["weather_encoded"] = weather_df["weather"].map(weather_mapping)

    # Convert average temperature to integer values for the observation sequence
    temp_min = weather_df["temp_avg"].min()
    O = (weather_df["temp_avg"] - temp_min).astype(int).values

    # Split data into training and testing sets
    train_size = int(0.8 * len(O))
    O_train, O_test = O[:train_size], O[train_size:]
    actual_train = weather_df["weather_encoded"].values[:train_size]
    actual_test = weather_df["weather_encoded"].values[train_size:]

    # Initialize and train the HMM
    num_states = len(weather_mapping)
    num_observations = O.max() + 1
    hmm = HiddenMarkovModel(num_states, num_observations)
    A_trained, B_trained, pi_trained = hmm.baum_welch_log(O_train)

    # Decode the most likely state sequence
    train_predicted_states = hmm.viterbi_algorithm_log(O_train)
    test_predicted_states = hmm.viterbi_algorithm_log(O_test)

    # Decode predicted states to weather labels
    train_decoded_states = [reverse_weather_mapping[s] for s in train_predicted_states]
    test_decoded_states = [reverse_weather_mapping[s] for s in test_predicted_states]

    # Evaluate accuracy
    train_accuracy = np.mean(train_predicted_states == actual_train)
    test_accuracy = np.mean(test_predicted_states == actual_test)

    print("\nHMM Performance:")
    print(f"Training Set Accuracy: {train_accuracy:.2%}")
    print(f"Testing Set Accuracy: {test_accuracy:.2%}")
