import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from scipy import linalg
import yfinance as yf
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

class StockDataAnalysis:
    def __init__(self, ticker, start_date, end_date, n_states=5):
        """
        Initialize the StockDataAnalysis with stock parameters and state count.
        
        Args:
            ticker (str): Stock ticker symbol.
            start_date (datetime): Start date for data retrieval.
            end_date (datetime): End date for data retrieval.
            n_states (int): Number of quantile-based states.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.n_states = n_states
        self.df = None
        self.transition_matrix = None

    def get_stock_data(self):
        """
        Download historical stock data from Yahoo Finance and preprocess it.
        """
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=self.start_date, end=self.end_date)
        df['Daily_Return'] = NotImplemented #TODO Extract the close price and calculate daily returns
        self.df = NotImplemented  #TODO Remove NaN values
        print(f"Stock data for {self.ticker} loaded successfully.")

    def assign_states(self):
        """
        Assign states based on return quantiles and add descriptive labels.
        """
        self.df['State'] = pd.qcut(self.df['Daily_Return'], q=self.n_states, labels=False)
        state_labels = NotImplemented #TODO Initialize state lables in a list in the order described in the question
        self.df['State_Description'] = pd.Categorical(
            [state_labels[int(i)] for i in self.df['State']]
        )
        print("States assigned based on quantiles.")

    def get_transition_matrix(self):
        """
        Construct the transition probability matrix from the sequence of states.
        
        Returns:
            np.array: Transition matrix (dimensions: n_states x n_states).
        
        TODO:
        - Ensure that states are properly handled as integers without NaNs.
        - Implement a loop or vectorized operation to count transitions between states.
        - Normalize the rows of the transition matrix to get probabilities.
        """
        # TODO: Extract the 'State' column from the DataFrame as a numpy array.
        states = NotImplemented # Ensure this is done without NaNs if needed.

        # Initialize a matrix to store transition counts.
        transitions = NotImplemented 

        # TODO: Iterate through the array of states and populate the transition matrix.
        # Hint: Use a loop to increment the count for transitions from states[i] to states[i + 1].
        for i in range(len(states) - 1):
            current_state = NotImplemented  # Ensure valid integer index
            next_state = NotImplemented  # Ensure valid integer index
            transitions[current_state][next_state] += 1  # Increment the count for observed transitions.

        
        epsilon = 1e-8
        row_sums = transitions.sum(axis=1) + epsilon

        # TODO: Normalize each row of the transition matrix to create probabilities.
        # Hint: Divide each element in a row by the sum of the row to ensure each row sums to 1.
        self.transition_matrix = NotImplemented

        return self.transition_matrix


    def get_stationary_distribution(self):
        """
        Calculate the stationary distribution of the Markov chain.
        
        Returns:
            np.array: Stationary distribution (dimensions: n_states).
        """
        eigenvals, eigenvects = NotImplemented # TODO Calculate the eigenvalues and eigenvectors
        stationary = eigenvects[:, np.where(np.isclose(eigenvals, 1))[0][0]].real
        stationary = NotImplemented #TODO Normalize the stationary distribution
        print("Stationary distribution calculated.")
        return stationary

    def analyze_markov_chain(self):
        """
        Analyze and visualize Markov chain properties.
        """
        stationary_dist = NotImplemented # TODO Calculate the stationary distribution
        state_labels = NotImplemented #TODO Initialize state lables in a list in the order described in the question

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        sns.heatmap(self.transition_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax1,
                    xticklabels=state_labels, yticklabels=state_labels)
        ax1.set_title('Transition Probability Matrix')
        ax1.set_xlabel('Next State')
        ax1.set_ylabel('Current State')

        bars = ax2.bar(state_labels, stationary_dist)
        ax2.set_title('Stationary Distribution')
        ax2.set_xlabel('State')
        ax2.set_ylabel('Probability')
        ax2.tick_params(axis='x', rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height * 100:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        print("\nMarkov Chain Analysis Complete.")
        print("\nTransition Matrix:")
        transition_df = pd.DataFrame(self.transition_matrix,
                                     columns=state_labels,
                                     index=state_labels)
        print(transition_df)

        print("\nStationary Distribution:")
        stationary_df = pd.DataFrame({
            'State': state_labels,
            'Probability': stationary_dist
        })
        print(stationary_df)

    def simulate_price_path(self, initial_price, days):
        """
        Simulate a single price path using the Markov chain model.
        
        Args:
            initial_price (float): Starting price for the simulation.
            days (int): Number of days to simulate.
        
        Returns:
            np.array: Simulated price path.
        
        TODO:
        - Implement the logic for price path simulation based on state transitions.
        """
        prices = [initial_price]
        current_state = np.random.choice(self.n_states)

        for _ in range(days):
            next_state = np.random.choice(self.n_states, p=self.transition_matrix[current_state])
            return_pct = np.random.choice(self.df[self.df['State'] == next_state]['Daily_Return'])
            next_price = prices[-1] * (1 + return_pct / 100)
            prices.append(next_price)
            current_state = next_state

        return np.array(prices)

    def perform_simulations(self, n_simulations=1000, forecast_days=100):
        """
        Perform multiple price simulations using the Markov model.
        
        TODO:
        - Store and return simulation results.
        """
        initial_price = self.df['Close'].iloc[-1] 

        simulations = np.zeros((n_simulations, forecast_days + 1))

        for i in range(n_simulations):
            simulations[i] = NotImplemented #TODO call the simulate_price_path method

        print(f"Performed {n_simulations} simulations for {forecast_days} days.")
        return simulations

    def backtest(df, transition_matrix, n_simulations=100):
        """
        Perform walk-forward backtesting using the Markov model.
        
        TODO:
        - Implement rolling window logic to create a dynamic transition matrix for each backtest step.
        - Ensure return distributions are created for each state within the rolling window.
        - Simulate next-day prices using the dynamic transition matrix and return distributions.
        """
        window_size = 252  # One trading year
        n_states = transition_matrix.shape[0]
        predictions = np.zeros((len(df), n_simulations))

        # TODO: Iterate through the dataset, starting from the end of the initial rolling window.
        for i in range(window_size, len(df)):
            # TODO: Extract a rolling window of the data to calculate the transition matrix.
            # Hint: Use `iloc` to get rows from `i - window_size` to `i`.
            window_data = NotImplemented  

            # Check if the window data has enough valid states to build a transition matrix.
            if window_data['State'].isna().any() or len(window_data['State'].unique()) < n_states:  # Replace with a condition to check for data sufficiency.
                continue

            # TODO: Create a transition matrix from the rolling window data.
            # Hint: Call a function like `get_transition_matrix` with the states in `window_data`.
            window_transitions = NotImplemented  

            # Create return distributions for each state, ensuring non-empty arrays
            returns_by_state = [
               window_data[window_data['State'] == state]['Daily_Return'].values
                for state in range(n_states)
            ]

            # TODO: Simulate next-day prices using the current price and the transition matrix.
            # Hint: Use `simulate_price_path` for the simulation.
            current_price = df['Close'].iloc[i - 1]
            for sim in range(n_simulations):
                sim_path = NotImplemented  
                predictions[i, sim] = NotImplemented  

        # TODO: Return or process the predictions as needed for analysis.
        valid_indices = np.where(predictions.sum(axis=1) != 0)[0]
        prediction_mean = np.mean(predictions[valid_indices], axis=1)
        prediction_std = NotImplemented
        lower_bound = NotImplemented
        upper_bound = NotImplemented

        return prediction_mean, lower_bound, upper_bound


    def plot_results(df, simulations, backtest_results):
        """
        Create visualization of historical prices, backtest results, and future simulations.
        """
        prediction_mean, lower_bound, upper_bound = backtest_results
        
        # Adjust the prediction arrays to match the index length
        valid_index = len(prediction_mean)
        df_index_subset = df.index[-valid_index:]  # Ensure the x-axis matches the length of the prediction arrays
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot backtesting results
        ax1.plot(df_index_subset, df['Close'].iloc[-valid_index:], color='black', label='Actual Prices')
        ax1.plot(df_index_subset, prediction_mean, color='blue', linestyle='--', label='Model Prediction')
        ax1.fill_between(df_index_subset, lower_bound, upper_bound, color='blue', alpha=0.1, label='95% Confidence Interval')
        ax1.set_title('Backtesting Results')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)

        # Plot future simulations
        future_dates = pd.date_range(
            df.index[-1], 
            periods=simulations.shape[1],
            freq='B'
        )
        
        # Plot individual simulations
        for sim in simulations[:50]:  # Plot first 50 simulations for clarity
            ax2.plot(future_dates, sim, alpha=0.1, color='gray')
        
        # Plot confidence intervals for simulations
        sim_mean = np.mean(simulations, axis=0)
        sim_std = np.std(simulations, axis=0)
        ax2.plot(future_dates, sim_mean, color='red', label='Mean Forecast')
        ax2.fill_between(
            future_dates,
            sim_mean - 1.96 * sim_std,
            sim_mean + 1.96 * sim_std,
            color='red', alpha=0.1,
            label='95% Confidence Interval'
        )
        
        # Plot historical prices for context
        ax2.plot(df.index[-30:], df['Close'][-30:], color='black', label='Historical Prices')
        
        ax2.set_title('Future Price Simulations')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price ($)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def generate_summary_stats(df, simulations, current_price):
        """
        Generate and display summary statistics for future price simulations using tabulate.
        """
        # Calculate average predicted price
        final_prices = simulations[:, -1]  # Last price from each simulation
        mean_final_price = np.mean(final_prices)
        min_final_price = np.min(final_prices)
        max_final_price = np.max(final_prices)
        
        # Calculate prediction intervals
        lower_90 = np.percentile(final_prices, 5)
        upper_90 = np.percentile(final_prices, 95)
        lower_95 = np.percentile(final_prices, 2.5)
        upper_95 = np.percentile(final_prices, 97.5)
        
        # Calculate simulated volatility
        simulated_volatility = np.std(simulations) / np.mean(simulations) * 100  # Percentage format
        
        # Historical volatility (standard deviation of daily returns)
        historical_volatility = df['Daily_Return'].std()
        
        # Create the summary statistics table
        summary_stats = [
            ['Current Price', f"${current_price:.2f}"],
            ['Predicted Price (in 100 days)', f"${mean_final_price:.2f}"],
            ['Average Predicted Price (100 days)', f"${np.mean(final_prices):.2f}"],
            ['Prediction Range', f"${min_final_price:.2f} - ${max_final_price:.2f}"],
            ['Historical Volatility (%)', f"{historical_volatility:.2f}%"],
            ['Simulated Volatility (%)', f"{simulated_volatility:.2f}%"],
            ['90% Confidence Interval (Final Price)', f"${lower_90:.2f} - ${upper_90:.2f}"],
            ['95% Confidence Interval (Final Price)', f"${lower_95:.2f} - ${upper_95:.2f}"]
        ]
        
        # Print the summary statistics using tabulate
        print("\nFuture Prediction Statistics:")
        return(tabulate(summary_stats, headers=['Metric', 'Value'], tablefmt='psql'))
     


if __name__ == "__main__":
    ticker = NotImplemented #TODO replace with Apple Inc. ticker
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2 * 365)

    stock_analysis = StockDataAnalysis(ticker, start_date, end_date)

    stock_analysis.get_stock_data()
    stock_analysis.assign_states()
    stock_analysis.get_transition_matrix()
    stock_analysis.analyze_markov_chain()

    simulations = stock_analysis.perform_simulations(n_simulations=1000, forecast_days=100)
    backtest_results = stock_analysis.backtest(n_simulations=100)

    # TODO: Call plot_results and generate_summary_stats methods
    current_price = NotImplemented # Extract the current closing stock price
