# src/models/optimizer.py

class Optimizer:
    def __init__(self, sigma, gamma):
        """
        Initialize the Optimizer with parameters for spread optimization.

        Args:
            sigma (float): Volatility of the pool's price.
            gamma (float): Concentration cost parameter.
        """
        self.sigma = sigma
        self.gamma = gamma

    def calculate_optimal_spread(self, fee_rate, eta=0.001):
        """
        Calculate the optimal spread (delta) based on the fee rate and eta.

        Args:
            fee_rate (float): Fee rate for the pool.
            eta (float): Small adjustment parameter.

        Returns:
            float: Optimal spread (delta).
        """
        return (2 * self.gamma + (self.sigma ** 2)) / (4 * (fee_rate - eta) + 1e-6)
