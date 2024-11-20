import unittest
import numpy as np
import json
from scipy.stats import normaltest, wasserstein_distance

from src.models.token import TokenDTO
from src.models.chain import ChainDTO
from src.prices.marginal_exchange_rate_generator_v2 import MarginalExchangeRateGenerator

class TestMarginalExchangeRateGenerator(unittest.TestCase):

    def setUp(self):
        # Assuming TokenDTO and historical data are available
        network = ChainDTO(name="Ethereum", network_id=1)

        self.token_a = TokenDTO(
            address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            name="USD Coin",
            symbol="USDC",
            decimals=6,
            network=network,
            coingecko_id="usd-coin",
            token_type="stable_token",
        )
        self.token_b = TokenDTO(
            address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            name="Wrapped Ether",
            symbol="WETH",
            decimals=18,
            network=network,
            coingecko_id="weth",
            token_type="collateral_token",
        )

        # Load historical data for tokens
        filepath = './data/ohlc_data/usd-coin_ohlc_data.json'
        with open(filepath, 'r') as file:
            data = json.load(file)
        self.token_a.ohlc_data = data

        filepath = './data/ohlc_data/weth_ohlc_data.json'
        with open(filepath, 'r') as file:
            data = json.load(file)
        self.token_b.ohlc_data = data

        # Initialize the MarginalExchangeRateGenerator
        self.generator = MarginalExchangeRateGenerator(
            token_a=self.token_a,
            token_b=self.token_b,
            freq="1h",
            initial_rate=self.token_a.price / self.token_b.price
        )

        # Generate simulated path and get historical data for tests
        time_horizon = 100
        num_steps = 100
        self.simulated_path = self.generator.generate_stochastic_path(time_horizon, num_steps, plot=False)
        self.historical_data = self.generator.exchange_rate_df_original['close'].values[:num_steps]

    def test_mean_of_simulated_path(self):
        # Test the mean of simulated path vs. historical mean
        sim_mean = np.mean(self.simulated_path)
        hist_mean = np.mean(self.historical_data)
        self.assertAlmostEqual(sim_mean, hist_mean, delta=0.1 * hist_mean,
                               msg=f"Mean of simulated path {sim_mean} differs from historical mean {hist_mean}")

    def test_variance_of_simulated_path(self):
        # Test the variance of simulated path vs. historical variance
        sim_variance = np.var(self.simulated_path)
        hist_variance = np.var(self.historical_data)
        self.assertAlmostEqual(sim_variance, hist_variance, delta=0.1 * hist_variance,
                               msg=f"Variance of simulated path {sim_variance} differs from historical variance {hist_variance}")

    def test_distribution_similarity(self):
        # Test the distribution similarity using Wasserstein Distance
        dist = wasserstein_distance(self.simulated_path, self.historical_data)
        hist_mean = np.mean(self.historical_data)
        self.assertLessEqual(dist, 0.1 * hist_mean,
                             msg=f"Wasserstein distance {dist} indicates dissimilar distributions")

    def test_normality_of_log_returns(self):
        # Test for normality of log returns if the model assumes it
        sim_log_returns = np.diff(np.log(self.simulated_path))
        hist_log_returns = np.diff(np.log(self.historical_data))
        sim_normality_p = normaltest(sim_log_returns).pvalue
        hist_normality_p = normaltest(hist_log_returns).pvalue
        self.assertGreater(sim_normality_p, 0.05, msg="Simulated path log returns deviate from normal distribution")
        self.assertGreater(hist_normality_p, 0.05, msg="Historical data log returns deviate from normal distribution")

if __name__ == "__main__":
    unittest.main()
