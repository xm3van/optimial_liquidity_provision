import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.models.token import TokenDTO
from typing import Optional

@dataclass
class MarginalExchangeRateGenerator:
    token_a: TokenDTO
    token_b: TokenDTO
    freq: str = "1h"  # Default frequency for estimation
    scaled: bool = False # Boolean to inform if it has be scaled already
    exchange_rate_dataframe: Optional[pd.DataFrame] = None
    exchange_rate_scaled: Optional[pd.DataFrame] = None
    initial_rate: Optional[float] = None
    sigma: Optional[float] = None  # Hourly volatility of Z_t
    mu_mean: Optional[float] = None  # Hourly drift for Z_t
    theta: Optional[float] = None  # Mean reversion speed for mu_t
    mu_long_term: Optional[float] = None  # Long-term mean for drift
    sigma_mu: Optional[float] = None  # Volatility of the drift process
    jump_intensity: Optional[float] = None  # Jump intensity for mu_t
    jump_magnitude: Optional[float] = None  # Jump magnitude for mu_t


    def __post_init__(self):
        if self.initial_rate is None:
            self.initial_rate = self.token_a.price / self.token_b.price
        self.estimate_parameters(plot=True)  # Estimate parameters at the hourly scale by default

    def estimate_parameters(self, plot: bool = True):
        """Estimate all required parameters at the default (hourly) frequency."""
        exchange_ohlc = self.exchange_rate()
        self.exchange_rate_dataframe = exchange_ohlc
        exchange_ohlc['returns'] = exchange_ohlc['close'].pct_change()
        log_returns = np.log(exchange_ohlc['close'] / exchange_ohlc['close'].shift(1)).dropna()

        # Set parameters
        self.sigma = log_returns.std()  # Set hourly volatility
        self.mu_mean = log_returns.mean()  # Set hourly drift
        self.mu_long_term = self.mu_mean
        self.theta = -np.log(np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1])  # Mean reversion speed
        self.sigma_mu = (log_returns - self.mu_mean).std()  # Volatility of the drift process

        # Calculate jump parameters
        threshold = 1.5 * log_returns.std()  
        self.jump_intensity = (np.abs(log_returns) > threshold).sum() / len(log_returns)
        self.jump_magnitude = np.mean(np.abs(log_returns[np.abs(log_returns) > threshold]))

        # Plot log returns if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(log_returns.dropna(), bins=50, color="pink")
            ax.set_xlabel("Volatility")
            ax.set_title(f"{self.token_a.symbol.upper()}<>{self.token_b.symbol.upper()}: Intraday Volatility Histogram")

    def update_parameters_for_scale(self, target_freq: str):
        """
        Update parameters to match a different target frequency.
        
        Args:
            target_freq (str): Target frequency to scale parameters (e.g., "1min", "5min", "1d").
        """
        if target_freq == "1h":
            scale_factor = 1  # Hourly scale is the base
        elif target_freq == "1min":
            scale_factor = np.sqrt(1 / 60)
        elif target_freq == "5min":
            scale_factor = np.sqrt(5 / 60)
        elif target_freq == "1d":
            scale_factor = np.sqrt(24)
        else:
            raise ValueError(f"Unsupported target frequency: {target_freq}")

        # Scale parameters based on the new frequency
        self.sigma *= scale_factor
        self.mu_mean *= scale_factor
        self.sigma_mu *= scale_factor
        self.jump_magnitude *= scale_factor

    def exchange_rate(self):
        """Calculate the OHLC for token_a/token_b exchange rate from OHLC data of token_a and token_b."""
        df_a = self.token_a.format_ohlc_data_to_dataframe()
        df_b = self.token_b.format_ohlc_data_to_dataframe()
        
        merged_df = df_a[['open', 'high', 'low', 'close']].rename(
            columns={'open': 'a_open', 'high': 'a_high', 'low': 'a_low', 'close': 'a_close'}
        ).join(
            df_b[['open', 'high', 'low', 'close']].rename(
                columns={'open': 'b_open', 'high': 'b_high', 'low': 'b_low', 'close': 'b_close'}
            ),
            how='inner'
        )

        merged_df['open'] = merged_df['a_open'] / merged_df['b_open']
        merged_df['high'] = merged_df['a_high'] / merged_df['b_low']
        merged_df['low'] = merged_df['a_low'] / merged_df['b_high']
        merged_df['close'] = merged_df['a_close'] / merged_df['b_close']
        
        return merged_df[['open', 'high', 'low', 'close']]

    def resample_ohlc_random_walk(self, df, freq):
        """
        Resample OHLC data using random walk interpolation with scaled intra-hour volatility as noise.
        
        Args:
            df (pd.DataFrame): Original OHLC DataFrame.

        Returns:
            pd.DataFrame: Resampled DataFrame with realistic intra-hour variability.
        """
        # Use scaled hourly volatility as the noise level for intra-period variability
        noise_level = self.calculate_scale_factor(freq) * self.sigma

        df_resampled = df.resample(freq).apply({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).ffill()

        for col in ['open', 'high', 'low', 'close']:
            noise = np.random.normal(0, noise_level, len(df_resampled))
            df_resampled[col] += noise.cumsum()
        
        return df_resampled
    
    def update_parameters_for_scale(self, target_freq: str):
        """
        Update parameters to match a different target frequency.
        
        Args:
            target_freq (str): Target frequency to scale parameters (e.g., "1min", "5min", "1d").
        """
        if self.scaled is False:
            self.update_parameters_for_scale(target_freq=target_freq)
            self.exchange_rate_scaled = self.resample_ohlc_random_walk(df=self.exchange_rate_dataframe,
                                                                       target_freq=target_freq)
            self.scaled = True
            self.freq = target_freq
        else:
            print(f"Exchange Rate is already scaled!")


    def simulate_drift_with_jumps(self, time_horizon, num_steps):
        """Simulate the dynamic drift μ_t as an Ornstein-Uhlenbeck process with jumps."""
        dt = time_horizon / num_steps
        mu_path = np.zeros(num_steps)
        mu_path[0] = self.mu_mean  

        for t in range(1, num_steps):
            dW_mu = np.random.normal(0, np.sqrt(dt))
            mu_path[t] = mu_path[t-1] + self.theta * (self.mu_long_term - mu_path[t-1]) * dt + self.sigma_mu * dW_mu

            if np.random.rand() < self.jump_intensity * dt:
                mu_path[t] += np.random.normal(0, self.jump_magnitude)

        self.mu_path = mu_path
        return mu_path

    def generate_stochastic_path(self, time_horizon, num_steps, plot=False):
        """Generate a simulated path for the marginal exchange rate using dynamic μ_t and stochastic dynamics."""
        dt = time_horizon / num_steps
        Z_path = np.zeros(num_steps)
        Z_path[0] = self.initial_rate

        self.simulate_drift_with_jumps(time_horizon, num_steps)

        for t in range(1, num_steps):
            dW = np.random.normal(0, np.sqrt(dt))  
            Z_path[t] = Z_path[t-1] * (1 + self.mu_path[t] * dt + self.sigma * dW)

        if plot:
            plt.figure(figsize=(12, 6))
            plt.plot(np.linspace(0, time_horizon, num_steps), Z_path, label="Simulated Exchange Rate Path")
            plt.title("Stochastic Path of the Marginal Exchange Rate")
            plt.xlabel("Time")
            plt.ylabel("Exchange Rate")
            plt.legend()
            plt.grid(True)
            plt.show()

        return Z_path


    def plot_closing_prices(self, df: Optional[pd.DataFrame] = None, title: str = 'Closing Price Chart') -> None:
        """
        Plot the closing prices from the OHLC data. The purpose it to quickly check for outliers.
        
        Args:
            df (Optional[pd.DataFrame]): DataFrame containing OHLC data. If None, tries to format OHLC data.
            title (str): Title for the chart.
        """
        df = self.exchange_rate()

        plt.figure(figsize=(10, 5))
        df['close'].plot(title=title)
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.grid(True)
        plt.show()