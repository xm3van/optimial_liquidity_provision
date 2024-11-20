import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.models.token import TokenDTO
from typing import Optional
import matplotlib.ticker as mticker


@dataclass
class MarginalExchangeRateGenerator:
    token_a: TokenDTO
    token_b: TokenDTO
    freq: str = "1h"  # Default frequency for estimation
    exchange_rate_df_original: Optional[pd.DataFrame] = None
    exchange_rate_df_scaled: Optional[pd.DataFrame] = None
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
        self.estimate_parameters(plot=True)

    def resample_ohlc_data(self, df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
        df_resampled = df.resample(freq).last().interpolate()
        df_resampled.sort_index(inplace=True)
        return df_resampled


    def get_rolling_std(self, df: pd.DataFrame, freq: str, window: str = '1D'):
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        rolling_std = df['log_returns'].rolling(window).std()
        rolling_mu = df['log_returns'].rolling(window).mean() 
        return rolling_std, rolling_mu

    def estimate_parameters(self, freq: str = '1h', window: str = '1D', plot: bool = True):
        self.exchange_rate_df_original = self.exchange_rate()
        df_resampled = self.resample_ohlc_data(self.exchange_rate_df_original, freq)
        self.exchange_rate_df_scaled = df_resampled
        rolling_std, rolling_mu = self.get_rolling_std(df_resampled, freq, window)

        # Sigma calculation
        self.sigma = rolling_std.mean()

        # Mu mean and mu long-term calculations
        self.mu_mean = rolling_mu.mean()
        long_window = '30D'
        rolling_mu_long = df_resampled['log_returns'].rolling(long_window).mean()
        self.mu_long_term = rolling_mu_long.mean()

        # Theta calculation
        valid_mu = rolling_mu.dropna()
        if len(valid_mu) > 1:
            lagged_mu = valid_mu.shift(-1).dropna()
            valid_mu = valid_mu.iloc[:-1]
            if len(valid_mu) > 1:
                autocorr = np.corrcoef(valid_mu, lagged_mu)[0, 1]
                self.theta = -np.log(autocorr) if autocorr > 0 else None
            else:
                self.theta = None
                print("Insufficient data after aligning for theta estimation.")
        else:
            self.theta = None
            print("Insufficient data to estimate theta reliably.")

        # Sigma_mu calculation
        self.sigma_mu = (rolling_mu - self.mu_mean).std()

        # Jump intensity and magnitude calculation using quantile threshold
        log_returns_abs = df_resampled['log_returns'].abs().dropna()
        if not log_returns_abs.empty:
            threshold = log_returns_abs.quantile(0.99)
            jumps = log_returns_abs > threshold
            if jumps.sum() > 0:
                self.jump_intensity = jumps.sum() / len(df_resampled)
                self.jump_magnitude = log_returns_abs[jumps].mean()
            else:
                self.jump_intensity = 0
                self.jump_magnitude = 0
                print("No jumps detected in data; jump intensity and magnitude set to 0.")
        else:
            self.jump_intensity = 0
            self.jump_magnitude = 0
            print("No log returns available for jump calculation.")

        # Plot histogram if requested
        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(df_resampled['log_returns'].dropna(), bins=100, color="pink", label="Log Returns")
            ax.set_xlabel("Log Return")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{self.token_a.symbol.upper()}<>{self.token_b.symbol.upper()}: Intraday Volatility Histogram")
            ax.legend()
            plt.grid(True)
            plt.show()

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

    # def resample_ohlc_random_walk(self, df, freq):
    #     """
    #     Resample OHLC data using random walk interpolation with scaled intra-hour volatility as noise.
        
    #     Args:
    #         df (pd.DataFrame): Original OHLC DataFrame.

    #     Returns:
    #         pd.DataFrame: Resampled DataFrame with realistic intra-hour variability.
    #     """
    #     # Use scaled hourly volatility as the noise level for intra-period variability
    #     noise_level = self.calculate_scale_factor(freq) * self.sigma

    #     df_resampled = df.resample(freq).apply({
    #         'open': 'first',
    #         'high': 'max',
    #         'low': 'min',
    #         'close': 'last'
    #     }).ffill()

    #     for col in ['open', 'high', 'low', 'close']:
    #         noise = np.random.normal(0, noise_level, len(df_resampled))
    #         df_resampled[col] += noise.cumsum()
        
    #     return df_resampled
    

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
            
            # Disable scientific notation on y-axis and force plain formatting
            plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.2f}'))
        
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


    def update_parameters_for_scale(self, target_freq: str):
        """
        Update parameters to match a different target frequency.
        
        Args:
            target_freq (str): Target frequency to scale parameters (e.g., "1min", "5min", "1d").
        """
        scale_factor = self.calculate_scale_factor(target_freq)

        # Scale parameters based on the new frequency
        self.sigma *= scale_factor
        self.mu_mean *= scale_factor
        self.sigma_mu *= scale_factor
        self.jump_magnitude *= scale_factor

    def calculate_scale_factor(self, target_freq: str) -> float:
        """Calculate the scaling factor for volatility based on the target frequency."""
        if target_freq == "1h":
            return 1  # Hourly scale is the base
        elif target_freq == "1min":
            return np.sqrt(1 / 60)
        elif target_freq == "5min":
            return np.sqrt(5 / 60)
        elif target_freq == "1d":
            return np.sqrt(24)
        else:
            raise ValueError(f"Unsupported target frequency: {target_freq}")


    # def resample_ohlc_random_walk(self, df, freq):
    #     """
    #     Resample OHLC data using random walk interpolation with scaled intra-hour volatility as noise.
        
    #     Args:
    #         df (pd.DataFrame): Original OHLC DataFrame.

    #     Returns:
    #         pd.DataFrame: Resampled DataFrame with realistic intra-hour variability.
    #     """
    #     noise_level = self.calculate_scale_factor(freq) * self.sigma

    #     df_resampled = df.resample(freq).apply({
    #         'open': 'first',
    #         'high': 'max',
    #         'low': 'min',
    #         'close': 'last'
    #     }).ffill()

    #     for col in ['open', 'high', 'low', 'close']:
    #         noise = np.random.normal(0, noise_level, len(df_resampled))
    #         df_resampled[col] += noise.cumsum()
        
    #     return df_resampled

    def update_and_resample(self, target_freq: str):
        """
        Update parameters to a different frequency and resample the exchange rate dataframe.
        
        Args:
            target_freq (str): Target frequency for scaling parameters and resampling the data.
        """
        # Update parameters to the new frequency
        self.update_parameters_for_scale(target_freq)

        # Resample the exchange rate dataframe and update it
        if self.exchange_rate_df_scaled is not None:
            self.exchange_rate_df_scaled = self.resample_ohlc_data(self.exchange_rate_df_scaled, target_freq)
        else:
            raise ValueError("exchange_rate_dataframe is None. Run estimate_parameters first to populate it.")

    