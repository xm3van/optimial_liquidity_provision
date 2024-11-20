"""Simple DTO for token data."""
from dataclasses import dataclass 
from src.models.chain import ChainDTO
import pandas as pd
import requests
from requests import get, post
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
import logging

from datetime import datetime
from datetime import timedelta
import json
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
        

class RateLimitExceededException(Exception):
    """Exception raised for rate limit exceeded (429 Too Many Requests)."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after



        
@dataclass()
class TokenDTO:
    """
    Data Transfer Object to store relevant
    token data.
    """

    address: str
    name: str
    symbol: str
    decimals: int
    network: ChainDTO
    coingecko_id: str
    price: Optional[float] = None
    timestamp: Optional[int] = None
    token_type: Optional[str] = None # To-do: make mandatory! 
    ohlc_data: Optional[List[Dict]] = None
    volatility_stats: Optional[List[Dict]] = None
    def __post_init__(self):
        if self.price is None:
            self.price = self.get_current_price()    

        
    def get_current_price(self) -> Optional[float]:
        """
        Fetches the current price of a token from the Coin Llama API and extracts the price.

        Args:
            token_address (str): The address of the token.

        Returns:
            Optional[float]: The current price of the token or None if an error occurs.
        """
        base_url = "https://coins.llama.fi/prices/current"
        url = f"{base_url}/{self.network.name}:{self.address}?searchWidth=12h"
        
        response = get(url)
        if response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            raise RateLimitExceededException("Rate limit exceeded.", retry_after=int(retry_after) if retry_after else None)
        response.raise_for_status()  # Raise an exception for HTTP errors

        data = response.json()
        price_info = data.get('coins', {}).get(f'{self.network.name}:{self.address}', {})
        return price_info.get('price')
    
    def fetch_ohlc_data(self, vs_currency: str = "usd", start_date: str = "2020-01-01", end_date: str = "2024-07-30", coingecko_id_alt: str = None) -> Optional[List[Dict]]:
        """
        Fetch OHLC data for the token from CoinGecko.
        
        Args:
            vs_currency (str): The quote currency.
            start_date (str): Start date for OHLC data in format 'YYYY-MM-DD'.
            end_date (str): End date for OHLC data in format 'YYYY-MM-DD'.
        
        Returns:
            Optional[List[Dict]]: List of OHLC data points or None if an error occurs.
        """
        start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
        request_count = 0
        rate_limit = 500  # Maximum requests per minute
        start_time = datetime.now()
        all_data = []
        current_end_date = end_date_dt

        while current_end_date > start_date_dt:
            segment_start = current_end_date - timedelta(days=31)
            if segment_start < start_date_dt:
                segment_start = start_date_dt

            start_timestamp = int(segment_start.timestamp())
            end_timestamp = int(current_end_date.timestamp())

            if request_count >= rate_limit:
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time < 60:
                    sleep_time = 60 - elapsed_time
                    print(f"Rate limit reached. Sleeping for {sleep_time} seconds.")
                    sleep(sleep_time)
                start_time = datetime.now()
                request_count = 0

            if coingecko_id_alt:
                url = f"https://pro-api.coingecko.com/api/v3/coins/{coingecko_id_alt}/ohlc/range"

            else: 
                url = f"https://pro-api.coingecko.com/api/v3/coins/{self.coingecko_id}/ohlc/range"
            headers = {
                "accept": "application/json",
                "x-cg-pro-api-key": os.getenv("COINGECKO_API_KEY")
            }
            params = {
                "vs_currency": vs_currency,
                "from": start_timestamp,
                "to": end_timestamp,
                "interval": "hourly"
            }

            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data_segment = response.json()
                if not data_segment:
                    print(f"No data available for {self.coingecko_id} from {datetime.fromtimestamp(start_timestamp)} to {datetime.fromtimestamp(end_timestamp)}. Stopping early.")
                    break
                all_data.extend(data_segment)
                request_count += 1
                current_end_date = segment_start
            else:
                print(f"Error fetching data for {self.coingecko_id}: {response.status_code}")
                return None

        all_data.reverse()
        return all_data
    
    def get_ohlc_data(self, vs_currency: str = "usd", start_date: str = "2020-01-01", end_date: str = "2024-07-30") -> None:
        """
        Fetch and save OHLC data.
        
        Args:
            vs_currency (str): The quote currency.
            start_date (str): Start date for OHLC data in format 'YYYY-MM-DD'.
            end_date (str): End date for OHLC data in format 'YYYY-MM-DD'.
        """
        ohlc_data = self.fetch_ohlc_data(vs_currency, start_date, end_date)

        if ohlc_data:
            self.ohlc_data = ohlc_data

    def save_ohlc_data_to_file(self, path: str, filename: Optional[str] = None):
        """
        Save OHLC data to a file.
        
        Args:
            ohlc_data (List[Dict]): OHLC data to save.
            filename (Optional[str]): Optional filename to save the data.
        """
        if self.ohlc_data:
            if not filename:
                filename = f"{self.coingecko_id}_ohlc_data.json"
            with open(os.path.join(path, filename), 'w') as file:
                json.dump(self.ohlc_data, file, indent=4)

    def load_ohlc_data(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load OHLC data from a JSON file into a DataFrame.

        Args:
            filepath (str): Path to the JSON file.

        Returns:
            Optional[pd.DataFrame]: DataFrame containing OHLC data or None if an error occurs.
        """
        try:
            df = pd.read_json(filepath)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close']
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Ensure data is in chronological order
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def format_ohlc_data_to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Format the OHLC data into a usable DataFrame.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame containing OHLC data or None if no data is available.
        """
        if not self.ohlc_data:
            print("No OHLC data available to format.")
            return None

        columns = ["timestamp", "open", "high", "low", "close"]
        df = pd.DataFrame(self.ohlc_data, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.sort_values('timestamp', inplace=True)
        df.set_index("timestamp", inplace=True)
        return df
    
    def plot_closing_prices(self, df: Optional[pd.DataFrame] = None, title: str = 'Closing Price Chart') -> None:
        """
        Plot the closing prices from the OHLC data. The purpose it to quickly check for outliers.
        
        Args:
            df (Optional[pd.DataFrame]): DataFrame containing OHLC data. If None, tries to format OHLC data.
            title (str): Title for the chart.
        """
        if df is None:
            df = self.format_ohlc_data_to_dataframe()

        if df is None or df.empty:
            print("DataFrame is empty. Cannot plot closing prices.")
            return

        plt.figure(figsize=(10, 5))
        df['close'].plot(title=title)
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.grid(True)
        plt.show()

    def resample_ohlc_data(self, df: pd.DataFrame, freq: str = '5T') -> pd.DataFrame:
        """
        Resample OHLC data to a specified frequency.

        Args:
            df (pd.DataFrame): Original OHLC DataFrame.
            freq (str): Frequency string (e.g., '5T' for 5 minutes).

        Returns:
            pd.DataFrame: Resampled DataFrame.
        """
        df_resampled = df.resample(freq).last().interpolate()
        df_resampled.sort_index(inplace=True)
        return df_resampled

    def get_factor(self, freq: str) -> int:
        """Annualizing factor."""
        if freq == "1min":
            return 365 * 24 * 60
        if freq == "5min":
            return 365 * 24 * 12
        if freq == "1h":
            return 365 * 24
        if freq == "1d":
            return 365
        raise ValueError(f"Invalid frequency: {freq}")

    def get_rolling_std(self, df: pd.DataFrame, freq: str, window: str = '1D'):
        """
        Calculate the rolling standard deviation and mean of the log returns.

        Args:
            df (pd.DataFrame): DataFrame containing OHLC data.
            freq (str): Frequency of data.
            window (str): Window size for rolling calculation.

        Returns:
            Tuple: Rolling standard deviation and rolling mean.
        """
        annual_factor = self.get_factor(freq)
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        rolling_std = df['log_returns'].rolling(window).std() * np.sqrt(annual_factor)
        rolling_mu = df['log_returns'].rolling(window).mean() * annual_factor
        return rolling_std, rolling_mu

    def deannualize_volatility(self, vol: float, freq: str) -> float:
        """
        Convert annualized volatility to daily volatility.

        Args:
            annual_vol (float): Annualized volatility.

        Returns:
            float: Daily volatility.
        """
        return vol / np.sqrt(self.get_factor(freq))

    def calc_stressed_volatilities(self, freq: str = '1h', window: str = '1D', plot: bool = False):
        """
        Calculate stressed volatilities from OHLC data and optionally plot the histogram.

        Args:
            filepath (str): Path to the JSON file with OHLC data.
            freq (str): Frequency for resampling the data.
            window (str): Window size for rolling calculation.
            plot (bool): Whether to plot the histogram of volatilities.

        Returns:
            dict: A dictionary with volatility statistics.
        """
        df = self.format_ohlc_data_to_dataframe()
        df_resampled = self.resample_ohlc_data(df, freq)
        rolling_std, rolling_mu = self.get_rolling_std(df_resampled, freq, window)

        stats = {
            "symbol": self.symbol,
            "vol_mean": rolling_std.mean(),
            "vol_median": rolling_std.quantile(0.5),
            "vol_p99": rolling_std.quantile(0.99),
            "vol_max": rolling_std.max(),
            "rolling_std": rolling_std.std(),
            "mu_mean": rolling_mu.mean(),
            "mu_median": rolling_mu.quantile(0.5),
            "mu_p05": rolling_mu.quantile(0.05),
            "mu_p01": rolling_mu.quantile(0.01),
            "mu_min": rolling_mu.min(),
            "freq": freq, 
            "window": window
        }

        self.volatility_stats = stats

        if plot:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(rolling_std.dropna(), bins=50, color="pink")
            ax.set_xlabel("Volatility")
            ax.set_title(f"{self.symbol.upper()}: Intraday Volatility Histogram")
            return stats, fig
        else:
            return stats


    def fit_distribution_from_vol_stats(self, distribution='lognormal'):
        """
        Fit a specified distribution using the volatility statistics calculated.

        Args:
            distribution (str): The type of distribution to fit ('lognormal' or 'normal').

        Returns:
            tuple: Parameters of the fitted distribution.
        """
        if not hasattr(self, 'volatility_stats'):
            print("No volatility statistics available. Please calculate them first.")
            return None

        vol_mean = self.volatility_stats['vol_mean']
        vol_std = self.volatility_stats['rolling_std']  # Approximate std dev from the stats available
        

        if distribution == 'lognormal':
            # Fit a lognormal distribution based on the mean and std
            shape = vol_std / vol_mean
            scale = vol_mean
            return shape, 0, scale
        elif distribution == 'normal':
            # Fit a normal distribution based on the mean and std
            return vol_mean, vol_std
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

    def estimate_price_drop_at_percentile(self, percentile: float, distribution='lognormal', freq: str = '1d') -> float:
        """
        Calculate the price drop percentage corresponding to a given percentile of the fitted distribution.

        Args:
            percentile (float): The percentile (between 0 and 1).
            distribution (str): The type of distribution to fit ('lognormal' or 'normal').
            freq (str): Frequency of data to deannualize the scale parameter.

        Returns:
            float: The price drop percentage at the given percentile.
        """
        params = self.fit_distribution_from_vol_stats(distribution=distribution)
        if params is None:
            return 0.0

        if distribution == 'lognormal':
            shape, loc, scale = params
            # Calculate the value at the given percentile
            value_at_percentile = lognorm.ppf(1 - percentile, shape, loc=loc, scale=scale)
        elif distribution == 'normal':
            mean, std = params
            value_at_percentile = norm.ppf(1 - percentile, loc=mean, scale=std)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        # Convert the log return to a price drop percentage
        price_drop_percentage = - (np.exp(value_at_percentile) - 1) * 100

        return max(0.0, price_drop_percentage)

    def estimate_price_drop_likelihood(self, drop_percentage: float, distribution='lognormal', freq: str = '1d') -> float:
        """
        Estimate the likelihood of a price drop of the specified percentage or more using a fitted distribution.

        Args:
            drop_percentage (float): The percentage drop to estimate the likelihood for.
            distribution (str): The type of distribution to fit ('lognormal' or 'normal').
            freq (str): Frequency of data to deannualize the scale parameter.

        Returns:
            float: Estimated likelihood of the price dropping by the given percentage or more.
        """
        params = self.fit_distribution_from_vol_stats(distribution=distribution)
        if params is None:
            return 0.0

        drop_threshold = np.log(1 - drop_percentage / 100)

        # if distribution == 'lognormal':
        #     shape, loc, scale = params
        #     likelihood = lognorm.sf(drop_threshold, shape, loc=loc, scale=scale)
        # elif distribution == 'normal':
        #     mean, std = params
        #     likelihood = norm.sf(drop_threshold, loc=mean, scale=std)
        if distribution == 'lognormal':
            shape, loc, scale = params
            likelihood = lognorm.cdf(drop_threshold, shape, loc=loc, scale=scale)
        elif distribution == 'normal':
            mean, std = params
            likelihood = norm.cdf(drop_threshold, loc=mean, scale=std)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        # Ensure the likelihood is within the range [0, 1]
        likelihood = max(0.0, min(likelihood, 1.0))

        return likelihood * 100

    def interpretation_price_drop_likelihood(self, drop_percentage: float, distribution='lognormal', freq: str = '1d') -> str:
        """
        Provide an interpretation of the cumulative likelihood of observing a price drop of the specified
        percentage or more using the fitted distribution.

        Args:
            drop_percentage (float): The percentage drop to estimate the likelihood for.
            distribution (str): The type of distribution to fit ('lognormal' or 'normal').
            freq (str): Frequency of data to deannualize the scale parameter.

        Returns:
            str: Interpretation of the likelihood.
        """
        likelihood = self.estimate_price_drop_likelihood(drop_percentage, distribution=distribution, freq=freq)

        return f"Based on a fitted {distribution} distribution (deannualized to {freq} frequency), the estimated probability of the price dropping by {drop_percentage}% or more is {likelihood}."

