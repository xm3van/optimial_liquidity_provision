# src/backtest.py

import pandas as pd
from src.models.lp import LiquidityProvider
from src.data.data_loader import DataLoader

class Backtest:
    def __init__(self, pool, liquidity_provider, data_loader, fee_rate=0.003, eta=0.001):
        """
        Initialize the Backtest with pool and LP details.

        Args:
            pool (PoolDTO): Pool object representing the liquidity pool.
            liquidity_provider (LiquidityProvider): LP object for strategy calculations.
            data_loader (DataLoader): Instance to load historical data.
            fee_rate (float): Fee rate for the pool.
            eta (float): Small parameter adjustment for spread optimization.
        """
        self.pool = pool
        self.lp = liquidity_provider
        self.data_loader = data_loader
        self.fee_rate = fee_rate
        self.eta = eta
        self.results = []

    def run(self):
        """Run the backtesting simulation and record results for each step."""
        
        # Load historical data for the pool
        self.pool.retrieve_data(self.data_loader)
        
        # Iterate over each time period (e.g., each row in the dataset)
        for _, row in self.pool.data.iterrows():
            self.pool.update_price(row['current_price'])
            
            # Calculate optimal spread (delta), PL, and fee income for this period
            delta_optimal = self.lp.optimize_spread(fee_rate=self.fee_rate, eta=self.eta)
            pl = self.lp.calculate_pl(self.pool.current_price, delta_optimal)
            fee_income = self.lp.calculate_fee_income(self.fee_rate, delta=delta_optimal)

            # Record results for this period
            period_result = {
                'timestamp': row['created_date'],
                'price': self.pool.current_price,
                'optimal_spread': delta_optimal,
                'predictable_loss': pl,
                'fee_income': fee_income,
                'net_profit': fee_income + pl  # Calculate net gain/loss for this period
            }
            self.results.append(period_result)
            
        # Convert results to a DataFrame for easier analysis
        self.results_df = pd.DataFrame(self.results)
        print("Backtesting complete. Results are stored in `results_df`.")

    def evaluate_performance(self):
        """Evaluate and print overall performance metrics for the backtest."""
        # Calculate cumulative net profit
        cumulative_profit = self.results_df['net_profit'].sum()
        total_fee_income = self.results_df['fee_income'].sum()
        total_predictable_loss = self.results_df['predictable_loss'].sum()

        print(f"Cumulative Net Profit: {cumulative_profit}")
        print(f"Total Fee Income: {total_fee_income}")
        print(f"Total Predictable Loss: {total_predictable_loss}")

        return {
            "cumulative_profit": cumulative_profit,
            "total_fee_income": total_fee_income,
            "total_predictable_loss": total_predictable_loss
        }
