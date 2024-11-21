# src/models/lp.py

from src.models.optimizer import Optimizer
from src.models.pool import ConcentratedLiquidityMarket
import logging
from typing import List

class LiquidityProvider:
    def __init__(self, lp_id: int, initial_wealth: float, delta_u: float, delta_l: float):
        """
        Initialize the Liquidity Provider (LP) with unique ID, initial wealth, and spread.

        Parameters:
        - lp_id (int): Unique identifier for the LP.
        - initial_wealth (float): Initial wealth of the LP.
        - delta_u (float): Upper spread as a fraction of the rate.
        - delta_l (float): Lower spread as a fraction of the rate.
        """
        if initial_wealth <= 0:
            raise ValueError("Initial wealth must be greater than zero.")
        if not (0 <= delta_u <= 1 and 0 <= delta_l <= 1):
            raise ValueError("Spreads (delta_u, delta_l) must be between 0 and 1.")

        self.lp_id: int = lp_id
        self.initial_wealth: float = initial_wealth
        self.delta_u: float = delta_u
        self.delta_l: float = delta_l
        self.active_positions: dict = {}  # Tracks all active positions by position ID
        self.historical_positions: list = []  # Tracks all withdrawn positions
        self.total_fee_income: float = 0.0  # Tracks total fee income
        self.total_wealth: float = initial_wealth  # Tracks total wealth (initial + earned)

    def __repr__(self) -> str:
        return (f"LiquidityProvider(id={self.lp_id}, initial_wealth={self.initial_wealth}, "
                f"active_positions={len(self.active_positions)}, total_wealth={self.total_wealth})")

    def deposit_liquidity(self, pool: ConcentratedLiquidityMarket, wealth: float = None, delta_l: float = None, delta_u: float = None):
        """
        Deposit liquidity into the pool and create a new position.
        """
        self._validate_pool(pool)

        # Use provided values or defaults
        wealth = wealth if wealth is not None else self.total_wealth
        delta_l = delta_l if delta_l is not None else self.delta_l
        delta_u = delta_u if delta_u is not None else self.delta_u

        # Ensure wealth does not exceed total wealth
        if wealth > self.total_wealth:
            raise ValueError("Cannot deposit more wealth than available.")

        # Deposit liquidity into the pool
        x0, y0, _ = pool.deposit(
            lp_id=self.lp_id,
            wealth=wealth,
            delta_l=delta_l,
            delta_u=delta_u
        )
        position_id = len(self.active_positions) + len(self.historical_positions) + 1

        # Retrieve the position from the pool
        position = pool.active_lp_positions.get(position_id) or pool.inactive_lp_positions.get(position_id)
        
        if position is None:
            raise ValueError("Failed to deposit liquidity due to misaligned tick ranges.")

        # Add the position to the LP's active positions
        self.active_positions[position_id] = position

        # Update LP's total wealth
        self.total_wealth -= wealth  # Subtract the deposited wealth

        logging.info(f"LP {self.lp_id}: Deposited liquidity into position {position_id}. x0={x0}, y0={y0}")


    def withdraw_liquidity(self, pool: ConcentratedLiquidityMarket, position_id: int, terminal_rate: float):
        """
        Withdraw liquidity from a specific position.
        """
        self._validate_pool(pool)

        # Ensure the position belongs to this LP
        if position_id not in self.active_positions:
            raise ValueError(f"Position ID {position_id} not found in LP's active positions.")

        # Query fees for the position before withdrawing
        fees_earned = self.query_fees(pool, position_id)

        # Withdraw the position from the pool
        results = pool.withdraw(position_id=position_id, terminal_rate=terminal_rate)
        alpha_T = results['alpha_T']

        # Retrieve the position data
        position = self.active_positions.pop(position_id)
        position['terminal_wealth'] = alpha_T
        position['fees_earned'] += fees_earned

        # Add the position to historical positions
        self.historical_positions.append(position)

        # Update LP's total wealth
        self.total_wealth += alpha_T + fees_earned  # Add back the withdrawn wealth and fees

        logging.info(f"LP {self.lp_id}: Withdrew position {position_id}. Terminal wealth={alpha_T}, Fees earned={fees_earned}, Total wealth={self.total_wealth}")

        return results

    def query_fees(self, pool: ConcentratedLiquidityMarket, position_id: int) -> float:
        """
        Query the pool for the total fees earned by a specific position.
        """
        self._validate_pool(pool)

        if position_id not in self.active_positions:
            raise ValueError(f"Position ID {position_id} not found in LP's active positions.")

        # Retrieve fees from the pool
        position = self.active_positions[position_id]
        fees = position.get('fees_earned', 0.0)

        # Update LP's total fee income
        self.total_fee_income += fees

        logging.info(f"LP {self.lp_id}: Queried fees for position {position_id}. Fee income={fees}")

        return fees

    def update_positions_based_on_price(self, pool: ConcentratedLiquidityMarket):
        """
        Update LP's positions based on the pool's current price.
        """
        self._validate_pool(pool)

        updated_active_positions = {}
        for position_id, position in self.active_positions.items():
            if position['Zl'] <= pool.Z0 <= position['Zu']:
                # Keep position active
                updated_active_positions[position_id] = position
            else:
                # Move position to historical
                position['fees_earned'] += pool.get_lp_fees(self.lp_id)
                self.historical_positions.append(position)

        # Update the LP's active positions
        self.active_positions = updated_active_positions

        # Log the update
        logging.info(f"LP {self.lp_id}: Updated active positions based on price. Active={len(self.active_positions)}, Historical={len(self.historical_positions)}")
    def get_active_positions_in_range(self, pool: ConcentratedLiquidityMarket) -> List[int]:
        """
        Get the list of active position IDs within the current active tick range.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to check for active ranges.

        Returns:
        - List[int]: List of position IDs that are active in the current tick range.
        """
        self._validate_pool(pool)
        return [pid for pid, pos in self.active_positions.items() if pos['Zl'] <= pool.Z0 <= pos['Zu']]


    def get_total_wealth(self) -> float:
        """
        Retrieve the LP's total wealth (current wealth + fees).

        Returns:
        - float: Total wealth of the LP.
        """
        return self.total_wealth

    def _validate_pool(self, pool: ConcentratedLiquidityMarket):
        """
        Validate the provided pool instance.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool instance to validate.
        """
        if not isinstance(pool, ConcentratedLiquidityMarket):
            raise TypeError("Provided object is not a valid ConcentratedLiquidityMarket instance.")
