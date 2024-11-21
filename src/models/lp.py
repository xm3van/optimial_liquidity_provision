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

    def deposit_liquidity(self, pool: ConcentratedLiquidityMarket):
        """
        Deposit liquidity into the pool and create a new position.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to deposit liquidity into.
        """
        self._validate_pool(pool)

        # Generate a unique position ID
        position_id = len(self.active_positions) + len(self.historical_positions) + 1

        # Deposit liquidity into the pool
        x0, y0 = pool.deposit(
            lp_id=self.lp_id,
            wealth=self.total_wealth,
            delta_l=self.delta_l,
            delta_u=self.delta_u
        )
        position = pool.active_lp_positions.get(position_id) or pool.inactive_lp_positions.get(position_id)
        
        if position is None:
            raise ValueError("Failed to deposit liquidity due to misaligned tick ranges.")

        # Add the position to the active or inactive dictionary
        if position_id in pool.active_lp_positions:
            self.active_positions[position_id] = position
        else:
            self.historical_positions.append(position)

        logging.info(f"LP {self.lp_id}: Deposited liquidity into position {position_id}. x0={x0}, y0={y0}")

    def withdraw_liquidity(self, pool: ConcentratedLiquidityMarket, position_id: int, terminal_rate: float):
        """
        Withdraw liquidity from a specific position.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to withdraw liquidity from.
        - position_id (int): The position ID to withdraw.
        - terminal_rate (float): The terminal rate at the time of withdrawal.

        Returns:
        - dict: Withdrawal details including terminal wealth and fees earned.
        """
        self._validate_pool(pool)

        # Check active and inactive positions
        if position_id in self.active_positions:
            position = self.active_positions.pop(position_id)
        elif position_id in self.historical_positions:
            position = self.historical_positions.pop(position_id)
        else:
            raise ValueError(f"Position ID {position_id} not found.")

        # Query fees for the position before withdrawing
        self.query_fees(pool, position_id)

        # Withdraw the position
        results = pool.withdraw(position_id=position_id, terminal_rate=terminal_rate)
        alpha_T = results['alpha_T']

        # Update the position record with fees and terminal wealth
        position['fees_earned'] += self.total_fee_income
        position['terminal_wealth'] = alpha_T
        self.historical_positions.append(position)

        # Update total wealth
        self.total_wealth += alpha_T + self.total_fee_income
        logging.info(f"LP {self.lp_id}: Withdrew position {position_id}. Terminal wealth={alpha_T}, Total wealth={self.total_wealth}")

        return results

    def query_fees(self, pool: ConcentratedLiquidityMarket, position_id: int):
        """
        Query the pool for the total fees earned by a specific position.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to query fees from.
        - position_id (int): The position ID to query fees for.
        """
        self._validate_pool(pool)

        if position_id not in self.active_positions:
            raise ValueError(f"Position ID {position_id} not found in active positions.")

        fees = pool.get_lp_fees(self.lp_id)
        self.total_fee_income += fees
        self.active_positions[position_id]['fees_earned'] += fees
        logging.info(f"LP {self.lp_id}: Queried fees for position {position_id}. Fee income={fees}")

    def update_positions_based_on_price(self, pool: ConcentratedLiquidityMarket):
        """
        Update LP's positions (active/inactive) based on the pool's current price.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to check for active/inactive ranges.
        """
        self._validate_pool(pool)

        # Update active positions
        new_active_positions = []
        for position_id, position in self.active_positions.items():
            if not (position['Zl'] <= pool.Z0 <= position['Zu']):
                self.historical_positions.append(position)
            else:
                new_active_positions.append(position_id)

        self.active_positions = {pid: self.active_positions[pid] for pid in new_active_positions}


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
