# src/models/lp.py

from src.models.optimizer import Optimizer
from src.models.pool import ConcentratedLiquidityMarket
import logging

class LiquidityProvider:
    def __init__(self, lp_id: int, initial_wealth: float, delta_u: float, delta_l: float):
        """
        Initialize the Liquidity Provider (LP) with unique ID, initial wealth, and spread.
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
        self.fee_income: float = 0.0
        self.total_wealth: float = initial_wealth

    def __repr__(self) -> str:
        return (f"LiquidityProvider(id={self.lp_id}, initial_wealth={self.initial_wealth}, "
                f"active_positions={len(self.active_positions)}, total_wealth={self.total_wealth})")

    def deposit_liquidity(self, pool: ConcentratedLiquidityMarket):
        """
        Deposit liquidity into the pool and create a new active position.

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
        self.active_positions[position_id] = {
            'position_id': position_id,
            'x0': x0,
            'y0': y0,
            'Zl': pool.Zl,
            'Zu': pool.Zu,
            'fees_earned': 0.0,
        }
        logging.info(f"LP {self.lp_id}: Deposited liquidity into position {position_id}. x0={x0}, y0={y0}")

    def withdraw_liquidity(self, pool: ConcentratedLiquidityMarket, position_id: int, terminal_rate: float):
        """
        Withdraw liquidity from a specific active position.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to withdraw liquidity from.
        - position_id (int): The position ID to withdraw.
        - terminal_rate (float): The terminal rate at the time of withdrawal.

        Returns:
        - dict: Withdrawal details including terminal wealth and fees earned.
        """
        self._validate_pool(pool)

        if position_id not in self.active_positions:
            raise ValueError(f"Position ID {position_id} not found in active positions.")

        # Query fees for the position before withdrawing
        self.query_fees(pool, position_id)

        # Withdraw the position
        results = pool.withdraw(position_id=position_id, terminal_rate=terminal_rate)
        alpha_T = results['alpha_T']

        # Update the position record with fees and terminal wealth
        position = self.active_positions.pop(position_id)
        position['fees_earned'] = self.fee_income
        position['terminal_wealth'] = alpha_T
        self.historical_positions.append(position)

        # Update total wealth
        self.total_wealth += alpha_T + self.fee_income
        logging.info(f"LP {self.lp_id}: Withdrew position {position_id}. Terminal wealth={alpha_T}, Total wealth={self.total_wealth}")

        return results
    

    def is_within_active_range(self, pool: ConcentratedLiquidityMarket) -> bool:
        """
        Check if the LP's position is within the active range of the pool.

        Parameters:
        - pool (ConcentratedLiquidityMarket): The pool to check the active range against.

        Returns:
        - bool: True if the LP's position is active; False otherwise.
        """
        self._validate_pool(pool)

        # Check all active positions
        for position in self.active_positions.values():
            Zl, Zu = position['Zl'], position['Zu']
            if Zl <= pool.Z0 <= Zu:
                return True  # At least one position is active

        return False



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

        self.fee_income = pool.get_lp_fees(self.lp_id)
        logging.info(f"LP {self.lp_id}: Queried fees for position {position_id}. Fee income={self.fee_income}")

    def get_position_details(self, position_id: int) -> dict:
        """
        Retrieve details of a specific position.

        Parameters:
        - position_id (int): The position ID to retrieve details for.

        Returns:
        - dict: Details of the position.
        """
        if position_id in self.active_positions:
            return self.active_positions[position_id]
        for position in self.historical_positions:
            if position['position_id'] == position_id:
                return position
        raise ValueError(f"Position ID {position_id} not found.")

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



# class LiquidityProvider:
#     def __init__(self, lp_id: int, initial_wealth: float, delta_u: float, delta_l: float):
#         """
#         Initialize the Liquidity Provider (LP) with unique ID, initial wealth, and spread.

#         Parameters:
#         - lp_id (int): Unique identifier for the LP.
#         - initial_wealth (float): Initial wealth (\tilde{x}_0).
#         - delta_u (float): Upper spread (\delta_u).
#         - delta_l (float): Lower spread (\delta_\ell).
#         """
#         if initial_wealth <= 0:
#             raise ValueError("Initial wealth must be greater than zero.")
#         if not (0 <= delta_u <= 1 and 0 <= delta_l <= 1):
#             raise ValueError("Spreads (delta_u, delta_l) must be between 0 and 1.")

#         self.lp_id: int = lp_id
#         self.initial_wealth: float = initial_wealth
#         self.delta_u: float = delta_u
#         self.delta_l: float = delta_l
#         self.position: dict = None
#         self.wealth: float = initial_wealth # wealth is current wealth at initalisation
#         self.fee_income: float = 0.0
#         self.is_active: bool = False

#     def __repr__(self) -> str:
#         """
#         String representation of the LiquidityProvider object.
#         """
#         return (f"LiquidityProvider(id={self.lp_id}, initial_wealth={self.initial_wealth}, "
#                 f"delta_u={self.delta_u}, delta_l={self.delta_l}, is_active={self.is_active})")

#     def deposit_liquidity(self, pool: ConcentratedLiquidityMarket) -> None:
#         """
#         Deposit liquidity into the pool.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to deposit liquidity into.
#         """
#         self._validate_pool(pool)

#         x0, y0 = pool.deposit(
#             lp_id=self.lp_id,
#             initial_wealth=self.initial_wealth,
#             delta_l=self.delta_l,
#             delta_u=self.delta_u
#         )
#         self.position = {
#             'x0': x0,
#             'y0': y0,
#             'Zl': pool.Zl,
#             'Zu': pool.Zu
#         }
#         self._update_active_status(pool)
#         logging.info(f"LP {self.lp_id} deposited liquidity: x0={x0}, y0={y0}. Active: {self.is_active}")

#     def withdraw_liquidity(self, pool: ConcentratedLiquidityMarket, terminal_rate: float) -> dict:
#         """
#         Withdraw liquidity from the pool and calculate terminal wealth.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to withdraw liquidity from.
#         - terminal_rate (float): The terminal rate at the time of withdrawal.

#         Returns:
#         - dict: Terminal holdings and wealth information.
#         """
#         self._validate_pool(pool)
#         if not self.position:
#             raise ValueError(f"LP {self.lp_id} has no active position to withdraw.")

#         # Withdraw position
#         results = pool.withdraw(lp_id=self.lp_id, terminal_rate=terminal_rate)
#         alpha_T = results['alpha_T']

#         # Query and add fees
#         self.query_fees(pool)

#         # Update current wealth to include terminal value and fees
#         self.wealth = alpha_T + self.fee_income
#         self.position = None  # Clear position after withdrawal

#         logging.info(f"LP {self.lp_id} withdrew liquidity: terminal_wealth={alpha_T}, total_wealth={self.wealth}")
#         return results

#     def query_fees(self, pool: ConcentratedLiquidityMarket) -> None:
#         """
#         Query the pool for the total fees earned by this LP and update fee income.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to query fees from.
#         """
#         self._validate_pool(pool)
#         if not self.position:
#             raise ValueError(f"LP {self.lp_id} has no active position in the pool.")

#         self.fee_income = pool.get_lp_fees(self.lp_id)
#         logging.info(f"LP {self.lp_id} queried fees: {self.fee_income}")

#     def is_within_active_range(self, pool: ConcentratedLiquidityMarket) -> bool:
#         """
#         Check if the LP's position is within the active range of the pool.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to check the active range against.

#         Returns:
#         - bool: True if the LP's position is active; False otherwise.
#         """
#         self._validate_pool(pool)
#         if not self.position:
#             return False
#         Zl, Zu = self.position['Zl'], self.position['Zu']
#         return Zl <= pool.Z0 <= Zu

#     def get_position_details(self) -> dict:
#         """
#         Retrieve details of the LP's current position.

#         Returns:
#         - dict: Details of the LP's position.
#         """
#         if not self.position:
#             raise ValueError(f"LP {self.lp_id} has no active position.")
#         return self.position

#     def get_terminal_wealth(self) -> float:
#         """
#         Retrieve the LP's terminal wealth after withdrawal.

#         Returns:
#         - float: Terminal wealth of the LP.
#         """
#         if self.wealth is None:
#             raise ValueError(f"LP {self.lp_id} has not withdrawn liquidity yet.")
#         return self.wealth

#     def get_status(self, pool: ConcentratedLiquidityMarket) -> dict:
#         """
#         Get the LP's status, including position details, active range status, and fee income.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to check the active range against.

#         Returns:
#         - dict: Status information of the LP.
#         """
#         active_status = "active" if self.is_within_active_range(pool) else "inactive"
#         position_details = self.get_position_details() if self.position else None
#         return {
#             'id': self.lp_id,
#             'status': active_status,
#             'fees_earned': self.fee_income,
#             'position': position_details
#         }

#     def _update_active_status(self, pool: ConcentratedLiquidityMarket) -> None:
#         """
#         Update the active status of the LP based on the pool's current rate.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool to check the active range against.
#         """
#         self.is_active = self.is_within_active_range(pool)

#     def _validate_pool(self, pool: ConcentratedLiquidityMarket) -> None:
#         """
#         Validate the provided pool instance.

#         Parameters:
#         - pool (ConcentratedLiquidityMarket): The pool instance to validate.
#         """
#         if not isinstance(pool, ConcentratedLiquidityMarket):
#             raise TypeError("Provided object is not a valid ConcentratedLiquidityMarket instance.")




# class LiquidityProvider:
#     def __init__(self, initial_wealth, sigma, gamma):
#         """
#         Initialize the Liquidity Provider with specific parameters.

#         Args:
#             initial_wealth (float): Initial wealth of the LP in the pool.
#             sigma (float): Volatility of the pool's price.
#             gamma (float): Concentration cost parameter.
#         """
#         self.wealth = initial_wealth
#         self.position_value = 0.0
#         self.fee_revenue = 0.0
#         self.rebalancing_costs = 0.0
#         self.sigma = sigma
#         self.gamma = gamma
#         self.optimizer = Optimizer(sigma, gamma)

#     def calculate_pl(self, price, delta):
#         """Calculate Predictable Loss (PL) based on current price and spread (delta)."""
#         if price and delta:
#             self.position_value = - (self.sigma ** 2 / (2 * delta)) * self.wealth
#         else:
#             raise ValueError("Price or delta not set.")
#         return self.position_value

#     def calculate_fee_income(self, fee_rate, delta):
#         """Calculate fee income with depth adjustment and concentration cost."""
#         if delta:
#             fee_income = (4 / delta) * fee_rate * self.wealth - (self.gamma / (delta ** 2)) * self.wealth
#             self.fee_revenue += fee_income  # Accumulate fees
#             return fee_income
#         else:
#             raise ValueError("Delta not set for fee income calculation.")

#     def calculate_rebalancing_costs(self, execution_cost_factor, y_quantity):
#         """Estimate the rebalancing costs based on quantity of asset Y and an execution cost factor."""
#         self.rebalancing_costs = execution_cost_factor * y_quantity  # Assuming cost in terms of reference asset
#         return self.rebalancing_costs

#     def update_wealth(self, price, delta, fee_rate, execution_cost_factor, y_quantity):
#         """
#         Update the total wealth by considering PL, fee income, and rebalancing costs.
        
#         Args:
#             price (float): Current price of the asset.
#             delta (float): Spread width.
#             fee_rate (float): Fee rate for liquidity.
#             execution_cost_factor (float): Cost factor for rebalancing trades.
#             y_quantity (float): Quantity of asset Y for rebalancing.
#         """
#         pl = self.calculate_pl(price, delta)
#         fees = self.calculate_fee_income(fee_rate, delta)
#         rebalancing_cost = self.calculate_rebalancing_costs(execution_cost_factor, y_quantity)
        
#         # Update total wealth
#         self.wealth += pl + fees - rebalancing_cost
#         return self.wealth

#     def optimize_spread(self, fee_rate, eta=0.001):
#         """Calculate the optimal spread (delta) for the LP."""
#         return self.optimizer.calculate_optimal_spread(fee_rate, eta)
