# src/models/pool.py
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from src.models.chain import ChainDTO
from src.models.token import TokenDTO 
import pandas as pd
import math 

@dataclass
class ConcentratedLiquidityMarket:

    # initalisation 
    pool_address: str
    token_a: TokenDTO
    token_b: TokenDTO
    dune_id: int
    network: ChainDTO
    protocol: str
    fee_rate: float = field(default=0.003)  # Π_t, pool fee rate (e.g., 0.3%)

    # state (current & historical)
    Z0: float = field(default=None) # current rate 
    Zl: float = field(default=None) # current lower tick 
    Zu: float = field(default=None) # curent upper tick 
    active_lp_positions: Dict[int, dict] = field(default_factory=dict)  # Active positions by LP ID
    historical_lp_positions: Dict[int, list] = field(default_factory=dict)  # Historical positions by LP ID
    fee_distribution: Dict[int, float] = field(default_factory=dict)  # LP ID to fee income mapping
   
    # metric
    total_fees_collected: float = field(default=0.0)  # Total fees collected



    def __post_init__(self):
        """Post-initialization for setting the pool's name."""
        self.name = f"{self.protocol}:{self.token_a.symbol}-{self.token_b.symbol}"

    def retrieve_data(self, data_loader):
        """Retrieve data for the pool using a DataLoader instance."""
        # To-do:
        # Populate all current LP positions
        self.data = pd.DataFrame(data_loader.load_data())

    def update_fee_rate(self, new_fee_rate: float):
        """
        Update the pool-wide fee rate (\(\Pi\)).
        """
        self.fee_rate = new_fee_rate
        print(f"Updated fee rate to {self.fee_rate}")

    def update_rate(self, rate):
        """Update the current price of the pool."""
        self.Z0 = rate

    def calculate_initial_holdings(self, initial_wealth, Zl, Zu):
        """ Equation 7.1 and 7.2 """
        Z0 = self.Z0
        x0 = y0 = 0  # Initialize holdings

        # Precompute square roots for efficiency
        sqrt_Zl = math.sqrt(Zl)
        sqrt_Zu = math.sqrt(Zu)

        # Case 1: Z0 ≤ Zℓ, LP provides only asset Y
        if Z0 <= Zl:
            x0 = 0
            y0 = initial_wealth / Z0

            # Compute kappa_tilde_0
            denominator = (1 / sqrt_Zl) - (1 / sqrt_Zu)
            self.kappa_tilde_0 = y0 / denominator

        # Case 2: Z0 > Zu, LP provides only asset X
        elif Z0 > Zu:
            x0 = initial_wealth
            y0 = 0

            # Compute kappa_tilde_0
            denominator = sqrt_Zu - sqrt_Zl
            self.kappa_tilde_0 = x0 / denominator

        # Case 3: Zℓ < Z0 ≤ Zu, LP provides both assets
        else:
            # Solve for x0 and y0 using equations (7.2)
            sqrt_Z0 = math.sqrt(Z0)
            sqrt_Zl = math.sqrt(Zl)
            sqrt_Zu = math.sqrt(Zu)

            # Calculate the liquidity depth κ̃0
            self.kappa_tilde_0 = initial_wealth / (
                (sqrt_Z0 - sqrt_Zl) + (1 / sqrt_Z0 - 1 / sqrt_Zu) * Z0
            )

            # Calculate x0 and y0
            x0 = self.kappa_tilde_0 * (sqrt_Z0 - sqrt_Zl)
            y0 = self.kappa_tilde_0 * (1 / sqrt_Z0 - 1 / sqrt_Zu)

        return x0, y0
    
    def calculate_terminal_value(self, terminal_rate):
        """ Equation 7.3 and 7.4 """
        Z_T = terminal_rate
        Zl = self.Zl
        Zu = self.Zu
        kappa_tilde_0 = self.kappa_tilde_0

        # Initialize terminal holdings
        x_T = y_T = 0

        # Case 1: Z_T ≤ Zℓ
        if Z_T <= Zl:
            x_T = 0
            y_T = kappa_tilde_0 * (1 / math.sqrt(Zl) - 1 / math.sqrt(Zu))

        # Case 2: Zℓ < Z_T ≤ Zu
        elif Z_T > Zl and Z_T <= Zu:
            sqrt_Z_T = math.sqrt(Z_T)
            sqrt_Zl = math.sqrt(Zl)
            sqrt_Zu = math.sqrt(Zu)

            x_T = kappa_tilde_0 * (sqrt_Z_T - sqrt_Zl)
            y_T = kappa_tilde_0 * (1 / sqrt_Z_T - 1 / sqrt_Zu)

        # Case 3: Z_T > Zu
        else:
            x_T = kappa_tilde_0 * (math.sqrt(Zu) - math.sqrt(Zl))
            y_T = 0

        # Calculate terminal position value α_T
        alpha_T = x_T + y_T * Z_T

        # Store terminal holdings
        self.x_T = x_T
        self.y_T = y_T

        return alpha_T
    
    def deposit(self, lp_id: int, wealth: float, delta_l: float, delta_u: float) -> Tuple[float, float]:
        """
        LP deposits liquidity into the pool.
        """
        sqrt_Z0 = math.sqrt(self.Z0)
        sqrt_Zl = sqrt_Z0 * (1 - delta_l)
        sqrt_Zu = sqrt_Z0 / (1 - delta_u)
        Zl = sqrt_Zl**2
        Zu = sqrt_Zu**2

        # Calculate initial holdings and liquidity depth
        x0, y0 = self.calculate_initial_holdings(wealth, Zl, Zu)
        position_id = len(self.active_lp_positions) + 1

        # Store active position
        self.active_lp_positions[position_id] = {
            'lp_id': lp_id,
            'position_id': position_id,
            'initial_wealth': wealth,
            'delta_l': delta_l,
            'delta_u': delta_u,
            'Zl': Zl,
            'Zu': Zu,
            'kappa_tilde_0': self.kappa_tilde_0,
            'x0': x0,
            'y0': y0,
            'fees_earned': 0.0
        }
        return x0, y0
    
    def withdraw(self, position_id: int, terminal_rate: float) -> dict:
        """
        LP withdraws liquidity from the pool.
        """
        if position_id not in self.active_lp_positions:
            raise ValueError(f"Position ID {position_id} not found in active positions.")

        # Retrieve and remove the active position
        position = self.active_lp_positions.pop(position_id)

        # Calculate terminal value
        self.Zl = position['Zl']
        self.Zu = position['Zu']
        self.kappa_tilde_0 = position['kappa_tilde_0']
        alpha_T = self.calculate_terminal_value(terminal_rate)

        # Add position to historical records
        lp_id = position['lp_id']
        position['terminal_rate'] = terminal_rate
        position['terminal_wealth'] = alpha_T
        if lp_id not in self.historical_lp_positions:
            self.historical_lp_positions[lp_id] = []
        self.historical_lp_positions[lp_id].append(position)

        return {
            'alpha_T': alpha_T,
            'x_T': self.x_T,
            'y_T': self.y_T
        }
    
    def total_pool_depth(self):
        """
        Calculate the total liquidity depth κ in the pool for the active range.
        Only includes LPs whose ranges cover the current rate Z0.
        """
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")

        active_depth = 0.0

        for position in self.active_lp_positions.values():
            # Include only LPs whose ranges cover Z0
            if position['Zl'] <= self.Z0 <= position['Zu']:
                active_depth += position['kappa_tilde_0']

        return active_depth
    
    def get_active_lps(self):
        """
        Get a list of LPs with active liquidity in the current pool rate range.
        """
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")

        active_lps = []

        for lp_id, position in self.active_lp_positions.items():
            if position['Zl'] <= self.Z0 <= position['Zu']:
                active_lps.append(lp_id)

        return active_lps


    def distribute_fees(self, trading_volume: float):
        """
        Distribute fees to LPs based on their liquidity contributions in active ranges.

        Parameters:
        - trading_volume: Total trading volume in the pool during the fee period.
        """
        if not self.active_lp_positions:
            print("No active LPs to distribute fees.")
            return

        # Calculate total fees collected
        total_fees = self.fee_rate * trading_volume
        self.total_fees_collected += total_fees

        # Calculate pool-wide liquidity depth
        total_depth = self.total_pool_depth()

        # Distribute fees to active LPs
        for lp_id, position in self.active_lp_positions.items():
            # Contribution proportional to kappa_tilde_0
            kappa_tilde = position['kappa_tilde_0']
            fee_share = (kappa_tilde / total_depth) * total_fees

            # Update LP's earned fees
            position['fees_earned'] += fee_share
            self.fee_distribution[lp_id] = position['fees_earned']

        print(f"Distributed {total_fees} in fees among active LPs.")


    def get_lp_fees(self, lp_id: int) -> float:
        """
        Get the total fees earned by a specific LP for active position.

        Parameters:
        - lp_id: The ID of the LP.

        Returns:
        - Total fees earned by the LP.
        """
        if lp_id not in self.active_lp_positions:
            raise ValueError(f"LP with ID {lp_id} not found in the pool.")
        return self.active_lp_positions[lp_id]['fees_earned']



    # def add_tick_range(self, Z_l: float, Z_u: float, kappa_tilde: float):
    #     """Add a new liquidity range (tick) to the pool."""
    #     self.tick_ranges.append((Z_l, Z_u, kappa_tilde))

    # def calculate_position(self, Z: float, Z_l: float, Z_u: float, kappa_tilde: float):
    #     """Calculate the asset quantities based on Z's position relative to Z_l and Z_u."""
    #     if Z <= Z_l:
    #         x = 0
    #         y = kappa_tilde * ((Z_l)**-0.5 - (Z_u)**-0.5)
    #     elif Z_l < Z <= Z_u:
    #         x = kappa_tilde * (Z**0.5 - (Z_l)**0.5)
    #         y = kappa_tilde * (Z**-0.5 - (Z_u)**-0.5)
    #     else:
    #         x = kappa_tilde * ((Z_u)**0.5 - (Z_l)**0.5)
    #         y = 0
    #     return x, y

    # def get_total_depth(self, Z: float) -> float:
    #     """Calculate the total depth of liquidity in the current tick range containing Z."""
    #     total_depth = sum(kappa_tilde for Z_l, Z_u, kappa_tilde in self.tick_ranges if Z_l < Z <= Z_u)
    #     return total_depth

    # def distribute_fees(self, Z: float, total_fees: float) -> List[float]:
    #     """Distribute fees to LPs based on their depth in the range containing Z."""
    #     total_depth = self.get_total_depth(Z)
    #     if total_depth == 0:
    #         return [0] * len(self.tick_ranges)  # No active LPs in this range

    #     fee_shares = []
    #     for Z_l, Z_u, kappa_tilde in self.tick_ranges:
    #         if Z_l < Z <= Z_u:
    #             fee_share = (kappa_tilde / total_depth) * total_fees
    #         else:
    #             fee_share = 0
    #         fee_shares.append(fee_share)
    #     return fee_shares

    # def update_positions(self, Z: float):
    #     """Recalculate asset quantities for each tick range based on current price Z."""
    #     updated_positions = []
    #     for Z_l, Z_u, kappa_tilde in self.tick_ranges:
    #         x, y = self.calculate_position(Z, Z_l, Z_u, kappa_tilde)
    #         updated_positions.append((x, y))
    #     return updated_positions
