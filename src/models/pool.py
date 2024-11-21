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
    tick_spacing: int =field(default=60)

    # state (current & historical)
    Z0: float = field(default=None) # current rate 
    active_lp_positions: Dict[int, dict] = field(default_factory=dict)  # Active positions by LP ID
    inactive_lp_positions: Dict[int, dict] = field(default_factory=dict)  # Inactive positions by position ID
    historical_lp_positions: Dict[int, list] = field(default_factory=dict)  # Historical positions by LP ID
    fee_distribution: Dict[int, float] = field(default_factory=dict)  # LP ID to fee income mapping
   
    ticks: List[float] = field(default_factory=list)  # Sorted list of tick prices
    tick_liquidities: Dict[int, float] = field(default_factory=dict)  # Tick index to net liquidity change mapping

    # metric
    total_fees_collected: float = field(default=0.0)  # Total fees collected

    def __post_init__(self):
        """Post-initialization for setting the pool's name."""
        self.name = f"{self.protocol}:{self.token_a.symbol}-{self.token_b.symbol}"
        self.initialize_ticks()

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
        self.update_positions_based_on_price()

    def initialize_ticks(self, min_tick: int = -887272, max_tick: int = 887272, tick_spacing: int = 60):
        """
        Initialize the global tick grid for the pool.

        Parameters:
        - min_tick (int): Minimum tick index.
        - max_tick (int): Maximum tick index.
        - tick_spacing (int): Number of ticks between initialized ticks (tick spacing).
        """
        self.tick_spacing = tick_spacing
        # Generate ticks from min_tick to max_tick with the specified spacing
        self.ticks = [tick for tick in range(min_tick, max_tick + 1, tick_spacing)]
        # Convert ticks to prices
        self.tick_to_price_map = {tick: self.tick_to_price(tick) for tick in self.ticks}
        print(f"Initialized ticks from {min_tick} to {max_tick} with spacing {tick_spacing}.")

    def tick_to_price(self, tick: int) -> float:
        """
        Convert a tick index to the corresponding price.
        """
        return 1.0001 ** tick


    def price_to_tick(self, price: float) -> int:
        """
        Convert a price to the nearest tick index.
        """
        return int(math.log(price) / math.log(1.0001))


    def calculate_initial_holdings(self, wealth, Zl, Zu):
        """ Equation 7.1 and 7.2 """
        Z0 = self.Z0
        x0 = y0 = 0  # Initialize holdings
        sqrt_Zl = math.sqrt(Zl)
        sqrt_Zu = math.sqrt(Zu)
        sqrt_Z0 = math.sqrt(Z0)

        # LP trading conditions
        if Z0 <= Zl:
            # Case 1: Provide only asset Y
            denominator = (1 / sqrt_Zl) - (1 / sqrt_Zu)
            kappa_tilde = wealth / (Z0 * denominator)
            x0 = 0
            y0 = kappa_tilde * denominator
        elif Z0 > Zu:
            # Case 2: Provide only asset X
            denominator = sqrt_Zu - sqrt_Zl
            kappa_tilde = wealth / denominator
            x0 = kappa_tilde * denominator
            y0 = 0
        else:
            # Case 3: Provide both assets
            denominator = (sqrt_Z0 - sqrt_Zl) + Z0 * ((1 / sqrt_Z0) - (1 / sqrt_Zu))
            kappa_tilde = wealth / denominator
            x0 = kappa_tilde * (sqrt_Z0 - sqrt_Zl)
            y0 = kappa_tilde * ((1 / sqrt_Z0) - (1 / sqrt_Zu))

        return x0, y0, kappa_tilde
    
    def calculate_terminal_value(self, Z_T: float, Zl: float, Zu: float, kappa_tilde: float) -> Tuple[float, float, float]:
        """ Equation 7.3 and 7.4 """
        sqrt_Z_T = math.sqrt(Z_T)
        sqrt_Zl = math.sqrt(Zl)
        sqrt_Zu = math.sqrt(Zu)

        # Initialize terminal holdings
        x_T = y_T = 0

        # Case 1: Z_T ≤ Zℓ
        if Z_T <= Zl:
            x_T = 0
            y_T = kappa_tilde * (1 / math.sqrt(Zl) - 1 / math.sqrt(Zu))

        # Case 2: Zℓ < Z_T ≤ Zu
        elif Zl < Z_T <= Zu:
            x_T = kappa_tilde * (sqrt_Z_T - sqrt_Zl)
            y_T = kappa_tilde * (1 / sqrt_Z_T - 1 / sqrt_Zu)

        # Case 3: Z_T > Zu
        else:
            x_T = kappa_tilde * (sqrt_Zu - sqrt_Zl)
            y_T = 0

        # Calculate terminal position value α_T
        alpha_T = x_T + y_T * Z_T
        return alpha_T, x_T, y_T
    
    def deposit(self, lp_id: int, wealth: float, delta_l: float, delta_u: float) -> Tuple[float, float]:
        """
        LP deposits liquidity into the pool.
        """
        sqrt_Z0 = math.sqrt(self.Z0)
        sqrt_Zl = sqrt_Z0 * (1 - delta_l / 2)
        sqrt_Zu = sqrt_Z0 / (1 - delta_u / 2)
        Zl = sqrt_Zl ** 2
        Zu = sqrt_Zu ** 2

        # Convert prices to ticks and snap to nearest valid ticks
        tick_l = self.price_to_tick(Zl)
        tick_u = self.price_to_tick(Zu)

        # Align ticks to valid tick boundaries (multiple of tick_spacing)
        tick_l = tick_l - (tick_l % self.tick_spacing)
        tick_u = tick_u + (self.tick_spacing - (tick_u % self.tick_spacing)) if tick_u % self.tick_spacing != 0 else tick_u

        # Ensure that tick_l < tick_u
        if tick_l >= tick_u:
            raise ValueError("Lower tick must be less than upper tick.")

        # Convert ticks back to prices
        Zl = self.tick_to_price(tick_l)
        Zu = self.tick_to_price(tick_u)

        # Calculate initial holdings and liquidity depth
        x0, y0, kappa_tilde = self.calculate_initial_holdings(wealth, Zl, Zu)

        position_id = len(self.active_lp_positions) + len(self.inactive_lp_positions) + 1

        # Store position
        position = {
            'lp_id': lp_id,
            'position_id': position_id,
            'initial_wealth': wealth,
            'delta_l': delta_l,
            'delta_u': delta_u,
            'Zl': Zl,
            'Zu': Zu,
            'tick_l': tick_l,
            'tick_u': tick_u,
            'kappa_tilde_0': kappa_tilde,
            'x0': x0,
            'y0': y0,
            'fees_earned': 0.0
        }

        # Determine if the position is active
        if Zl <= self.Z0 <= Zu:
            self.active_lp_positions[position_id] = position
        else:
            self.inactive_lp_positions[position_id] = position

        # Update tick liquidities
        self.update_tick_liquidities(tick_l, tick_u, kappa_tilde)

        return x0, y0, position_id

    
    def get_nearest_tick_price(self, price: float) -> float:
        """
        Find the nearest tick price to the given price.
        """
        nearest_tick_price = min(self.ticks, key=lambda x: abs(x - price))
        return nearest_tick_price
    
    def update_tick_liquidities(self, tick_l: int, tick_u: int, kappa_tilde: float):
        """
        Update net liquidity changes at tick boundaries for the LP's position.
        """
        # Increase liquidity at tick_l
        self.tick_liquidities[tick_l] = self.tick_liquidities.get(tick_l, 0.0) + kappa_tilde
        # Decrease liquidity at tick_u
        self.tick_liquidities[tick_u] = self.tick_liquidities.get(tick_u, 0.0) - kappa_tilde

        # debug statement
        print(f"Updated tick liquidity: Tick_L {tick_l} -> {self.tick_liquidities[tick_l]}, "
          f"Tick_U {tick_u} -> {self.tick_liquidities[tick_u]}.")

    def withdraw(self, position_id: int, terminal_rate: float) -> dict:
        """
        LP withdraws liquidity from the pool.
        """
        # Check if position is active or inactive
        if position_id in self.active_lp_positions:
            position = self.active_lp_positions.pop(position_id)
        elif position_id in self.inactive_lp_positions:
            position = self.inactive_lp_positions.pop(position_id)
        else:
            raise ValueError(f"Position ID {position_id} not found.")

        lp_id = position['lp_id']
        Zl = position['Zl']
        Zu = position['Zu']
        kappa_tilde = position['kappa_tilde_0']

        # Remove liquidity from tick ranges
        self.remove_tick_liquidities(position['tick_l'], position['tick_u'], kappa_tilde)

        # Calculate terminal value
        alpha_T, x_T, y_T = self.calculate_terminal_value(terminal_rate, Zl, Zu, kappa_tilde)

        # Reset fee earnings for this position
        position['fees_earned'] = 0.0

        # Add position to historical records
        position['terminal_rate'] = terminal_rate
        position['terminal_wealth'] = alpha_T
        if lp_id not in self.historical_lp_positions:
            self.historical_lp_positions[lp_id] = []
        self.historical_lp_positions[lp_id].append(position)

        # Debugging output
        print(f"Position {position_id} withdrawn. Terminal wealth: {alpha_T}")


        return {
            'alpha_T': alpha_T,
            'x_T': x_T,
            'y_T': y_T
        }

    
    def remove_tick_liquidities(self, tick_l: int, tick_u: int, kappa_tilde: float):
        """
        Reverse liquidity changes at tick boundaries for an LP's withdrawn position.
        """
        if tick_l in self.tick_liquidities:
            self.tick_liquidities[tick_l] -= kappa_tilde
            if abs(self.tick_liquidities[tick_l]) < 1e-9:  # Remove negligible values
                del self.tick_liquidities[tick_l]
        if tick_u in self.tick_liquidities:
            self.tick_liquidities[tick_u] += kappa_tilde
            if abs(self.tick_liquidities[tick_u]) < 1e-9:
                del self.tick_liquidities[tick_u]

        # Debugging output
        print(f"Removed tick liquidity: Tick_L {tick_l}, Tick_U {tick_u}.")

    def reset_tick_liquidities(self):
        """
        Recalculate tick liquidities from all active LP positions.
        Ensures tick_liquidities is in sync with current active positions.
        """
        self.tick_liquidities.clear()
        for position in self.active_lp_positions.values():
            self.update_tick_liquidities(position['tick_l'], position['tick_u'], position['kappa_tilde_0'])

        # Debugging output
        print("Tick liquidities recalculated from active positions.")



    def update_positions_based_on_price(self):
        """
        Update LP positions (active/inactive) based on the current price.
        Move positions between active and inactive based on their tick range and Z0.
        """
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")

        positions_to_activate = []
        positions_to_deactivate = []

        # Process inactive positions
        for position_id, position in self.inactive_lp_positions.items():
            if position['Zl'] <= self.Z0 <= position['Zu']:
                positions_to_activate.append(position_id)

        for position_id in positions_to_activate:
            self.active_lp_positions[position_id] = self.inactive_lp_positions.pop(position_id)

        # Process active positions
        for position_id, position in self.active_lp_positions.items():
            if not (position['Zl'] <= self.Z0 <= position['Zu']):
                positions_to_deactivate.append(position_id)

        for position_id in positions_to_deactivate:
            self.inactive_lp_positions[position_id] = self.active_lp_positions.pop(position_id)

        print(f"Activated {len(positions_to_activate)} positions. Deactivated {len(positions_to_deactivate)} positions.")

    def reset_fee_state(self):
        """
        Reset fee state for all active LP positions and global fee distribution mapping.
        """
        for position in self.active_lp_positions.values():
            position['fees_earned'] = 0.0
        self.fee_distribution.clear()

        # Debugging output
        print("Fee state reset. All earned fees cleared.")
                   
    def total_pool_depth(self):
        """
        Calculate the total liquidity depth κ at the current tick.
        """
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")

        # Find the current tick
        current_tick = self.price_to_tick(self.Z0)
        current_tick = current_tick - (current_tick % self.tick_spacing)

        # Sort ticks in ascending order
        sorted_ticks = sorted(self.tick_liquidities.keys())
        
        # Initialize total liquidity
        total_liquidity = 0.0

        for tick in sorted_ticks:
            # Apply net liquidity change at tick
            net_liquidity_change = self.tick_liquidities[tick]
            total_liquidity += net_liquidity_change

            if tick >= current_tick:
                # We have reached or passed the current tick
                break

        return total_liquidity
    
    def calculate_current_tick(self) -> int:
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")
        tick = self.price_to_tick(self.Z0)
        current_tick = tick - (tick % self.tick_spacing)
        return current_tick


    def distribute_fees(self, trading_volume: float):
        """
        Distribute fees to LPs based on their liquidity contributions,
        incorporating the dependency on Z_t^{1/2}.

        Formula:
        - Total Fees: Π_t = fee_rate * V_t
        - Adjusted Total Depth: 2κ Z_t^{1/2}
        - Fee Share: fee_share_i = (kappa_tilde_i / Adjusted Total Depth) * Total Fees
        """
        if self.Z0 is None:
            raise ValueError("Current pool rate Z0 is not set.")

        sqrt_Z0 = math.sqrt(self.Z0)

        total_fees = self.fee_rate * trading_volume
        self.total_fees_collected += total_fees

        # Calculate total liquidity contributed by LPs at the current price
        total_liquidity = 0.0
        for position in self.active_lp_positions.values():
            if position['Zl'] <= self.Z0 <= position['Zu']:
                total_liquidity += position['kappa_tilde_0']

        if total_liquidity == 0:
            print("No liquidity at the current price.")
            return

        adjusted_total_depth = 2 * total_liquidity * sqrt_Z0

        # Distribute fees proportionally to active LPs
        for position in self.active_lp_positions.values():
            lp_id = position['lp_id']
            if position['Zl'] <= self.Z0 <= position['Zu']:
                kappa_tilde = position['kappa_tilde_0']
                adjusted_kappa = 2 * kappa_tilde * sqrt_Z0
                fee_share = (adjusted_kappa / adjusted_total_depth) * total_fees
                position['fees_earned'] += fee_share
                self.fee_distribution[lp_id] = self.fee_distribution.get(lp_id, 0.0) + fee_share
                # Debugging output
                print(f"LP {lp_id}: Fee Share {fee_share}, Total Earned {position['fees_earned']}.")

        print(f"Distributed {total_fees} in fees among active LPs.")
        


    def get_lp_fees(self, lp_id: int) -> float:
        """
        Get the total fees earned by a specific LP across all active positions.

        Parameters:
        - lp_id: The ID of the LP.

        Returns:
        - Total fees earned by the LP.
        """
        total_fees = 0.0
        for position in self.active_lp_positions.values():
            if position['lp_id'] == lp_id:
                total_fees += position['fees_earned']
        return total_fees

    def get_active_lps(self):
        """
        Get a list of LP IDs with active liquidity in the current pool rate range.
        """
        active_lps = set()
        for position in self.active_lp_positions.values():
            if position['Zl'] <= self.Z0 <= position['Zu']:
                active_lps.add(position['lp_id'])
        return list(active_lps)
    
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

