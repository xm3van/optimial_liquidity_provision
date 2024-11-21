import unittest
from src.models.pool import ConcentratedLiquidityMarket
from src.models.token import TokenDTO
from src.models.chain import ChainDTO
import math


class TestConcentratedLiquidityMarket(unittest.TestCase):
    def setUp(self):
        """
        Set up test environment with common parameters and a sample pool instance.
        """
        # Mock TokenDTOs and ChainDTO
        self.chain = ChainDTO(name="Ethereum", network_id=1)
        self.token_a = TokenDTO(
            address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
            name="Wrapped Ether",
            symbol="WETH",
            decimals=18,
            network=self.chain,
            coingecko_id="weth",
            token_type="collateral_token",
        )
        self.token_b = TokenDTO(
            address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
            name="USD Coin",
            symbol="USDC",
            decimals=6,
            network=self.chain,
            coingecko_id="usd-coin",
            token_type="stable_token",
        )

        # Pool parameters
        self.pool_address = "0x123"
        self.dune_id = 12345
        self.protocol = "UniswapV3"

        # Initialize pool instance
        self.pool = ConcentratedLiquidityMarket(
            pool_address=self.pool_address,
            token_a=self.token_a,
            token_b=self.token_b,
            dune_id=self.dune_id,
            network=self.chain,
            protocol=self.protocol,
            Z0=100  # Initial rate
        )

        # Initialize ticks
        self.pool.initialize_ticks(
            min_tick=-887272,
            max_tick=887272,
            tick_spacing=60  # Adjust tick spacing as needed
        )

    def test_calculate_current_tick(self):
        """
        Test the calculation of the current tick based on the pool's current price.
        """
        # Set the pool's current price Z0
        self.pool.update_rate(100.0)

        # Calculate the current tick
        current_tick = self.pool.calculate_current_tick()

        # Expected tick (calculated manually)
        tick = int(math.log(100.0) / math.log(1.0001))
        expected_tick = tick - (tick % self.pool.tick_spacing)

        print(f"Calculated tick: {current_tick}, Expected tick: {expected_tick}")

        # Assert that the calculated tick matches the expected tick
        self.assertEqual(current_tick, expected_tick, "Calculated current tick is incorrect.")

    def test_deposit_and_withdraw(self):
        """
        Test deposit and withdraw functionality.
        """
        initial_wealth = 100
        lp_id = 1
        delta_l = 0.05
        delta_u = 0.05

        # Deposit liquidity
        x0, y0, _ = self.pool.deposit(lp_id=lp_id, wealth=initial_wealth, delta_l=delta_l, delta_u=delta_u)
        position_id = 1  # Assuming position IDs start from 1
        position = self.pool.active_lp_positions.get(position_id) or self.pool.inactive_lp_positions.get(position_id)

        self.assertIsNotNone(position, "LP position should exist after deposit.")
        self.assertAlmostEqual(position['x0'], x0, places=6, msg="x0 should match recorded value.")
        self.assertAlmostEqual(position['y0'], y0, places=6, msg="y0 should match recorded value.")

        # Withdraw liquidity
        terminal_rate = 105
        results = self.pool.withdraw(position_id=position_id, terminal_rate=terminal_rate)
        self.assertNotIn(position_id, self.pool.active_lp_positions, "Position should be removed from active positions after withdrawal.")
        self.assertNotIn(position_id, self.pool.inactive_lp_positions, "Position should be removed from inactive positions after withdrawal.")
        self.assertIn('alpha_T', results, "Results should include terminal wealth.")
        self.assertGreater(results['alpha_T'], 0, "Terminal wealth should be positive.")

    def test_fee_distribution(self):
        """
        Test fee distribution among multiple LPs.
        """
        # Set initial pool rate
        self.pool.update_rate(100)

        # Deposit multiple LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.05, delta_u=0.05)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.1, delta_u=0.1)

        # Verify active LPs
        active_lps = self.pool.get_active_lps()
        print(f"Active LPs before fee distribution: {active_lps}")

        # Distribute fees
        trading_volume = 100000
        self.pool.distribute_fees(trading_volume=trading_volume)

        # Retrieve and print fee details
        lp1_fees = self.pool.get_lp_fees(1)
        lp2_fees = self.pool.get_lp_fees(2)
        total_fees = lp1_fees + lp2_fees

        print(f"LP1 Fees: {lp1_fees}")
        print(f"LP2 Fees: {lp2_fees}")
        print(f"Total Fees Collected: {self.pool.total_fees_collected}")

        # Verify fee distribution
        self.assertGreater(lp1_fees, 0, "LP1 should receive a share of the fees.")
        self.assertGreater(lp2_fees, 0, "LP2 should receive a share of the fees.")
        self.assertAlmostEqual(total_fees, self.pool.total_fees_collected, places=2, msg="Total fees should match collected fees.")

   
    def test_total_pool_depth(self):
        """
        Test total_pool_depth calculates active liquidity correctly.
        """
        # Set the pool's current price Z0 to ensure LPs are active
        self.pool.update_rate(100.0)  # Set Z0 to 100

        # Deposit multiple LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.05, delta_u=0.05)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.1, delta_u=0.1)

        # Calculate total depth using the pool's method
        total_depth = self.pool.total_pool_depth()

        # Manually calculate expected total liquidity at the current tick
        current_tick = self.pool.calculate_current_tick()

        # Debugging output
        print(f"Current Tick: {current_tick}")


    def test_active_positions(self):
        """
        Test positions are correctly marked as active or inactive.
        """
        # Deposit an LP
        lp_id = 1
        self.pool.deposit(lp_id=lp_id, wealth=100, delta_l=0.05, delta_u=0.05)
        position_id = 1  # Assuming position IDs start from 1

        # Verify initial active status
        active_positions = self.pool.get_active_lps()
        self.assertIn(position_id, active_positions, "LP1's position should be active initially.")

        # Update rate outside active range
        self.pool.update_rate(150)
        active_positions = self.pool.get_active_lps()
        self.assertNotIn(position_id, active_positions, "LP1's position should be inactive when rate is outside range.")

    
    def test_tick_alignment(self):
        """
        Test that LP deposits align tick boundaries correctly.
        """
        # Set the pool's current price Z0
        self.pool.update_rate(100.0)

        # LP deposits with specified delta_l and delta_u
        lp_id = 1
        wealth = 100
        delta_l = 0.05
        delta_u = 0.05

        x0, y0, _ = self.pool.deposit(lp_id=lp_id, wealth=wealth, delta_l=delta_l, delta_u=delta_u)

        # Retrieve the position
        position = self.pool.active_lp_positions.get(1) or self.pool.inactive_lp_positions.get(1)

        # Manually calculate expected tick boundaries and aligned prices
        sqrt_Z0 = math.sqrt(self.pool.Z0)
        sqrt_Zl = sqrt_Z0 * (1 - delta_l / 2)
        sqrt_Zu = sqrt_Z0 / (1 - delta_u / 2)
        Zl = sqrt_Zl ** 2
        Zu = sqrt_Zu ** 2

        # Convert prices to ticks and align to tick boundaries
        tick_l = self.pool.price_to_tick(Zl)
        tick_u = self.pool.price_to_tick(Zu)

        tick_l_aligned = tick_l - (tick_l % self.pool.tick_spacing)
        tick_u_aligned = tick_u + (self.pool.tick_spacing - (tick_u % self.pool.tick_spacing)) if tick_u % self.pool.tick_spacing != 0 else tick_u

        # Convert aligned ticks back to prices
        expected_Zl = self.pool.tick_to_price(tick_l_aligned)
        expected_Zu = self.pool.tick_to_price(tick_u_aligned)

        # Assert that Zl and Zu in the position match the expected aligned prices
        self.assertAlmostEqual(position['Zl'], expected_Zl, places=6, msg="Zl should align with tick boundaries.")
        self.assertAlmostEqual(position['Zu'], expected_Zu, places=6, msg="Zu should align with tick boundaries.")

    def test_overlapping_lp_ranges(self):
        """
        Test fee distribution for overlapping LP ranges.
        """
        # Deposit overlapping LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.1, delta_u=0.1)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.15, delta_u=0.15)

        # Distribute fees
        self.pool.distribute_fees(trading_volume=100000)

        # Verify that both LPs received fees
        lp1_fees = self.pool.get_lp_fees(1)
        lp2_fees = self.pool.get_lp_fees(2)

        self.assertGreater(lp1_fees, 0, "LP1 should receive fees.")
        self.assertGreater(lp2_fees, 0, "LP2 should receive fees.")
        self.assertAlmostEqual(lp1_fees + lp2_fees, self.pool.total_fees_collected, places=6, msg="Fee distribution should be accurate.")

    def test_update_rate_behavior(self):
        """
        Test the behavior of the pool when the rate is updated.
        """
        # Deposit LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.05, delta_u=0.05)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.1, delta_u=0.1)

        # Update rate to activate/deactivate LPs
        self.pool.update_rate(105)
        active_lps = self.pool.get_active_lps()

        # Verify activation/deactivation
        self.assertIn(1, active_lps, "LP1 should be active.")
        self.assertIn(2, active_lps, "LP2 should be active.")

        # Update rate outside both ranges
        self.pool.update_rate(150)
        active_lps = self.pool.get_active_lps()
        self.assertEqual(len(active_lps), 0, "All LPs should be inactive.")

    def test_behavior_near_tick_boundaries(self):
        """
        Test deposits and withdrawals near tick boundaries.
        """
        # Set Z0 near a tick boundary
        tick_price = self.pool.tick_to_price(45000)
        self.pool.update_rate(tick_price)

        # Deposit LP near boundary
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.001, delta_u=0.001)
        position = self.pool.active_lp_positions[1]
        self.assertTrue(position['tick_l'] <= 45000 <= position['tick_u'], "Position should include the boundary tick.")

        # Withdraw and verify
        results = self.pool.withdraw(position_id=1, terminal_rate=tick_price)
        self.assertIn('alpha_T', results, "Terminal wealth should be calculated.")


if __name__ == "__main__":
    unittest.main()
