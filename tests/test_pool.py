# import unittest
# from src.models.pool import ConcentratedLiquidityMarket
# from src.models.token import TokenDTO
# from src.models.chain import ChainDTO
# import math

# class TestConcentratedLiquidityMarket(unittest.TestCase):
#     def setUp(self):
#         """
#         Set up test environment with common parameters and a sample pool instance.
#         """
#         # Mock TokenDTOs and ChainDTO
#         self.chain = ChainDTO(name="Ethereum", network_id=1)
#         self.token_a = TokenDTO(
#             address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
#             name="Wrapped Ether",
#             symbol="WETH",
#             decimals=18,
#             network=self.chain,
#             coingecko_id="weth",
#             token_type="collateral_token",
#         )
#         self.token_b = TokenDTO(
#             address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
#             name="USD Coin",
#             symbol="USDC",
#             decimals=6,
#             network=self.chain,
#             coingecko_id="usd-coin",
#             token_type="stable_token",
#         )

#         # Pool parameters
#         self.pool_address = "0x123"
#         self.dune_id = 12345
#         self.protocol = "UniswapV3"

#         # Initialize pool instance
#         self.pool = ConcentratedLiquidityMarket(
#             pool_address=self.pool_address,
#             token_a=self.token_a,
#             token_b=self.token_b,
#             dune_id=self.dune_id,
#             network=self.chain,
#             protocol=self.protocol,
#             Z0=100,  # Initial rate
#             Zl=90,  # Lower tick
#             Zu=110  # Upper tick
#         )

#     def test_calculate_initial_holdings_case1(self):
#         """
#         Test calculate_initial_holdings for Case 1: Z0 ≤ Zl (only asset Y).
#         """
#         self.pool.Z0 = 80  # Set Z0 below Zl
#         initial_wealth = 100

#         x0, y0 = self.pool.calculate_initial_holdings(initial_wealth)

#         # Compute expected results
#         expected_y0 = initial_wealth / self.pool.Z0
#         expected_kappa = expected_y0 / (
#             1 / math.sqrt(self.pool.Zl) - 1 / math.sqrt(self.pool.Zu)
#         )

#         # Verify results
#         self.assertAlmostEqual(x0, 0, msg="x0 should be 0 for Case 1.")
#         self.assertAlmostEqual(y0, expected_y0, places=6, msg="y0 calculation is incorrect for Case 1.")
#         self.assertAlmostEqual(self.pool.kappa_tilde_0, expected_kappa, places=6, msg="κ̃0 calculation is incorrect for Case 1.")

#     def test_calculate_initial_holdings_case2(self):
#         """
#         Test calculate_initial_holdings for Case 2: Z0 > Zu (only asset X).
#         """
#         self.pool.Z0 = 120  # Set Z0 above Zu
#         initial_wealth = 100

#         x0, y0 = self.pool.calculate_initial_holdings(initial_wealth)

#         # Compute expected results
#         expected_x0 = initial_wealth
#         expected_kappa = expected_x0 / (math.sqrt(self.pool.Zu) - math.sqrt(self.pool.Zl))

#         # Verify results
#         self.assertAlmostEqual(x0, expected_x0, places=6, msg="x0 calculation is incorrect for Case 2.")
#         self.assertAlmostEqual(y0, 0, msg="y0 should be 0 for Case 2.")
#         self.assertAlmostEqual(self.pool.kappa_tilde_0, expected_kappa, places=6, msg="κ̃0 calculation is incorrect for Case 2.")

#     def test_calculate_initial_holdings_case3(self):
#         """
#         Test calculate_initial_holdings for Case 3: Zl < Z0 ≤ Zu (both assets).
#         """
#         self.pool.Z0 = 100  # Set Z0 within range
#         initial_wealth = 100

#         x0, y0 = self.pool.calculate_initial_holdings(initial_wealth)

#         # Compute expected results
#         sqrt_Z0 = math.sqrt(self.pool.Z0)
#         sqrt_Zl = math.sqrt(self.pool.Zl)
#         sqrt_Zu = math.sqrt(self.pool.Zu)
#         denominator = (sqrt_Z0 - sqrt_Zl) + (1 / sqrt_Z0 - 1 / sqrt_Zu) * self.pool.Z0
#         expected_kappa = initial_wealth / denominator
#         expected_x0 = expected_kappa * (sqrt_Z0 - sqrt_Zl)
#         expected_y0 = expected_kappa * (1 / sqrt_Z0 - 1 / sqrt_Zu)

#         # Verify results
#         self.assertAlmostEqual(x0, expected_x0, places=6, msg="x0 calculation is incorrect for Case 3.")
#         self.assertAlmostEqual(y0, expected_y0, places=6, msg="y0 calculation is incorrect for Case 3.")
#         self.assertAlmostEqual(self.pool.kappa_tilde_0, expected_kappa, places=6, msg="κ̃0 calculation is incorrect for Case 3.")

#     def test_calculate_terminal_value_case1(self):
#         """
#         Test calculate_terminal_value for Case 1: Z_T ≤ Zl (only asset Y).
#         """
#         self.pool.Z0 = 80
#         terminal_rate = 85  # Z_T ≤ Zl
#         initial_wealth = 100
#         self.pool.calculate_initial_holdings(initial_wealth)

#         alpha_T = self.pool.calculate_terminal_value(terminal_rate)

#         # Compute expected results
#         expected_x_T = 0
#         expected_y_T = self.pool.kappa_tilde_0 * (
#             1 / math.sqrt(self.pool.Zl) - 1 / math.sqrt(self.pool.Zu)
#         )
#         expected_alpha_T = expected_x_T + expected_y_T * terminal_rate

#         # Verify results
#         self.assertAlmostEqual(self.pool.x_T, expected_x_T, places=6, msg="x_T should be 0 for Case 1.")
#         self.assertAlmostEqual(self.pool.y_T, expected_y_T, places=6, msg="y_T calculation is incorrect for Case 1.")
#         self.assertAlmostEqual(alpha_T, expected_alpha_T, places=6, msg="Terminal value calculation is incorrect for Case 1.")

#     def test_calculate_terminal_value_case3(self):
#         """
#         Test calculate_terminal_value for Case 3: Z_T > Zu (only asset X).
#         """
#         self.pool.Z0 = 100
#         terminal_rate = 120  # Z_T > Zu
#         initial_wealth = 100
#         self.pool.calculate_initial_holdings(initial_wealth)

#         alpha_T = self.pool.calculate_terminal_value(terminal_rate)

#         # Compute expected results
#         expected_x_T = self.pool.kappa_tilde_0 * (
#             math.sqrt(self.pool.Zu) - math.sqrt(self.pool.Zl)
#         )
#         expected_y_T = 0
#         expected_alpha_T = expected_x_T + expected_y_T * terminal_rate

#         # Verify results
#         self.assertAlmostEqual(self.pool.x_T, expected_x_T, places=6, msg="x_T calculation is incorrect for Case 3.")
#         self.assertAlmostEqual(self.pool.y_T, expected_y_T, places=6, msg="y_T should be 0 for Case 3.")
#         self.assertAlmostEqual(alpha_T, expected_alpha_T, places=6, msg="Terminal value calculation is incorrect for Case 3.")

# if __name__ == "__main__":
#     unittest.main()

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
            Z0=100,  # Initial rate
            Zl=90,  # Lower tick
            Zu=110,  # Upper tick
        )

    def test_deposit_and_withdraw(self):
        """
        Test deposit and withdraw functionality.
        """
        initial_wealth = 100
        lp_id = 1
        position_id =1
        delta_l = 0.05
        delta_u = 0.05

        # Deposit liquidity
        x0, y0 = self.pool.deposit(lp_id=lp_id, wealth=initial_wealth, delta_l=delta_l, delta_u=delta_u)
        self.assertIn(lp_id, self.pool.active_lp_positions, "LP should be added to the pool's positions after deposit.")
        self.assertEqual(self.pool.active_lp_positions[lp_id]['x0'], x0, "Initial x0 should match the pool's record.")
        self.assertEqual(self.pool.active_lp_positions[lp_id]['y0'], y0, "Initial y0 should match the pool's record.")

        # Withdraw liquidity
        terminal_rate = 105
        results = self.pool.withdraw(position_id=position_id, terminal_rate=terminal_rate)
        self.assertNotIn(lp_id, self.pool.active_lp_positions, "LP should be removed from the pool's positions after withdrawal.")
        self.assertIn('alpha_T', results, "Withdrawal results should include terminal wealth.")

    def test_fee_distribution(self):
        """
        Test fee distribution among multiple LPs.
        """
        # Deposit multiple LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.05, delta_u=0.05)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.1, delta_u=0.1)

        # Distribute fees
        trading_volume = 100000
        self.pool.distribute_fees(trading_volume=trading_volume)

        # Verify fee distribution
        lp1_fees = self.pool.get_lp_fees(1)
        lp2_fees = self.pool.get_lp_fees(2)
        total_fees = lp1_fees + lp2_fees

        self.assertGreater(lp1_fees, 0, "LP1 should receive a share of the fees.")
        self.assertGreater(lp2_fees, 0, "LP2 should receive a share of the fees.")
        self.assertAlmostEqual(total_fees, self.pool.total_fees_collected, places=2, msg="Total fees should match the pool's collected fees.")

    def test_total_pool_depth(self):
        """
        Test total_pool_depth calculates active liquidity correctly.
        """
        # Deposit multiple LPs
        self.pool.deposit(lp_id=1, wealth=100, delta_l=0.05, delta_u=0.05)
        self.pool.deposit(lp_id=2, wealth=200, delta_l=0.1, delta_u=0.1)

        # Verify total pool depth
        total_depth = self.pool.total_pool_depth()
        expected_depth = sum(position['kappa_tilde_0'] for position in self.pool.active_lp_positions.values())
        self.assertAlmostEqual(total_depth, expected_depth, places=6, msg="Total pool depth calculation is incorrect.")

    def test_active_positions(self):
        """
        Test positions are correctly marked as active or inactive.
        """
        # Deposit an LP
        lp_id = 1
        self.pool.deposit(lp_id=lp_id, wealth=100, delta_l=0.05, delta_u=0.05)

        # Verify active status at initial rate
        self.assertTrue(self.pool.Zl <= self.pool.Z0 <= self.pool.Zu, "Initial pool rate should be within the active range.")
        self.assertEqual(self.pool.active_lp_positions[lp_id]['Zl'], self.pool.Zl, "LP's lower bound should match pool's Zl.")
        self.assertEqual(self.pool.active_lp_positions[lp_id]['Zu'], self.pool.Zu, "LP's upper bound should match pool's Zu.")

        # Update rate to move outside range
        self.pool.update_rate(150)
        self.assertFalse(self.pool.Zl <= self.pool.Z0 <= self.pool.Zu, "Pool rate should now be outside the active range.")

if __name__ == "__main__":
    unittest.main()
