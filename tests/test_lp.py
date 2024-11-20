import unittest
from src.models.pool import ConcentratedLiquidityMarket
from src.models.lp import LiquidityProvider
from src.models.token import TokenDTO
from src.models.chain import ChainDTO


class TestLiquidityProvider(unittest.TestCase):
    def setUp(self):
        """
        Set up the pool and LPs for testing.
        """
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
        )

        # Initialize LPs
        self.lp1 = LiquidityProvider(lp_id=1, initial_wealth=100, delta_u=0.05, delta_l=0.05)
        self.lp2 = LiquidityProvider(lp_id=2, initial_wealth=200, delta_u=0.1, delta_l=0.1)

    def test_deposit_liquidity(self):
        """
        Test LP deposits liquidity into the pool.
        """
        self.lp1.deposit_liquidity(self.pool)
        self.assertIn(1, self.pool.active_lp_positions, "LP1's position should be registered in the pool.")
        self.assertEqual(len(self.lp1.active_positions), 1, "LP1 should have one active position.")

    def test_withdraw_liquidity(self):
        """
        Test LP withdraws liquidity from the pool.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = next(iter(self.lp1.active_positions.keys()))
        self.pool.update_rate(105)
        results = self.lp1.withdraw_liquidity(self.pool, position_id=position_id, terminal_rate=105)
        self.assertEqual(len(self.lp1.active_positions), 0, "LP1 should have no active positions after withdrawal.")
        self.assertIn(position_id, [p['position_id'] for p in self.lp1.historical_positions],
                      "Withdrawn position should be moved to historical positions.")

    def test_query_fees(self):
        """
        Test LP queries fee income from the pool.
        """
        self.lp1.deposit_liquidity(self.pool)
        self.pool.distribute_fees(trading_volume=100000)
        self.lp1.query_fees(self.pool, position_id=1)
        self.assertGreater(self.lp1.fee_income, 0, "LP1 should earn fees after fee distribution.")

    def test_is_within_active_range(self):
        """
        Test if LP is correctly marked as active or inactive based on the pool rate.
        """
        self.pool.Zl = 90  # Ensure the pool's lower tick is set
        self.pool.Zu = 110  # Ensure the pool's upper tick is set
        self.lp1.deposit_liquidity(self.pool)

        self.assertTrue(self.lp1.is_within_active_range(self.pool), "LP1 should be active at the initial pool rate.")
        self.pool.update_rate(150)  # Move rate outside LP's range
        self.assertFalse(self.lp1.is_within_active_range(self.pool), "LP1 should be inactive after rate moves out of range.")


    def test_get_position_details(self):
        """
        Test retrieving position details of the LP.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = next(iter(self.lp1.active_positions.keys()))
        position_details = self.lp1.get_position_details(position_id)
        self.assertIsInstance(position_details, dict, "Position details should be a dictionary.")
        self.assertIn('x0', position_details, "Position details should include initial asset x0.")
        self.assertIn('Zl', position_details, "Position details should include the lower range Zl.")

    def test_get_terminal_wealth(self):
        """
        Test retrieving terminal wealth after withdrawal.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = next(iter(self.lp1.active_positions.keys()))
        self.pool.update_rate(105)
        self.lp1.withdraw_liquidity(self.pool, position_id=position_id, terminal_rate=105)
        terminal_wealth = self.lp1.get_total_wealth()
        self.assertGreater(terminal_wealth, 0, "Terminal wealth should be positive after withdrawal.")

    def test_invalid_deposit(self):
        """
        Test invalid deposit parameters.
        """
        with self.assertRaises(ValueError):
            LiquidityProvider(lp_id=3, initial_wealth=0, delta_u=0.05, delta_l=0.05)

        with self.assertRaises(ValueError):
            LiquidityProvider(lp_id=4, initial_wealth=100, delta_u=1.5, delta_l=0.05)

    def test_invalid_withdraw(self):
        """
        Test withdrawing without an active position.
        """
        with self.assertRaises(ValueError):
            self.lp1.withdraw_liquidity(self.pool, position_id=1, terminal_rate=105)

    def test_fee_distribution(self):
        """
        Test fee distribution among multiple LPs.
        """
        self.lp1.deposit_liquidity(self.pool)
        self.lp2.deposit_liquidity(self.pool)
        self.pool.distribute_fees(trading_volume=100000)

        lp1_fees = self.pool.get_lp_fees(self.lp1.lp_id)
        lp2_fees = self.pool.get_lp_fees(self.lp2.lp_id)
        total_fees = lp1_fees + lp2_fees

        self.assertGreater(lp1_fees, 0, "LP1 should receive a share of the fees.")
        self.assertGreater(lp2_fees, 0, "LP2 should receive a share of the fees.")
        self.assertAlmostEqual(total_fees, self.pool.total_fees_collected, places=2,
                               msg="Sum of LP fees should equal total pool fees collected.")


if __name__ == '__main__':
    unittest.main()
