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
        self.pool = ConcentratedLiquidityMarket(
            pool_address="0x123",
            token_a=self.token_a,
            token_b=self.token_b,
            dune_id=12345,
            network=self.chain,
            protocol="UniswapV3",
            Z0=100,  # Initial rate
        )

        # LPs
        self.lp1 = LiquidityProvider(lp_id=1, initial_wealth=100, delta_u=0.05, delta_l=0.05)
        self.lp2 = LiquidityProvider(lp_id=2, initial_wealth=200, delta_u=0.1, delta_l=0.1)

    def test_deposit_liquidity(self):
        """
        Test LP deposits liquidity into the pool.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = list(self.lp1.active_positions.keys())[0]
        self.assertIn(position_id, self.pool.active_lp_positions, "LP1's position should be registered in the pool.")
        self.assertEqual(len(self.lp1.active_positions), 1, "LP1 should have one active position.")
        self.assertEqual(self.lp1.total_wealth, 0, "Total wealth should be reduced by the deposited amount.")

    def test_withdraw_liquidity(self):
        """
        Test LP withdraws liquidity from the pool.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = list(self.lp1.active_positions.keys())[0]

        # Withdraw liquidity
        self.pool.update_rate(105)
        results = self.lp1.withdraw_liquidity(self.pool, position_id=position_id, terminal_rate=105)
        self.assertEqual(len(self.lp1.active_positions), 0, "LP1 should have no active positions after withdrawal.")
        self.assertIn(position_id, [p['position_id'] for p in self.lp1.historical_positions],
                      "Withdrawn position should be moved to historical positions.")
        self.assertGreater(results['alpha_T'], 0, "Terminal wealth should be positive.")

    def test_fee_tracking(self):
        """
        Test accurate fee tracking for multiple positions.
        """
        self.lp1.deposit_liquidity(self.pool)
        self.lp2.deposit_liquidity(self.pool)

        # Distribute fees
        self.pool.distribute_fees(trading_volume=100000)

        # Query fees
        for lp in [self.lp1, self.lp2]:
            for position_id in lp.active_positions.keys():
                lp.query_fees(self.pool, position_id)

        self.assertGreater(self.lp1.total_fee_income, 0, "LP1 should have earned fees.")
        self.assertGreater(self.lp2.total_fee_income, 0, "LP2 should have earned fees.")

    def test_position_active_range(self):
        """
        Test positions are correctly marked as active or inactive.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = next(iter(self.lp1.active_positions.keys()))

        # Ensure the position is active initially
        self.assertIn(position_id, self.lp1.active_positions, "Position should be active initially.")

        # Update rate to move outside the active range
        self.pool.update_rate(150)
        self.lp1.update_positions_based_on_price(self.pool)

        # Ensure the position is now inactive
        self.assertNotIn(position_id, self.lp1.active_positions, "Position should be inactive when rate moves out of range.")

    def test_invalid_operations(self):
        """
        Test invalid operations such as over-depositing or withdrawing non-existent positions.
        """
        with self.assertRaises(ValueError):
            self.lp1.deposit_liquidity(self.pool, wealth=200)  # Wealth exceeds total wealth

        with self.assertRaises(ValueError):
            self.lp1.withdraw_liquidity(self.pool, position_id=999, terminal_rate=100)  # Invalid position ID

    def test_terminal_wealth_update(self):
        """
        Test LP's total wealth after deposit, fees, and withdrawal.
        """
        self.lp1.deposit_liquidity(self.pool)
        position_id = list(self.lp1.active_positions.keys())[0]

        # Simulate trading and fee distribution
        self.pool.distribute_fees(trading_volume=100000)
        self.lp1.query_fees(self.pool, position_id)

        # Withdraw liquidity
        terminal_rate = 105
        self.lp1.withdraw_liquidity(self.pool, position_id=position_id, terminal_rate=terminal_rate)

        # Verify wealth update
        self.assertGreater(self.lp1.total_wealth, 100, "LP's total wealth should increase after earning fees and withdrawing.")

    def test_fee_distribution_accuracy(self):
        """
        Test fee distribution is correctly proportional to liquidity provided.
        """
        self.lp1.deposit_liquidity(self.pool)
        self.lp2.deposit_liquidity(self.pool)

        # Distribute fees
        trading_volume = 100000
        self.pool.distribute_fees(trading_volume=trading_volume)

        # Retrieve fees
        lp1_fees = self.pool.get_lp_fees(self.lp1.lp_id)
        lp2_fees = self.pool.get_lp_fees(self.lp2.lp_id)
        total_fees = lp1_fees + lp2_fees

        self.assertAlmostEqual(total_fees, self.pool.total_fees_collected, places=2,
                               msg="Sum of LP fees should equal total pool fees collected.")
        

    def test_wealth_after_withdrawal(self):
        """
        Test LP's total wealth after a deposit, fee distribution, and withdrawal.
        """
        self.lp1.deposit_liquidity(self.pool, wealth=50)
        self.pool.distribute_fees(trading_volume=100000)

        position_id = next(iter(self.lp1.active_positions.keys()))
        self.lp1.withdraw_liquidity(self.pool, position_id=position_id, terminal_rate=105)

        # Check that wealth increased by the terminal value and fees
        self.assertGreater(self.lp1.get_total_wealth(), 50, "Total wealth should increase after withdrawal and fee earnings.")

    def test_active_positions_in_range(self):
        """
        Test retrieving active positions within the current tick range.
        """
        self.lp1.deposit_liquidity(self.pool, wealth=100, delta_l=0.05, delta_u=0.05)

        # Update pool rate to ensure the position is active
        self.pool.update_rate(100.0)
        active_positions = self.lp1.get_active_positions_in_range(self.pool)

        self.assertGreater(len(active_positions), 0, "There should be at least one active position in the current range.")
        for position_id in active_positions:
            position = self.lp1.active_positions[position_id]
            self.assertTrue(position['Zl'] <= self.pool.Z0 <= position['Zu'], "Active position should be within the current range.")


    def test_invalid_initialization(self):
        """
        Test initializing an LP with invalid parameters.
        """
        with self.assertRaises(ValueError, msg="Should raise ValueError for negative wealth."):
            LiquidityProvider(lp_id=1, initial_wealth=-100, delta_u=0.05, delta_l=0.05)

        with self.assertRaises(ValueError, msg="Should raise ValueError for invalid spreads."):
            LiquidityProvider(lp_id=2, initial_wealth=100, delta_u=1.5, delta_l=0.05)



if __name__ == '__main__':
    unittest.main()