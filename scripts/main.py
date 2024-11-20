# Example usage in main script
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from src.models.pool import ConcentratedLiquidityMarket
from src.models.token import TokenDTO
from src.models.chain import ChainDTO
from src.models.lp import LiquidityProvider
from src.data.data_loader import DataLoader
from src.prices.marginal_exchange_rate_generator_v2 import MarginalExchangeRateGenerator

# Initialize Pool
network = ChainDTO(name="Ethereum", network_id=1)

weth = TokenDTO(
    address="0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    name="Wrapped Ether",
    symbol="WETH",
    decimals=18,
    network=network,
    coingecko_id="weth",
    token_type="collateral_token",
)
usdc = TokenDTO(
    address="0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    name="USD Coin",
    symbol="USDC",
    decimals=6,
    network=network,
    coingecko_id="usd-coin",
    token_type="stable_token",
)
pool = ConcentratedLiquidityMarket(pool_address="0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", 
               token_a=weth, 
               token_b=usdc, 
               dune_id=4251434, 
               network=network, 
               protocol="UniswapV3")

# Initialize Liquidity Provider with specific parameters
lp = LiquidityProvider(initial_wealth=1e6, sigma=0.2, gamma=5e-7)

# Load data and set pool price
data_loader = DataLoader(data_dir=f"data/raw/", query_id=pool.dune_id)
pool.retrieve_data(data_loader)

weth_usdc_rate = MarginalExchangeRateGenerator(token_a=weth, token_b=usdc, freq="1min")
path = weth_usdc_rate.generate_stochastic_path(time_horizon=1, num_steps=24*60, plot=True)


# Example of updating price and calculating metrics for the LP
for Z0 in path:
    pool.update_rate(Z0)
    delta_optimal = lp.optimize_spread(fee_rate=0.003, eta=0.001)
    pl = lp.calculate_pl(pool.current_price, delta_optimal)
    fee_income = lp.calculate_fee_income(fee_rate=0.003, delta=delta_optimal)

    print(f"Optimal Spread: {delta_optimal}, Predictable Loss: {pl}, Fee Income: {fee_income}")
    break