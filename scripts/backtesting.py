from src.models.pool import ConcentratedLiquidityMarket
from src.models.token import TokenDTO
from src.models.chain import ChainDTO
from src.models.lp import LiquidityProvider
from src.data.data_loader import DataLoader
from src.utils.backtesting import Backtest
# Initialize Pool and LP
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

lp = LiquidityProvider(initial_wealth=1e6, sigma=0.2, gamma=5e-7)

# Initialize DataLoader and Backtest
data_loader = DataLoader(data_dir=f"data/raw/", query_id=pool.dune_id)
backtest = Backtest(pool, lp, data_loader, fee_rate=0.003, eta=0.001)

# Run the backtest
backtest.run()

# Evaluate performance
performance = backtest.evaluate_performance()
print(performance)
