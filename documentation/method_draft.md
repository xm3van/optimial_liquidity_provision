# Methodology

## Data Requirements
1.	Marginal exchnage rate simulator
    a. Price Data (WETH/USD Close Prices)
    b. Price Data (USDC/USD Close Prices)
    c. We calculate the Marginal Exchange rate of WETH/USDC by dividing WETH/USD/ USDC/USD
2.	Liquidity Position Data: --> https://dune.com/queries/4251434/7150389/ 
    a.	Tick Ranges (tickLower, tickUpper, currentTick)
    b.	Liquidity Depth (liquidity, liquidity_usd)
    c.	Token Balances (token0_amount, token1_amount)
    d.	Fee Rate (fee): Percentage for calculating fee income.
    e.	Timestamps: For both price and position data, to align time series and analyze historical performance.


## Implementation Plan 
1. Marginal Rate Generator:
    - fetch tokens in USD terms # DONE
    - generate exchange rate # DONE
    - generate price path of this for Z #DONE
2. Pool Class
    - Clueing marginal rate and state of the pool together 
2. Liquidity providers Class
    - Accurately capture position value α and update around position value 
    - Accurately capture fee income 
    - Accurately capture Rebalancing costs and gas fees
3. Optimizer Class
    - optimizer optimising spread 
4. Parameterisation 
    - Parameterise 
5. Backtesting 
    - backtesting 



# To-do: 
- For token class implement function to ensure data quality (i.e. filtering based on standard deviation)
- Ensure that the Marginal Exchange Rate Simulator passes the unit test (for demo purposes move on)