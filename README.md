# Optimial Liquidity Provision for CFM


```
project-root/
│
├── data/                          # Data folder for input files
│   ├── raw/                       # Historical Pools Data (e.g., .csv)
│   ├── ohlc_data/                 # Token OHLC Data
│   └── processed/                 # n/a for now
│
├── src/                           # Source code
│   ├── __init__.py                # Makes src a package for easier imports
│   
│   ├── data/                      # Data handling and processing
│   │   ├── __init__.py
│   │   ├── data_loader.py         # DataLoader class for loading and fetching data
│   │   └── data_processor.py      # n/a for now
│   
│   ├── models/                    # Core classes for the model and calculations
│   │   ├── __init__.py
│   │   ├── pool.py                # Pool class with main methods for the liquidity pool
│   │   ├── token.py               # Token class with methods for token within liquidity pool 
│   │   ├── chain.py               # Chain Class identifying the relevant network
│   │   ├── optimizer.py           # Optimizer class for optimal spread calculations
│   │   └── lp.py                  # PLCalculator + FeeCalculator
│   
│   ├── utils/                     # Utility functions and configuration
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration for model parameters
│   │   └── helpers.py             # Helper functions, e.g., for logging, data validation
│   
├── scripts/                       # Scripts to run the project
│   ├── main.py                    # Main script to initialize and run the Pool simulation
│   └── backtest.py                # Backtesting script to evaluate performance
│
├── notebooks/                     # Jupyter notebooks for exploration and testing
│   └── analysis.ipynb             # Notebook for EDA and model exploration
│
├── tests/                         # Unit tests for the src code
│   ├── __init__.py
│   ├── test_data_loader.py        # Tests for data loading and processing
│   ├── test_pool.py               # Tests for the Pool class
│   ├── test_optimizer.py          # Tests for the Optimizer class
│   ├── test_fee_calculator.py     # Tests for the FeeCalculator class
│   └── test_pl_calculator.py      # Tests for the PLCalculator class
│
├── .gitignore                     # Git ignore file (e.g., ignore data/processed and .ipynb_checkpoints)
├── README.md                      # Project overview and instructions
├── requirements.txt               # Dependencies for the project
└── setup.py                       # Optional: Setup script to make src installable as a package

``