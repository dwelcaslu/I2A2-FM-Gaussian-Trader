# I2A2-FM-Naive-Trader

This Trading strategy was based on the genetic algorithm approach.

All the source code with the main scripts are in ./src:
- indicators: Performs some market indicator calculations
- stockmarket: Implements the MarkerOperator class.
This class is responsible for simulating the market operations.
- arena: Implements the TraderArena class, with the genetic algorithm simulation.

All the notebooks with the analysis are in ./notebooks:
- nbk01_data_exploration: data exploration
- nbk01_naive_bayes: basic test of the naive bayes algorithm
- nbk02_market_operator: basic test of the MarkerOperator class.
- nbk02_trader_arena: basic test of the TraderArena class.
- nbk03_trader_test01_all: Implementation of the trading algorithm
considering the following indicators: macd, signal, histogram and williams_r.
- nbk03_trader_test02_williams: Implementation of the trading algorithm
considering only the williams_r indicator.
- nbk03_trader_test03_macd: Implementation of the trading algorithm
considering the following indicators: macd, signal and histogram.
