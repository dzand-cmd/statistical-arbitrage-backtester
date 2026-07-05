# Statistical Arbitrage Backtester

**Author:** Dzandu Selorm (dzand-cmd)  
**Project Type:** Quantitative Trading  
**Language:** Python  
**Status:** Complete

---

## Overview

This project implements a statistical arbitrage backtesting engine that identifies tradable asset pairs using correlation and cointegration analysis. It generates mean-reversion signals based on z-score spreads and evaluates trading strategies under historical market data.

The objective is to test whether statistically linked assets exhibit exploitable short-term price divergences and to evaluate the robustness of pair-trading strategies using standard performance metrics.  


## Project Structure  

statistical-arbitrage-backtester/
│
├── statistical_arbitrage_backtester.py   # Core backtesting engine
├── README.md                             # Project documentation


## Methodology  
- Select asset pairs using cointegration testing
- Estimate hedge ratio using linear regression
- Construct spread and normalize via rolling statistics
- Generate mean-reversion trading signals using z-score thresholds
- Backtest strategy using lagged execution assumption


## Backtesting Framework  
- Position enters when spread deviates beyond threshold
- Position exits upon mean reversion
- Returns computed using lagged signal application
- Includes transaction-aware PnL approximation (simplified)


## Assumptions  
- Spread between cointegrated assets is stationary
- Relationship between assets is stable over backtest window
- Execution occurs at close prices with no slippage modeled
- Transaction costs are ignored (baseline version)


## Outputs  
- Cointegration test statistics
- Hedge ratio estimates
- Spread time series and z-score evolution
- Strategy returns and Sharpe ratio
- Drawdown and performance curves


## How to run  
git clone https://github.com/your-username/stat-arb-backtester.git
cd stat-arb-backtester
pip install numpy pandas statsmodels matplotlib
python statistical_arbitrage_backtester.py







