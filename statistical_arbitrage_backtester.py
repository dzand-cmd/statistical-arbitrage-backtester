import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

# ==========================================
# PARAMETERS
# ==========================================

TICKER_A = "KO"
TICKER_B = "PEP"

START_DATE = "2020-01-01"
END_DATE = "2025-01-01"

ENTRY_Z = 2.0
EXIT_Z = 0.5

# ==========================================
# DOWNLOAD DATA
# ==========================================

print("Downloading data...")

a = yf.download(TICKER_A,start=START_DATE,end=END_DATE,auto_adjust=True,progress=False)
b = yf.download(TICKER_B,start=START_DATE,end=END_DATE,auto_adjust=True,progress=False)

# ==========================================
# BUILD DATAFRAME
# ==========================================

df = pd.DataFrame()

df["A"] = a["Close"]
df["B"] = b["Close"]
df.dropna(inplace=True)

print("Observations:", len(df))

# ==========================================
# COINTEGRATION TEST
# ==========================================

score, pvalue, _ = coint(df["A"],df["B"])

print("\nCointegration Results")
print("---------------------")
print("P-value:", pvalue)

# ==========================================
# HEDGE RATIO ESTIMATION
# ==========================================

X = sm.add_constant(df["B"])
model = sm.OLS(df["A"],X).fit()
beta = model.params["B"]

print("Hedge Ratio (Beta):", beta)

# ==========================================
# SPREAD CONSTRUCTION
# ==========================================

df["spread"] = (df["A"] - beta * df["B"])

# ==========================================
# Z-SCORE
# ==========================================

spread_mean = df["spread"].mean()
spread_std = df["spread"].std()

df["zscore"] = (df["spread"] - spread_mean) / spread_std

# ==========================================
# SIGNAL GENERATION
# ==========================================

position = 0
positions = []

for z in df["zscore"]:
    if position == 0:
        if z > ENTRY_Z:
            position = -1
        elif z < -ENTRY_Z:
            position = 1
    else:
        if abs(z) < EXIT_Z:
            position = 0
    positions.append(position)

df["position"] = positions

# ==========================================
# STRATEGY RETURNS
# ==========================================

df["spread_return"] = (df["spread"].diff())
df["strategy_return"] = (df["position"].shift(1) * (-df["spread_return"]))
df["strategy_return"] = (df["strategy_return"].fillna(0))

# ==========================================
# EQUITY CURVE
# ==========================================

df["equity"] = (1 + df["strategy_return"]).cumprod()

# ==========================================
# PERFORMANCE METRICS
# ==========================================

returns = df["strategy_return"]
mean_return = returns.mean()
std_return = returns.std()

if std_return > 0:
    sharpe = np.sqrt(252) * mean_return / std_return
else:
    sharpe = np.nan

win_rate = (returns > 0).mean()

running_max = (df["equity"].cummax())

drawdown = (df["equity"]/ running_max - 1)

max_drawdown = drawdown.min()

print("\nStrategy Performance")
print("---------------------")
print("Sharpe Ratio:", sharpe)
print("Win Rate:", win_rate)
print("Max Drawdown:", max_drawdown)

# ==========================================
# GRAPH 1
# PRICE SERIES
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df.index,df["A"],label=TICKER_A)
plt.plot(df.index,df["B"],label=TICKER_B)
plt.title("Price Series")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ==========================================
# GRAPH 2
# SPREAD
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df.index,df["spread"])
plt.title("Cointegrated Spread")
plt.xlabel("Date")
plt.ylabel("Spread")
plt.show()

# ==========================================
# GRAPH 3
# Z-SCORE
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df.index,df["zscore"])
plt.axhline(ENTRY_Z,linestyle="--")
plt.axhline(-ENTRY_Z,linestyle="--")
plt.axhline(0,linestyle="-")
plt.title("Spread Z-Score")
plt.xlabel("Date")
plt.ylabel("Z-Score")
plt.show()

# ==========================================
# GRAPH 4
# POSITIONS
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df.index,df["position"])
plt.title("Trading Position")
plt.xlabel("Date")
plt.ylabel("Position")
plt.show()

# ==========================================
# GRAPH 5
# EQUITY CURVE
# ==========================================

plt.figure(figsize=(12,5))
plt.plot(df.index,df["equity"])
plt.title("Strategy Equity Curve")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.show()