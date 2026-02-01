# predictions.py — 1-year advanced portfolio forecast with Prophet

import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np

# ---- Your Portfolio ----
holdings = {
    "BTC": 1.5,
    "SOL": 4.48932,
    "XRP": 6.187,
    "XLM": 52.939,
    "ADA": 21.8174,
}

# CoinGecko IDs
coins = {
    "BTC": "bitcoin",
    "SOL": "solana",
    "XRP": "ripple",
    "XLM": "stellar",
    "ADA": "cardano",
}

vs_currency = "usd"
history_days = 1825  # ~5 years
forecast_days = 365

def fetch_history(cg_id, days=1825):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": str(days), "interval": "daily"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    prices = pd.DataFrame(data["prices"], columns=["ts", "price"])
    prices["ds"] = pd.to_datetime(prices["ts"], unit="ms")
    prices.rename(columns={"price": "y"}, inplace=True)
    return prices[["ds", "y"]]

def forecast_coin(prices_df, symbol, amount):
    m = Prophet(daily_seasonality=True, yearly_seasonality=True)
    m.fit(prices_df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)
    forecast["value_usd"] = forecast["yhat"] * amount

    # Plot price forecast
    fig1 = m.plot(forecast)
    plt.title(f"{symbol} Price Forecast ({forecast_days} days)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.savefig(f"forecast_{symbol}.png", dpi=150)
    plt.close(fig1)

    return forecast

# Store all forecasts
portfolio_forecasts = []

for sym, cg_id in coins.items():
    print(f"Fetching and forecasting {sym}...")
    hist = fetch_history(cg_id, history_days)
    forecast = forecast_coin(hist, sym, holdings[sym])
    forecast["symbol"] = sym
    portfolio_forecasts.append(forecast)

# Merge portfolio values
merged = portfolio_forecasts[0][["ds"]].copy()
for f in portfolio_forecasts:
    merged = merged.merge(f[["ds", "value_usd"]], on="ds", how="left", suffixes=("", f"_{f['symbol'].iloc[0]}"))

# Sum across all coins to get portfolio total
merged["portfolio_total"] = merged.drop(columns=["ds"]).sum(axis=1)

# Plot portfolio projection
plt.figure(figsize=(12,6))
plt.plot(merged["ds"], merged["portfolio_total"], label="Projected Portfolio Value")
plt.axvline(pd.Timestamp.today(), color="red", linestyle="--", label="Today")
plt.title("Portfolio Forecast for Next Year")
plt.xlabel("Date")
plt.ylabel("Total Value (USD)")
plt.legend()
plt.tight_layout()
plt.savefig("portfolio_forecast_1year.png", dpi=150)
plt.show()

print("Saved individual coin forecasts as forecast_<symbol>.png")
print("Saved portfolio projection as portfolio_forecast_1year.png")
