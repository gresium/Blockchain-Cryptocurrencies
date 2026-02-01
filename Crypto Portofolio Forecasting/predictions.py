# predictions.py
# Next-day forecast per coin + projected total portfolio value
# Uses CoinGecko (no key). Simple RandomForest model with lag/rolling features.

import requests, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ---- YOUR PORTFOLIO (edit amounts as needed) ----
holdings = {
    "BTC": 0.062134,
    "SOL": 2.48932,
    "XRP": 106.187,
    "XLM": 142.939,
    "ADA": 21.8174,
}

# Coin tickers -> CoinGecko IDs
coins = {
    "BTC": "bitcoin",
    "SOL": "solana",
    "XRP": "ripple",
    "XLM": "stellar",
    "ADA": "cardano",
}

vs_currency = "usd"
lookback_days = 120
lags = [1, 2, 3, 7, 14]
rolls = [3, 7, 14]

def fetch_history(cg_id: str, days: int = 120, vs="usd") -> pd.DataFrame:
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    r = requests.get(url, params={"vs_currency": vs, "days": str(days), "interval": "daily"}, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "prices" not in data or not data["prices"]:
        raise RuntimeError(f"No price data for {cg_id}")
    df = pd.DataFrame(data["prices"], columns=["ts", "price"])
    df["date"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df = df[["date", "price"]].drop_duplicates("date").reset_index(drop=True)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for L in lags:
        out[f"lag_{L}"] = out["price"].shift(L)
    for w in rolls:
        out[f"roll_mean_{w}"] = out["price"].rolling(w).mean()
        out[f"roll_std_{w}"]  = out["price"].rolling(w).std()
    out["ret_1d"] = out["price"].pct_change(1)
    out = out.dropna().reset_index(drop=True)
    return out

def train_and_predict_next(df_feat: pd.DataFrame) -> float:
    X = df_feat.drop(columns=["date", "price"])
    y = df_feat["price"]
    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X, y)
    x_next = X.iloc[[-1]]
    return float(model.predict(x_next)[0])

rows = []
plots = []
for sym, cgid in coins.items():
    try:
        df_hist = fetch_history(cgid, lookback_days, vs_currency)
        df_feat = make_features(df_hist)
        if df_feat.empty:
            raise RuntimeError("Not enough rows after feature creation.")
        pred_next = train_and_predict_next(df_feat)
        last_date = df_feat["date"].iloc[-1]
        last_price = float(df_feat["price"].iloc[-1])
        amt = holdings.get(sym, 0.0)
        curr_val = amt * last_price
        pred_val = amt * pred_next
        rows.append({
            "symbol": sym,
            "last_date": last_date,
            "amount": amt,
            "last_price_usd": last_price,
            "pred_next_price_usd": pred_next,
            "current_value_usd": curr_val,
            "predicted_value_usd": pred_val,
            "exp_change_usd": pred_next - last_price,
            "exp_change_%": 100 * (pred_next - last_price) / last_price if last_price else np.nan
        })
        plots.append((sym, df_hist[["date","price"]], pred_next))
    except Exception as e:
        rows.append({
            "symbol": sym,
            "last_date": None,
            "amount": holdings.get(sym, 0.0),
            "last_price_usd": None,
            "pred_next_price_usd": None,
            "current_value_usd": None,
            "predicted_value_usd": None,
            "exp_change_usd": None,
            "exp_change_%": None,
        })
        print(f"[WARN] {sym}: {e}")

# ---- Summary table (per coin + totals) ----
df_out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
curr_total = float(df_out["current_value_usd"].fillna(0).sum())
pred_total = float(df_out["predicted_value_usd"].fillna(0).sum())
delta_total = pred_total - curr_total
delta_total_pct = 100 * delta_total / curr_total if curr_total else np.nan

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print("\nPer-coin forecast:")
print(df_out[[
    "symbol","amount","last_price_usd","pred_next_price_usd",
    "current_value_usd","predicted_value_usd","exp_change_%"
]].to_string(index=False))

print("\nCurrent portfolio total: ${:,.2f}".format(curr_total))
print("Predicted next-day total: ${:,.2f}".format(pred_total))
print("Expected change: ${:,.2f} ({:+.2f}%)".format(delta_total, delta_total_pct))

# Save CSV snapshot
df_out.to_csv("portfolio_forecast_nextday.csv", index=False)
print("Saved -> portfolio_forecast_nextday.csv")

# ---- Plots: one chart per coin ----
for sym, hist, pred in plots:
    plt.figure(figsize=(10,5))
    plt.plot(hist["date"], hist["price"], label=f"{sym} historical")
    plt.axhline(pred, linestyle="--", label=f"{sym} predicted next-day")
    plt.title(f"{sym}: Price & Next-Day Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    out_png = f"pred_{sym}.png"
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved chart -> {out_png}")
