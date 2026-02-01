import os, requests, pandas as pd, matplotlib.pyplot as plt

holdings = {
    "BTC": 0.062134,
    "SOL": 2.48932,
    "XRP": 106.187,
    "XLM": 142.939,
    "ADA": 21.8174,
}

API_KEY = os.getenv("CMC_API_KEY") or "aacf2e06-bbf9-445c-98e0-648ded56cc1e"
headers = {"X-CMC_PRO_API_KEY": API_KEY, "Accept": "application/json"}

# Manual CMC ID mapping
id_map = {
    "BTC": 1,
    "SOL": 5426,
    "XRP": 52,
    "XLM": 512,
    "ADA": 2010
}

ids = ",".join(str(id_map[sym]) for sym in holdings.keys())

# Get quotes
url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
r_q = requests.get(url, headers=headers, params={"id": ids, "convert": "USD"})
r_q.raise_for_status()
payload = r_q.json()

rows = []
for sym, amt in holdings.items():
    cid = str(id_map[sym])
    q = payload["data"][cid]["quote"]["USD"]
    price = q["price"]
    rows.append({
        "symbol": sym,
        "amount": amt,
        "price_usd": price,
        "value_usd": amt * price,
        "chg_24h_%": q.get("percent_change_24h")
    })

df = pd.DataFrame(rows).sort_values("value_usd", ascending=False).reset_index(drop=True)
total = df["value_usd"].sum()
df["alloc_%"] = 100 * df["value_usd"] / total

print(df.to_string(index=False))
print("\nPortfolio total: ${:,.2f}".format(total))

# Charts
plt.figure(figsize=(9,5))
plt.bar(df["symbol"], df["value_usd"])
plt.title("Portfolio Value by Asset (USD)")
plt.ylabel("Value (USD)")
plt.xlabel("Asset")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9,5))
plt.bar(df["symbol"], df["amount"])
plt.title("Holdings by Asset (Native Units)")
plt.ylabel("Amount")
plt.xlabel("Asset")
plt.tight_layout()
plt.show()

