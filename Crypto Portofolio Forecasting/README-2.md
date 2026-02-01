# Advanced Crypto Portfolio Forecasting
Machine Learning–Driven Predictive Analytics for Cryptocurrency Portfolios

# Overview
This project forecasts cryptocurrency price movements and simulates portfolio growth using machine learning.
It combines historical market data, regression-based models, and portfolio aggregation to generate both next-day and 1-year forecasts for key crypto assets:
Bitcoin (BTC)
Cardano (ADA)
Solana (SOL)
Ripple (XRP)
Stellar (XLM)
The system is built entirely in Python using pandas, scikit-learn, matplotlib, and numpy, providing data-driven financial insights with reproducible outputs.

# Objectives
Predict short-term and long-term prices for major cryptocurrencies
Model total portfolio growth over one year
Provide clear visualizations of historical vs. predicted data
Offer a flexible structure for adding or removing assets

# Repository Contents
predictions-advanced.py — Core ML model for crypto price forecasting
portofolio.py — Aggregates forecasts into total portfolio projection
coin_price_forecasts_1year.csv — 1-year forecasts for each asset
portfolio_forecast_nextday.csv — Next-day portfolio prediction
portfolio_forecast_1year.csv — Annual projection summary
pred_BTC.png — BTC next-day forecast chart
pred_ADA.png — ADA next-day forecast chart
pred_SOL.png — SOL next-day forecast chart
pred_XRP.png — XRP next-day forecast chart
pred_XLM.png — XLM next-day forecast chart
portfolio_forecast_1year.png — Combined portfolio forecast visualization

# Customization
The scripts are fully editable to match your personal holdings or research needs.
To adapt the project to your own assets:

1.	Open predictions-advanced.py and replace coin names or CSV file paths with your own data sources.
2.	In portofolio.py, edit the list of assets and adjust weightings or allocation percentages to reflect your real portfolio.
3.	Add or remove any cryptocurrency to expand or simplify your analysis.

Example: If your portfolio only includes BTC, ETH, and DOGE, replace the existing tickers with those and modify file imports accordingly.
This makes the framework usable for any combination of assets, including traditional markets.

# How to Run 
  Clone the repository
  git clone https://github.com/gresium/Crypto-Portofolio-Prediction-Charts-.git
  
  cd Crypto-Portofolio-Prediction-Charts-

# Install dependencies
  pip install pandas numpy matplotlib scikit-learn

# Run the scripts
  python predictions-advanced.py
  python portofolio.py

# Key Features
Machine-learning crypto price predictions
Next-day and long-term portfolio simulations
High-quality visualizations in .png format
Configurable to any asset or allocation amount
Generates reproducible CSV forecasts

# Interpretation
This project demonstrates how machine learning can be applied to financial forecasting.
It highlights volatility patterns, price momentum, and portfolio balance through visual analytics.
All forecasts are for educational and research purposes only — not investment advice

# Author 
Developed by Gresa Hisa (@gresium)
AI & Cybersecurity Engineer | AI &  Machine Learning Specialist
GitHub: github.com/gresium









