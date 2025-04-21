import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler

st.set_page_config(page_title="📊 Alpha Vantage Portfolio Optimizer", layout="centered")
st.title("📊 Optimize Portfolio using Alpha Vantage")
st.markdown("This app uses **Alpha Vantage** to fetch real-time market data and optimize a portfolio using simulated annealing.")

# Setup API key (user must provide their own)
api_key = st.text_input("Enter your Alpha Vantage API Key:", type="password")
tickers_input = st.text_area("Enter up to 10 tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA", height=100)
ticker_list = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

if not api_key:
    st.warning("Please enter your Alpha Vantage API key.")
    st.stop()

if not 2 <= len(ticker_list) <= 10:
    st.warning("Please enter between 2 and 10 tickers.")
    st.stop()

risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)
top_k = st.slider("Top-k Assets to Select", 2, len(ticker_list), min(5, len(ticker_list)))

# Fetch historical data (last 100 trading days)
def fetch_alpha_vantage_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": api_key
    }
    r = requests.get(url, params=params)
    try:
        data = r.json()["Time Series (Daily)"]
        df = pd.DataFrame(data).T
        df = df.rename(columns={"5. adjusted close": symbol})
        df[symbol] = df[symbol].astype(float)
        df.index = pd.to_datetime(df.index)
        return df[[symbol]]
    except Exception:
        return None

# Combine all valid data
st.info("📥 Fetching data from Alpha Vantage...")
valid_tickers = []
price_data = pd.DataFrame()

for ticker in ticker_list:
    df = fetch_alpha_vantage_data(ticker, api_key)
    if df is not None and len(df.columns) == 1:
        valid_tickers.append(ticker)
        if price_data.empty:
            price_data = df
        else:
            price_data = price_data.join(df, how="outer")

if len(valid_tickers) < top_k:
    st.error(f"Only {len(valid_tickers)} valid tickers returned. This is fewer than top_k = {top_k}.")
    st.stop()

price_data = price_data.sort_index().dropna()
returns_df = price_data.pct_change().dropna()

mean_returns = returns_df.mean().values
cov_matrix = returns_df.cov().values
tickers = list(returns_df.columns)

# Build QUBO
n = len(tickers)
Q = {}
for i in range(n):
    for j in range(n):
        if i == j:
            Q[(i, i)] = -mean_returns[i] + risk_aversion * cov_matrix[i][i]
        else:
            Q[(i, j)] = risk_aversion * cov_matrix[i][j]

# Top-k constraint
penalty = 4.0
for i in range(n):
    Q[(i, i)] += penalty * (1 - 2 * top_k)
    for j in range(i+1, n):
        Q[(i, j)] += 2 * penalty

if st.button("Optimize Portfolio"):
    with st.spinner("Running simulated annealing..."):
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(Q, num_reads=500)
        best_sample = sampleset.first.sample
        selected_indices = [i for i in range(n) if best_sample[i] == 1]
        energy = sampleset.first.energy

    st.success("Optimization Complete")
    st.subheader("📈 Selected Tickers:")
    st.write([tickers[i] for i in selected_indices])
    st.write(f"Sample Energy: `{energy:.4f}`")

    port_return = np.sum([mean_returns[i] for i in selected_indices])
    port_var = np.sum([cov_matrix[i][j] for i in selected_indices for j in selected_indices])
    port_sharpe = port_return / (np.sqrt(port_var) + 1e-6)

    st.metric("📈 Expected Return", f"{port_return:.2%}")
    st.metric("📉 Expected Risk (Variance)", f"{port_var:.4f}")
    st.metric("🧠 Sharpe Ratio", f"{port_sharpe:.2f}")
