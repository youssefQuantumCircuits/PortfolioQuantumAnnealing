import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler

st.set_page_config(page_title="📊 Real-Data Portfolio Optimizer", layout="centered")
st.title("📊 Optimize Portfolio with Real Market Data")
st.markdown("This app fetches historical returns of real stocks and uses **simulated annealing** to find an optimal portfolio.")

# Settings
tickers = st.text_area("Enter up to 50 tickers (comma-separated)", "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,BRK-B,JNJ,V", height=100)
ticker_list = [ticker.strip().upper() for ticker in tickers.split(",") if ticker.strip()]

if not 5 <= len(ticker_list) <= 50:
    st.warning("Please enter between 5 and 50 tickers.")
    st.stop()

risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)
top_k = st.slider("Top-k Assets to Select", 5, min(20, len(ticker_list)), 10)

# Fetch historical prices (6 months)
end = datetime.today()
start = end - timedelta(days=180)
data = yf.download(ticker_list, start=start, end=end)

# Handle multi-index data
if isinstance(data.columns, pd.MultiIndex):
    data = data['Adj Close']
elif 'Adj Close' in data.columns:
    data = data[['Adj Close']]
    data.columns = [ticker_list[0]]

if data.isnull().all().any():
    st.error("Some tickers failed to download or returned all NaN data.")
    st.stop()

# Calculate daily returns and summary stats
returns_df = data.pct_change().dropna()
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

# Add top-k constraint
penalty = 4.0
for i in range(n):
    Q[(i, i)] += penalty * (1 - 2 * top_k)
    for j in range(i+1, n):
        Q[(i, j)] += 2 * penalty

# Run optimizer
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
