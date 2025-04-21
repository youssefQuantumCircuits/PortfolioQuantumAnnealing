import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Enhanced Portfolio Optimizer", layout="centered")
st.title("📊 yfinance Portfolio Optimizer (12-Month, Benchmark & Heatmap)")

st.markdown("This app uses **yfinance** to fetch stock data, compares to a benchmark (SPY), and shows a correlation heatmap.")

tickers_input = st.text_area("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):", "AAPL,MSFT,NVDA,TSLA,GOOGL", height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not 2 <= len(tickers) <= 20:
    st.warning("Please enter between 2 and 20 tickers.")
    st.stop()

top_k = st.slider("Top-k Assets to Select", 2, len(tickers), min(5, len(tickers)))
risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)

# Fetch yfinance data for 1 year
end_date = datetime.today()
start_date = end_date - timedelta(days=365)
raw_data = yf.download(tickers + ['SPY'], start=start_date, end=end_date)

# Determine price type
if isinstance(raw_data.columns, pd.MultiIndex):
    price_type = 'Adj Close' if 'Adj Close' in raw_data.columns.levels[0] else 'Close'
    data = raw_data[price_type]
else:
    st.error("❌ Data returned is not in MultiIndex format.")
    st.stop()

# Clean and compute returns
data.dropna(axis=1, how='all', inplace=True)
returns_df = data.pct_change().dropna()
benchmark_returns = returns_df['SPY']
asset_returns = returns_df.drop(columns=['SPY'])

# Show correlation heatmap
st.subheader("📊 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(asset_returns.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Run optimization
mean_returns = asset_returns.mean().values
cov_matrix = asset_returns.cov().values
tickers = list(asset_returns.columns)

if len(tickers) < top_k:
    st.error("❌ Not enough valid assets after cleaning.")
    st.stop()

n = len(tickers)
Q = {}
for i in range(n):
    for j in range(n):
        if i == j:
            Q[(i, i)] = -mean_returns[i] + risk_aversion * cov_matrix[i][i]
        else:
            Q[(i, j)] = risk_aversion * cov_matrix[i][j]

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

    selected_returns = asset_returns[[tickers[i] for i in selected_indices]].mean(axis=1)
    port_return = selected_returns.mean()
    port_var = selected_returns.var()
    port_sharpe = port_return / (np.sqrt(port_var) + 1e-6)

    # Benchmark comparison
    benchmark_avg = benchmark_returns.mean()
    benchmark_std = benchmark_returns.std()
    benchmark_sharpe = benchmark_avg / (benchmark_std + 1e-6)

    st.metric("📈 Portfolio Expected Return", f"{port_return:.2%}")
    st.metric("📉 Portfolio Risk (Variance)", f"{port_var:.4f}")
    st.metric("🧠 Portfolio Sharpe Ratio", f"{port_sharpe:.2f}")

    st.metric("📊 SPY Sharpe Ratio", f"{benchmark_sharpe:.2f}")
