import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler

st.set_page_config(page_title="📊 yfinance Debug Portfolio Optimizer", layout="centered")
st.title("📊 yfinance Portfolio Optimizer (with Debug Info)")

st.markdown("Connected via VPN? Paste tickers and debug Yahoo Finance's response if it fails.")

tickers_input = st.text_area("Enter tickers (comma-separated, e.g., AAPL,MSFT,GOOGL):", "AAPL,MSFT,NVDA,TSLA,GOOGL", height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if not 2 <= len(tickers) <= 20:
    st.warning("Please enter between 2 and 20 tickers.")
    st.stop()

top_k = st.slider("Top-k Assets to Select", 2, len(tickers), min(5, len(tickers)))
risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)

st.info("📥 Downloading data using yfinance (last 6 months)...")
end_date = datetime.today()
start_date = end_date - timedelta(days=180)
data = yf.download(tickers, start=start_date, end=end_date)

st.subheader("🧪 Raw DataFrame Preview:")
st.dataframe(data.head())

# Extract Adj Close or fallback
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.levels[0]:
        data = data['Adj Close']
        st.success("✅ Found and extracted 'Adj Close' data.")
    else:
        st.warning("⚠️ 'Adj Close' not found — showing top-level column structure:")
        st.code(str(data.columns.levels[0]))
        st.stop()
else:
    st.error("❌ Data structure was not MultiIndex. Please check your VPN or ticker input.")
    st.dataframe(data.head())
    st.stop()

data.dropna(axis=1, how='all', inplace=True)
returns_df = data.pct_change().dropna()

if returns_df.empty or len(returns_df.columns) < top_k:
    st.error("❌ Not enough valid return data.")
    st.stop()

mean_returns = returns_df.mean().values
cov_matrix = returns_df.cov().values
tickers = list(returns_df.columns)

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

    port_return = np.sum([mean_returns[i] for i in selected_indices])
    port_var = np.sum([cov_matrix[i][j] for i in selected_indices for j in selected_indices])
    port_sharpe = port_return / (np.sqrt(port_var) + 1e-6)

    st.metric("📈 Expected Return", f"{port_return:.2%}")
    st.metric("📉 Expected Risk (Variance)", f"{port_var:.4f}")
    st.metric("🧠 Sharpe Ratio", f"{port_sharpe:.2f}")
