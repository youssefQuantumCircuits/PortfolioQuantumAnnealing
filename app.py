import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="📊 Final Quant Optimizer", layout="wide")
st.title("📊 Optimized Portfolio with Weighting, Sharpe, Costs, Cumulative Plot")

tickers_input = st.text_area("Enter tickers (comma-separated):", "AAPL,MSFT,NVDA,TSLA,GOOGL", height=100)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
if not 2 <= len(tickers) <= 20:
    st.warning("Please enter between 2 and 20 tickers.")
    st.stop()

top_k = st.slider("Top-k Assets to Select", 2, len(tickers), min(5, len(tickers)))
risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)
trans_cost = st.slider("Transaction Cost (%)", 0.0, 0.05, 0.01)

end_date = datetime.today()
start_date = end_date - timedelta(days=365)
raw_data = yf.download(tickers + ['SPY'], start=start_date, end=end_date)

if isinstance(raw_data.columns, pd.MultiIndex):
    price_type = 'Adj Close' if 'Adj Close' in raw_data.columns.levels[0] else 'Close'
    data = raw_data[price_type]
else:
    st.error("❌ Data structure is not MultiIndex.")
    st.stop()

data.dropna(axis=1, how='all', inplace=True)
returns = data.pct_change().dropna()
benchmark_returns = returns['SPY']
asset_returns = returns.drop(columns=['SPY'])

# Heatmap
st.subheader("📊 Return Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(asset_returns.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Optimization (past 3 months)
opt_returns = asset_returns[-63:]
mean_returns = opt_returns.mean().values
cov_matrix = opt_returns.cov().values
opt_tickers = list(opt_returns.columns)

n = len(opt_tickers)
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
        binary_weights = np.array([best_sample[i] for i in range(n)])

    st.success("Optimization Complete")
    selected_names = [opt_tickers[i] for i in range(n) if binary_weights[i] == 1]
    st.subheader("📈 Selected Portfolio")
    st.write(selected_names)

    if binary_weights.sum() == 0:
        st.warning("No valid assets selected.")
        st.stop()

    weights = binary_weights / binary_weights.sum()
    port_returns = (asset_returns[selected_names] @ weights).dropna()
    port_return_avg = port_returns.mean()
    port_var = port_returns.var()
    port_sharpe = port_return_avg / (np.sqrt(port_var) + 1e-6)

    spy_avg = benchmark_returns.mean()
    spy_std = benchmark_returns.std()
    spy_sharpe = spy_avg / (spy_std + 1e-6)

    net_return = port_return_avg - trans_cost * weights.sum()

    st.metric("📈 Expected Return (Gross)", f"{port_return_avg:.2%}")
    st.metric("💸 Net Return (After Cost)", f"{net_return:.2%}")
    st.metric("📉 Risk (Variance)", f"{port_var:.4f}")
    st.metric("🧠 Sharpe Ratio", f"{port_sharpe:.2f}")
    st.metric("📊 SPY Sharpe Ratio", f"{spy_sharpe:.2f}")

    st.subheader("📈 Cumulative Return Comparison")
    port_cum = (1 + port_returns).cumprod()
    spy_cum = (1 + benchmark_returns).cumprod()
    fig2, ax2 = plt.subplots()
    ax2.plot(port_cum, label="Optimized Portfolio")
    ax2.plot(spy_cum, label="SPY", linestyle="--")
    ax2.set_ylabel("Cumulative Return")
    ax2.legend()
    st.pyplot(fig2)
