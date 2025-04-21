import streamlit as st
import numpy as np
import pandas as pd
import investpy
from datetime import datetime, timedelta
from dimod import SimulatedAnnealingSampler

st.set_page_config(page_title="📈 EGX Portfolio Optimizer", layout="centered")
st.title("📈 Optimize Egyptian Stocks (EGX) using investpy")

st.markdown("This app uses **investpy** to fetch EGX stock data and optimize a portfolio using simulated annealing.")

# EGX stock names (from investpy.get_stocks(country='egypt'))
valid_egx_names = [
    "Commercial International Bank (Egypt)", "Telecom Egypt", "Talaat Moustafa Group",
    "Ezz Steel", "Eastern Company", "Elsewedy Electric", "Qalaa Holdings",
    "Palm Hills Developments", "Heliopolis Company for Housing and Development", "Sidi Kerir Petrochemicals"
]

selected_stocks = st.multiselect("Select EGX stocks to optimize:", valid_egx_names, default=valid_egx_names[:5])
top_k = st.slider("Top-k Assets to Select", 2, len(selected_stocks), min(5, len(selected_stocks)))
risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)

# Fetch EGX stock data
def fetch_egx_data(name):
    try:
        df = investpy.get_stock_historical_data(stock=name,
                                                country='egypt',
                                                from_date=(datetime.today() - timedelta(days=180)).strftime('%d/%m/%Y'),
                                                to_date=datetime.today().strftime('%d/%m/%Y'))
        df = df[['Close']].rename(columns={"Close": name})
        return df
    except Exception as e:
        st.warning(f"❌ Could not fetch `{name}`: {e}")
        return None

st.info("📥 Downloading EGX data using investpy...")
price_data = pd.DataFrame()
valid_selected = []

for name in selected_stocks:
    df = fetch_egx_data(name)
    if df is not None:
        valid_selected.append(name)
        if price_data.empty:
            price_data = df
        else:
            price_data = price_data.join(df, how="outer")

if len(valid_selected) < top_k:
    st.error(f"Only {len(valid_selected)} valid assets found. This is fewer than top_k = {top_k}.")
    st.stop()

# Process return data
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
    st.subheader("📈 Selected EGX Stocks:")
    st.write([tickers[i] for i in selected_indices])
    st.write(f"Sample Energy: `{energy:.4f}`")

    port_return = np.sum([mean_returns[i] for i in selected_indices])
    port_var = np.sum([cov_matrix[i][j] for i in selected_indices for j in selected_indices])
    port_sharpe = port_return / (np.sqrt(port_var) + 1e-6)

    st.metric("📈 Expected Return", f"{port_return:.2%}")
    st.metric("📉 Expected Risk (Variance)", f"{port_var:.4f}")
    st.metric("🧠 Sharpe Ratio", f"{port_sharpe:.2f}")
