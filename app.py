import streamlit as st
import numpy as np
import os
from dotenv import load_dotenv
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# Load API token from .env
load_dotenv()
token = os.getenv("DWAVE_API_TOKEN")

st.title("🌌 Quantum Portfolio Optimizer")
st.write("This app uses a D-Wave quantum annealer to solve a basic portfolio optimization problem over 5 assets.")

if not token:
    st.error("DWAVE_API_TOKEN not found. Please set it in a .env file.")
    st.stop()

# Define expected returns and covariances
returns = np.array([0.12, 0.10, 0.07, 0.03, 0.05])
cov_matrix = np.array([
    [0.10, 0.02, 0.01, 0.00, 0.00],
    [0.02, 0.08, 0.01, 0.00, 0.00],
    [0.01, 0.01, 0.09, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.07, 0.02],
    [0.00, 0.00, 0.00, 0.02, 0.08]
])

risk_aversion = st.slider("Risk Aversion", 0.0, 1.0, 0.5)

# Build QUBO
Q = {}
n = len(returns)
for i in range(n):
    for j in range(n):
        if i == j:
            Q[(i, i)] = -returns[i] + risk_aversion * cov_matrix[i][i]
        else:
            Q[(i, j)] = risk_aversion * cov_matrix[i][j]

if st.button("Optimize Portfolio"):
    with st.spinner("Running on D-Wave Quantum Annealer..."):
        os.environ["DWAVE_API_TOKEN"] = token  # Set token for DWaveSampler
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample_qubo(Q, num_reads=100)
        best_sample = sampleset.first.sample
        selected_assets = [i for i in range(n) if best_sample[i] == 1]
        energy = sampleset.first.energy

    st.success("Optimization Complete")
    st.write("### Selected Assets:")
    st.write([f"Asset {i+1}" for i in selected_assets])
    st.write(f"Sample Energy: {energy}")
