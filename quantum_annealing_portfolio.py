# Quantum Annealing Portfolio Optimization using D-Wave Ocean SDK
# Requires: pip install dwave-ocean-sdk

from dwave.system import DWaveSampler, EmbeddingComposite
import dimod
import numpy as np

# Define expected returns and covariances for 5 assets
returns = np.array([0.12, 0.10, 0.07, 0.03, 0.05])
cov_matrix = np.array([
    [0.10, 0.02, 0.01, 0.00, 0.00],
    [0.02, 0.08, 0.01, 0.00, 0.00],
    [0.01, 0.01, 0.09, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.07, 0.02],
    [0.00, 0.00, 0.00, 0.02, 0.08]
])

# Objective: maximize return - risk_aversion * variance
risk_aversion = 0.5

# Binary decision variables: x_i = 1 if asset i is included, else 0
Q = {}
n = len(returns)
for i in range(n):
    for j in range(n):
        if i == j:
            Q[(i, i)] = -returns[i] + risk_aversion * cov_matrix[i][i]
        else:
            Q[(i, j)] = risk_aversion * cov_matrix[i][j]

# Solve using D-Wave quantum annealer
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample_qubo(Q, num_reads=100)

# Interpret result
best_sample = sampleset.first.sample
selected_assets = [i for i in range(n) if best_sample[i] == 1]
print("Selected assets:", selected_assets)
print("Sample energy:", sampleset.first.energy)
