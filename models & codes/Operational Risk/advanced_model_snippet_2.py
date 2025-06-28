### Scenario Analysis & Monte Carlo Simulation

import numpy as np
import pandas as pd

# Example: Simulate potential losses for a cyber-attack scenario
np.random.seed(42)
n_simulations = 10000
mean_loss = 50000   # average loss per event
std_dev = 20000     # standard deviation of loss
annual_frequency = 3  # expected number of such events per year

# Monte Carlo simulation of annual loss distribution
total_losses = []
for _ in range(n_simulations):
    n_events = np.random.poisson(annual_frequency)
    losses = np.random.normal(mean_loss, std_dev, n_events)
    total_losses.append(np.sum(losses[losses > 0]))  # ignore negative losses

# Calculate risk metrics
VaR_99 = np.percentile(total_losses, 99)
expected_loss = np.mean(total_losses)

print(f"Expected Annual Loss: ${expected_loss:,.2f}")
print(f"Value at Risk (99%): ${VaR_99:,.2f}")
