#!/usr/bin/env python3
"""
Test script to verify weights saving functionality
"""

# %%
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# Create sample data
np.random.seed(42)
n = 1000

data = pd.DataFrame(
    {
        "user_id": range(n),
        "experiment": np.random.choice(["exp1", "exp2"], n),
        "treatment": np.random.binomial(1, 0.5, n),
        "outcome1": np.random.normal(10, 2, n),
        "outcome2": np.random.normal(5, 1, n),
        "covariate1": np.random.normal(0, 1, n),
        "covariate2": np.random.binomial(1, 0.3, n),
    }
)

# Adjust outcomes based on treatment for some effect
data.loc[data["treatment"] == 1, "outcome1"] += 0.5
data.loc[data["treatment"] == 1, "outcome2"] += 0.3

print("Sample data created:")
print(data.head())
print(f"Data shape: {data.shape}")

# Test ExperimentAnalyzer with unit_identifier
analyzer = ExperimentAnalyzer(
    data=data,
    outcomes=["outcome1", "outcome2"],
    treatment_col="treatment",
    experiment_identifier=["experiment"],
    covariates=["covariate1", "covariate2"],
    adjustment="balance",
    unit_identifier="user_id",
    balance_method="ps-logistic",
)

print("\nRunning get_effects with balance adjustment...")
analyzer.get_effects()

print("\nResults:")
print(analyzer.results.head())

print("\nWeights:")
weights = analyzer.weights
if weights is not None:
    print(f"Weights shape: {weights.shape}")
    print("Weights columns:", weights.columns.tolist())
    print("Sample weights:")
    print(weights.head(10))
else:
    print("No weights saved")

# %%
