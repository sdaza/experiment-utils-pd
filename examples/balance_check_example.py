"""
Example demonstrating the new balance checking functionality.

This example shows how to use:
1. The standalone check_covariate_balance() function
2. The ExperimentAnalyzer.check_balance() method
"""

# %% Imports
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer
from experiment_utils.utils import check_covariate_balance

# %% Create sample data with covariates
np.random.seed(42)
n = 1000

df = pd.DataFrame(
    {
        "experiment": np.random.choice(["exp1", "exp2"], n),
        "treatment": np.random.choice([0, 1], n),
        "outcome": np.random.randn(n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "is_member": np.random.choice([0, 1], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    }
)

# %% Example 1: Standalone function
print("1. Using standalone check_covariate_balance() function")

balance_standalone = check_covariate_balance(
    data=df,
    treatment_col="treatment",
    covariates=["age", "income", "is_member", "region"],
    threshold=0.1,
)

print("\nBalance Results (standalone):")
print(balance_standalone.to_string(index=False))

balanced_pct = balance_standalone["balance_flag"].mean() * 100
print(f"\nBalance Summary: {balanced_pct:.1f}% of covariates are balanced")

imbalanced = balance_standalone[balance_standalone["balance_flag"] == 0]
if not imbalanced.empty:
    print(f"Imbalanced covariates: {imbalanced['covariate'].tolist()}")

# %% Example 2: ExperimentAnalyzer method (single experiment)
print("2. Using ExperimentAnalyzer.check_balance() method (single experiment)")

# Filter to one experiment for simplicity
df_single = df[df["experiment"] == "exp1"].copy()

analyzer = ExperimentAnalyzer(
    data=df_single,
    outcomes=["outcome"],
    treatment_col="treatment",
    covariates=["age", "income", "is_member", "region"],
)

balance_method = analyzer.check_balance(threshold=0.1)

print("\nBalance Results (ExperimentAnalyzer):")
print(balance_method.to_string(index=False))

# %% Example 3: Multiple experiments
print("3. Checking balance across multiple experiments")

analyzer_multi = ExperimentAnalyzer(
    data=df,
    outcomes=["outcome"],
    treatment_col="treatment",
    experiment_identifier="experiment",
    covariates=["age", "income", "is_member"],
)

balance_multi = analyzer_multi.check_balance(threshold=0.1)

print("\nBalance Results (multiple experiments):")
print(balance_multi.to_string(index=False))

# Summary by experiment
for exp in balance_multi["experiment"].unique():
    exp_balance = balance_multi[balance_multi["experiment"] == exp]
    balanced_pct = exp_balance["balance_flag"].mean() * 100
    print(f"\n{exp}: {balanced_pct:.1f}% balanced")

# %% Example 4: Check balance before and after running get_effects
print("4. Checking balance independently of get_effects()")

analyzer_independent = ExperimentAnalyzer(
    data=df_single,
    outcomes=["outcome"],
    treatment_col="treatment",
    covariates=["age", "income"],
)

# Check balance BEFORE calling get_effects
# print("\nBalance check BEFORE get_effects():")
# balance_before = analyzer_independent.check_balance()
# print(f"Covariates checked: {balance_before['covariate'].tolist()}")
# print(f"Balance: {balance_before['balance_flag'].mean():.2%}")

# Now run get_effects (which also checks balance internally)
analyzer_independent.get_effects()

# # Check balance AFTER get_effects
# print("\nBalance check AFTER get_effects():")
# balance_after = analyzer_independent.check_balance()
# print(f"Covariates checked: {balance_after['covariate'].tolist()}")
# print(f"Balance: {balance_after['balance_flag'].mean():.2%}")

# # Also access the balance from get_effects
# print("\nBalance from get_effects() (stored in .balance property):")
# internal_balance = analyzer_independent.balance
# if internal_balance is not None:
#     print(f"Covariates: {internal_balance['covariate'].tolist()}")
#     print(f"Balance: {internal_balance['balance_flag'].mean():.2%}")

# %% Example 5: Custom thresholds
print("5. Using different balance thresholds")

analyzer_threshold = ExperimentAnalyzer(
    data=df_single,
    outcomes=["outcome"],
    treatment_col="treatment",
    covariates=["age", "income"],
)

# Strict threshold
balance_strict = analyzer_threshold.check_balance(threshold=0.05)
print("\nStrict threshold (0.05):")
print(f"Balanced: {balance_strict['balance_flag'].sum()}/{len(balance_strict)}")

# Lenient threshold
balance_lenient = analyzer_threshold.check_balance(threshold=0.2)
print("\nLenient threshold (0.2):")
print(f"Balanced: {balance_lenient['balance_flag'].sum()}/{len(balance_lenient)}")
