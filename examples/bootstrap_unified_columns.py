"""
Quick example demonstrating the unified column structure for bootstrap inference.

With the updated implementation, bootstrap and asymptotic inference use the SAME column names:
- pvalue
- stat_significance
- abs_effect_lower
- abs_effect_upper

The only difference is the 'inference_method' column which indicates 'asymptotic' or 'bootstrap'.
"""

import numpy as np
import pandas as pd
from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# Set random seed
np.random.seed(42)

# Generate sample data
n = 500
data = pd.DataFrame(
    {
        "experiment_id": "test",
        "treatment": np.random.binomial(1, 0.5, n),
        "age": np.random.normal(35, 10, n),
        "outcome": np.random.normal(100, 20, n),
    }
)
data.loc[data["treatment"] == 1, "outcome"] += 5  # Add treatment effect

print("=" * 80)
print("Unified Column Structure Demo")
print("=" * 80)

# Asymptotic inference
print("\n1. Asymptotic Inference")
print("-" * 80)
analyzer_async = ExperimentAnalyzer(
    data=data, outcomes=["outcome"], treatment_col="treatment", experiment_identifier=["experiment_id"], bootstrap=False
)
analyzer_async.get_effects()
results_async = analyzer_async.results

print("\nColumns:", list(results_async.columns))
print("\nKey columns:")
print(results_async[["outcome", "inference_method", "pvalue", "abs_effect_lower", "abs_effect_upper"]])

# Bootstrap inference
print("\n\n2. Bootstrap Inference")
print("-" * 80)
analyzer_boot = ExperimentAnalyzer(
    data=data,
    outcomes=["outcome"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    bootstrap=True,
    bootstrap_iterations=500,
    bootstrap_seed=42,
)
analyzer_boot.get_effects()
results_boot = analyzer_boot.results

print("\nColumns:", list(results_boot.columns))
print("\nKey columns:")
print(results_boot[["outcome", "inference_method", "pvalue", "abs_effect_lower", "abs_effect_upper"]])

# Combined comparison
print("\n\n3. Side-by-Side Comparison")
print("-" * 80)

comparison = pd.DataFrame(
    {
        "Method": ["Asymptotic", "Bootstrap"],
        "inference_method": [results_async["inference_method"].values[0], results_boot["inference_method"].values[0]],
        "pvalue": [results_async["pvalue"].values[0], results_boot["pvalue"].values[0]],
        "CI_lower": [results_async["abs_effect_lower"].values[0], results_boot["abs_effect_lower"].values[0]],
        "CI_upper": [results_async["abs_effect_upper"].values[0], results_boot["abs_effect_upper"].values[0]],
    }
)

print("\n", comparison.to_string(index=False))

print("\n" + "=" * 80)
print("Key Insight: Same column names, different values based on inference_method!")
print("=" * 80)
