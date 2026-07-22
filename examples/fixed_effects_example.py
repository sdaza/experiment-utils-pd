# %% [markdown]
# # Unit fixed effects
#
# Within-unit panel: treatment switches inside units. Shows
# `ExperimentAnalyzer` with `fixed_effects=` and clustered SEs recovering a
# known treatment effect.
#
# Run:
#
#     uv run python examples/fixed_effects_example.py

# %%
import numpy as np
import pandas as pd
from numpy.random import default_rng

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# %% [markdown]
# ## Simulate a within-unit panel
#
# 80 units × 6 periods; treatment switches within unit so units are "switchers".

# %%
rng = default_rng(42)

n_units = 80
n_periods = 6
true_effect = 2.5

unit_ids = np.repeat(np.arange(n_units), n_periods)
period_ids = np.tile(np.arange(n_periods), n_units)

# Unit-level heterogeneity (fixed effect)
unit_fe = rng.normal(0, 5, n_units)
unit_fe_expanded = np.repeat(unit_fe, n_periods)

# Time-varying covariate
cov = rng.normal(0, 1, n_units * n_periods)

# Treatment varies within unit: each unit is treated in some periods
treatment = rng.binomial(1, 0.5, n_units * n_periods)

# Outcome: unit FE + treatment effect + covariate effect + noise
outcome = unit_fe_expanded + true_effect * treatment + 0.8 * cov + rng.normal(0, 1, n_units * n_periods)

df = pd.DataFrame(
    {
        "unit": unit_ids,
        "period": period_ids,
        "treatment": treatment,
        "cov": cov,
        "revenue": outcome,
    }
)

print(f"Dataset: {df.shape[0]} rows, {n_units} units, {n_periods} periods")
print(f"True treatment effect: {true_effect}")

# %% [markdown]
# ## Estimate with unit fixed effects

# %%
analyzer = ExperimentAnalyzer(
    data=df,
    outcomes=["revenue"],
    treatment_col="treatment",
    regression_covariates=["cov"],  # time-varying control
    fixed_effects=["unit"],  # within-unit identification
    cluster_col="unit",  # clustered standard errors
    fixed_effects_min_switcher_pct=10.0,
)

analyzer.get_effects()
res = analyzer.results

diag_cols = [c for c in ["fe_absorbed", "n_units", "n_switchers", "pct_switchers"] if c in res.columns]
effect_cols = ["outcome", "model_type", "absolute_effect", "pvalue", "stat_significance"] + diag_cols

print("\n--- Fixed Effects Results ---")
print(res[effect_cols].to_string(index=False))
