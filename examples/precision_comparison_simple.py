# %% [markdown]
# # Precision comparison (minimal)
#
# Shortest example of automatic precision comparison after regression adjustment.
# Outcome depends strongly on `pre_spend`, so adjustment should reduce SE.
#
# Run:
#
#     uv run python examples/precision_comparison_simple.py

# %%
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

rng = np.random.default_rng(42)
n = 800

df = pd.DataFrame(
    {
        "experiment_id": 1,
        "treatment": rng.binomial(1, 0.5, n),
        "age": rng.normal(40, 10, n),
        "pre_spend": rng.normal(100, 25, n),
    }
)

df["revenue"] = 20 + 3.0 * df["treatment"] + 0.2 * df["age"] + 0.8 * df["pre_spend"] + rng.normal(0, 20, n)

# %% [markdown]
# ## Estimate effects and print precision summary

# %%
analyzer = ExperimentAnalyzer(
    data=df,
    outcomes=["revenue"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    regression_covariates=["age", "pre_spend"],
)

analyzer.get_effects()

summary_cols = [
    "outcome",
    "adjustment",
    "absolute_effect",
    "unadjusted_absolute_effect",
    "standard_error",
    "unadjusted_standard_error",
    "standard_error_reduction",
    "precision_gain",
]

print(analyzer.precision_summary[summary_cols].round(4))
