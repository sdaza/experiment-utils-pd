"""Simple precision comparison example.

Run with:

    python examples/precision_comparison.py

This shows how regression adjustment changes the standard error and precision of
the treatment effect estimate. With bootstrap=True, both the adjusted model and
the unadjusted reference are bootstrapped on the same resamples.
"""


# %% setup
import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()
from experiment_utils.experiment_analyzer import ExperimentAnalyzer


def make_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    treatment = rng.binomial(1, 0.5, n)
    age = rng.normal(38, 11, n)
    tenure = rng.exponential(2.5, n)
    pre_revenue = rng.normal(100, 25, n)

    revenue = (
        40
        + 4.0 * treatment
        + 0.15 * age
        + 1.5 * tenure
        + 0.75 * pre_revenue
        + rng.normal(0, 18, n)
    )

    return pd.DataFrame(
        {
            "experiment_id": 1,
            "treatment": treatment,
            "age": age,
            "tenure": tenure,
            "pre_revenue": pre_revenue,
            "revenue": revenue,
        }
    )


if __name__ == "__main__":
    df = make_data()

    analyzer = ExperimentAnalyzer(
        data=df,
        outcomes=["revenue"],
        treatment_col="treatment",
        experiment_identifier=["experiment_id"],
        regression_covariates=["age", "tenure", "pre_revenue"],
        bootstrap=True,
        bootstrap_iterations=200,
        bootstrap_seed=123,
    )

    analyzer.get_effects()

    columns = [
        "outcome",
        "adjustment",
        "absolute_effect",
        "unadjusted_absolute_effect",
        "standard_error",
        "unadjusted_standard_error",
        "standard_error_reduction",
        "precision_gain",
        "ci_width_reduction",
    ]

    with pd.option_context("display.max_columns", None, "display.width", 140):
        print(analyzer.precision_summary[columns].round(4))

# %%
