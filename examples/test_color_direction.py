# %% Imports
import numpy as np
import pandas as pd

from experiment_utils import ExperimentAnalyzer

# %%
# Simulate data with mix of positive/negative, significant/not-significant effects
np.random.seed(42)

LIFTS = {
    ("US", "email"): (6.0, 0.05),
    ("US", "push"): (2.0, 0.01),
    ("US", "in-app"): (0.0, 0.00),
    ("EU", "email"): (-2.0, -0.02),
    ("EU", "push"): (4.5, 0.04),
    ("EU", "in-app"): (1.0, 0.01),
}


def make_experiment(country, exp_type, n, revenue_lift, conversion_lift):
    treatment = np.random.binomial(1, 0.5, n)
    revenue = np.random.normal(50, 20, n) + revenue_lift * treatment
    converted = np.random.binomial(1, np.clip(0.12 + conversion_lift * treatment, 0, 1), n)
    return pd.DataFrame(
        {
            "country": country,
            "type": exp_type,
            "treatment": treatment,
            "revenue": revenue,
            "converted": converted,
        }
    )


df = pd.concat(
    [make_experiment(c, t, n=700, revenue_lift=rl, conversion_lift=cl) for (c, t), (rl, cl) in LIFTS.items()],
    ignore_index=True,
)

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue", "converted"],
    experiment_identifier=["country", "type"],
    correction=None,
)
analyzer.get_effects()

# %%
# Case 1 — Default colors (no color_direction)
fig1 = analyzer.plot_effects(
    title="Default Colors",
    pct_points=True,
    combine_values=True,
    save_path="docs/assets/plot_default.png",
)

# %%
# Case 2 — color_direction=True (red/gray/light-green/dark-green)
fig2 = analyzer.plot_effects(
    title="Color Direction (red / gray / light green / dark green)",
    pct_points=True,
    combine_values=True,
    color_direction=True,
    save_path="docs/assets/plot_color_direction.png",
)

# %%
# Case 3 — color_direction + meta-analysis pooled row
figs = analyzer.plot_effects(
    group_by="country",
    pct_points=True,
    combine_values=True,
    meta_analysis=True,
    color_direction=True,
    save_path="docs/assets/plot_meta_color_direction.png",
)
