# %% [markdown]
# # `plot_effects` gallery
#
# Synthetic multi-experiment data (country × type) to exercise absolute /
# relative annotations, grouping, meta-analysis rows, and single-experiment
# layouts.
#
# Run:
#
#     uv run python examples/plot_effects_example.py

# %%
import numpy as np
import pandas as pd

from experiment_utils import ExperimentAnalyzer

# %% [markdown]
# ## Simulate A/B experiments across country × type

# %%
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
    age = np.random.normal(35, 10, n)
    revenue = np.random.normal(50, 20, n) + revenue_lift * treatment
    converted = np.random.binomial(1, np.clip(0.12 + conversion_lift * treatment, 0, 1), n)
    return pd.DataFrame(
        {
            "country": country,
            "type": exp_type,
            "treatment": treatment,
            "revenue": revenue,
            "converted": converted,
            "age": age,
        }
    )


df = pd.concat(
    [make_experiment(c, t, n=700, revenue_lift=rl, conversion_lift=cl) for (c, t), (rl, cl) in LIFTS.items()],
    ignore_index=True,
)

# %% [markdown]
# ## Run analysis

# %%
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue", "converted"],
    experiment_identifier=["country", "type"],
    correction=None,
)
analyzer.get_effects()

print(
    analyzer.results[
        [
            "country",
            "type",
            "outcome",
            "absolute_effect",
            "relative_effect",
            "standard_error",
            "pvalue",
            "stat_significance",
        ]
    ].to_string(index=False)
)

# %% [markdown]
# ## Case 1 — absolute effects, sorted by magnitude (defaults)

# %%
analyzer.plot_effects(title="Treatment Effects — Revenue & Conversion")

# %% [markdown]
# ## Case 2 — conversion as percentage points

# %%
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — Percentage Points",
    pct_points=True,
)

# %% [markdown]
# ## Case 3 — combine absolute (pp) + relative %

# %%
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — Absolute (Relative) Effect",
    effect="absolute",
    pct_points=True,
    combine_values=True,
)

# %% [markdown]
# ## Case 4 — combine starting from relative

# %%
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — Relative (Absolute) Effect",
    effect="relative",
    pct_points=True,
    combine_values=True,
)

# %% [markdown]
# ## Case 5 — side-by-side absolute and relative panels

# %%
analyzer.plot_effects(
    title="Effects — Absolute & Relative Side-by-Side",
    effect=["absolute", "relative"],
    pct_points=True,
)

# %% [markdown]
# ## Case 6 — group by country + meta-analysis row

# %%
figs_by_country = analyzer.plot_effects(
    group_by="country",
    pct_points=True,
    combine_values=True,
    meta_analysis=True,
)

figs_by_country["EU"]

# %% [markdown]
# ## Case 7 — group by experiment type

# %%
figs_by_type = analyzer.plot_effects(
    group_by="type",
    pct_points=True,
)

figs_by_type["email"]

# %% [markdown]
# ## Case 8 — single outcome with pooled meta-analysis row

# %%
analyzer.plot_effects(
    outcomes="revenue",
    meta_analysis=True,
    combine_values=True,
    title="Revenue Effect — with Pooled Estimate",
)

# %% [markdown]
# ## Case 9 — outcomes on the y-axis (`y="outcome"`)

# %%
analyzer.plot_effects(
    y="outcome",
    title="Effects by Experiment",
)

# %% [markdown]
# ## Case 10 — single experiment, multiple outcomes

# %%
np.random.seed(99)
n_single = 1200
t_single = np.random.binomial(1, 0.5, n_single)
df_single = pd.DataFrame(
    {
        "experiment": "email_campaign",
        "treatment": t_single,
        "revenue": np.random.normal(50, 20, n_single) + 5.5 * t_single,
        "converted": np.random.binomial(1, np.clip(0.12 + 0.04 * t_single, 0, 1), n_single),
        "orders": np.random.poisson(2 + t_single, n_single),
        "sessions": np.random.normal(3, 1, n_single) + 0.3 * t_single,
    }
)

analyzer_single = ExperimentAnalyzer(
    data=df_single,
    treatment_col="treatment",
    outcomes=["revenue", "converted", "orders", "sessions"],
    experiment_identifier="experiment",
    correction=None,
)
analyzer_single.get_effects()

analyzer_single.plot_effects(
    y="outcome",
    title="Email Campaign Results",
    show_panel_titles=False,
)

# %% [markdown]
# ## Case 11 — single experiment, conversion pp + combine_values

# %%
analyzer_single.plot_effects(
    outcomes="converted",
    y="outcome",
    title="Email Campaign — Conversion (Absolute + Relative)",
    panel_titles="Treatment vs Control",
    pct_points=True,
    combine_values=True,
)
