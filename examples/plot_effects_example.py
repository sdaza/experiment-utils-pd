# %% Imports
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — swap to "TkAgg" or remove for interactive use
import numpy as np
import pandas as pd

from experiment_utils import ExperimentAnalyzer

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Simulate A/B experiments across country × type combinations
# ─────────────────────────────────────────────────────────────────────────────
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

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Run analysis
# ─────────────────────────────────────────────────────────────────────────────
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
        ["country", "type", "outcome", "absolute_effect", "relative_effect", "standard_error", "pvalue", "stat_significance"]
    ].to_string(index=False)
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — absolute effects, sorted by magnitude
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    title="Treatment Effects — Revenue & Conversion",
    sort_by_magnitude=True,
    show_values=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — conversion rate in percentage points (pct_points=True)
#           values shown as "+3.0pp" instead of "+0.03"
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — Percentage Points",
    pct_points=True,
    show_values=True,
    sort_by_magnitude=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — combined label: absolute (pp) with relative % in parentheses
#           "+3.0pp (+15.4%)" in a single absolute-effect panel
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — pp with Relative in Parentheses",
    effect="absolute",
    pct_points=True,
    show_values=True,
    combined_label=True,
    sort_by_magnitude=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — combined label starting from relative:
#           "+15.4% (+3.0pp)" — relative panel, absolute in parentheses
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    outcomes="converted",
    title="Conversion Rate — Relative with pp in Parentheses",
    effect="relative",
    pct_points=True,
    show_values=True,
    combined_label=True,
    sort_by_magnitude=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — side-by-side absolute (pp) and relative panels
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    title="Conversion Rate — Absolute (pp) & Relative Side-by-Side",
    effect=["absolute", "relative"],
    pct_points=True,
    show_values=True,
    sort_by_magnitude=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 6 — group by country (one figure per country, rows = type)
#           with pooled meta-analysis row
# ─────────────────────────────────────────────────────────────────────────────
figs_by_country = analyzer.plot_effects(
    group_by="country",
    pct_points=True,
    show_values=True,
    sort_by_magnitude=True,
    meta_analysis=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 7 — group by type (one figure per experiment type, rows = country)
# ─────────────────────────────────────────────────────────────────────────────
figs_by_type = analyzer.plot_effects(
    group_by="type",
    pct_points=True,
    show_values=True,
    sort_by_magnitude=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 8 — single outcome with pooled meta-analysis row
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    outcomes="revenue",
    meta_analysis=True,
    show_values=True,
    sort_by_magnitude=True,
    title="Revenue Effect — with Pooled Estimate",
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 9 — y="outcome": outcomes on the y-axis, experiments as panels
#           useful when there are many outcomes and few experiments
# ─────────────────────────────────────────────────────────────────────────────
analyzer.plot_effects(
    y="outcome",
    title="Effects by Experiment",
    show_values=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 10 — single experiment with multiple outcomes
#            panel_titles overrides the auto subplot title
# ─────────────────────────────────────────────────────────────────────────────
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
    panel_titles="Treatment vs Control",
    show_values=True,
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 11 — single experiment, conversion in pp + combined label
# ─────────────────────────────────────────────────────────────────────────────
analyzer_single.plot_effects(
    outcomes="converted",
    y="outcome",
    title="Email Campaign — Conversion (pp + relative)",
    panel_titles="Treatment vs Control",
    pct_points=True,
    show_values=True,
    combined_label=True,
)

print("\nAll examples rendered (no files saved).")
