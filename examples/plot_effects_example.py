# %% Imports
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — swap to "TkAgg" or remove for interactive use
import matplotlib.pyplot as plt
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
    pvalue_adjustment=None,
)
analyzer.get_effects()

print(
    analyzer.results[
        ["country", "type", "outcome", "absolute_effect", "standard_error", "pvalue", "stat_significance"]
    ].to_string(index=False)
)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 1 — all experiments, sorted by magnitude
# ─────────────────────────────────────────────────────────────────────────────
fig1 = analyzer.plot_effects(
    title="Treatment Effects — Revenue & Conversion",
    sort_by_magnitude=True,
    show_labels=True,
)
fig1.savefig("examples/plot_effects_basic.png", bbox_inches="tight", dpi=150)
plt.close(fig1)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 2 — group by country (one figure per country, rows = type)
# ─────────────────────────────────────────────────────────────────────────────
figs_by_country = analyzer.plot_effects(
    group_by="country",
    sort_by_magnitude=True,
    meta_analysis=True,
)
for country, fig in figs_by_country.items():
    fname = f"examples/plot_effects_country_{country.lower()}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 3 — group by type (one figure per experiment type, rows = country)
# ─────────────────────────────────────────────────────────────────────────────
figs_by_type = analyzer.plot_effects(
    group_by="type",
    sort_by_magnitude=True,
)
for exp_type, fig in figs_by_type.items():
    fname = f"examples/plot_effects_type_{exp_type.replace('-', '_')}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 4 — single outcome with pooled meta-analysis row
# ─────────────────────────────────────────────────────────────────────────────
fig4 = analyzer.plot_effects(
    outcomes="revenue",
    meta_analysis=True,
    sort_by_magnitude=True,
    title="Revenue Effect — with Pooled Estimate",
)
fig4.savefig("examples/plot_effects_meta.png", bbox_inches="tight", dpi=150)
plt.close(fig4)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 5 — y="outcome": outcomes on the y-axis, experiments as panels
#           useful when there are many outcomes and few experiments
# ─────────────────────────────────────────────────────────────────────────────
fig5 = analyzer.plot_effects(
    y="outcome",
    title="Effects by Experiment",
)
fig5.savefig("examples/plot_effects_y_outcome.png", bbox_inches="tight", dpi=150)
plt.close(fig5)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 6 — single experiment with multiple outcomes (most common y="outcome" use)
#           panel_titles overrides the auto subplot title
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
    pvalue_adjustment=None,
)
analyzer_single.get_effects()

fig6 = analyzer_single.plot_effects(
    y="outcome",
    title="Email Campaign Results",
    panel_titles="Treatment vs Control",
)
fig6.savefig("examples/plot_effects_single_exp.png", bbox_inches="tight", dpi=150)
plt.close(fig6)

print("\nAll plots saved to examples/")
print("  plot_effects_basic.png")
print("  plot_effects_country_us.png  /  plot_effects_country_eu.png")
print("  plot_effects_type_email.png  /  plot_effects_type_push.png  /  plot_effects_type_in_app.png")
print("  plot_effects_meta.png")
print("  plot_effects_y_outcome.png")
print("  plot_effects_single_exp.png")
