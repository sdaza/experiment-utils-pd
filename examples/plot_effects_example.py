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

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 7 — nested / grouped: multiple variants on the same outcome row
# Simulate an experiment with two treatment variants (V1, V2) and one control
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(99)

VARIANTS = {
    "control": (0.0, 0.0),
    "variant_A": (4.0, 0.03),
    "variant_B": (8.5, 0.06),
}

rows = []
for arm, (rl, cl) in VARIANTS.items():
    n = 600
    rows.append(
        pd.DataFrame(
            {
                "treatment": arm,
                "revenue": np.random.normal(50, 20, n) + rl,
                "converted": np.random.binomial(1, np.clip(0.12 + cl, 0, 1), n),
                "engagement": np.random.normal(3.0, 1.2, n) + rl * 0.05,
            }
        )
    )

df_multi = pd.concat(rows, ignore_index=True)

analyzer_multi = ExperimentAnalyzer(
    data=df_multi,
    treatment_col="treatment",
    outcomes=["revenue", "converted", "engagement"],
)
analyzer_multi.get_effects()

# color_by="treatment_group" puts both variants on the same outcome row
fig7 = analyzer_multi.plot_effects(
    y="outcome",
    color_by="treatment_group",
    show_labels=True,
    title="Variant A vs Variant B — side-by-side",
)
fig7.savefig("examples/plot_effects_color_by.png", bbox_inches="tight", dpi=150)
plt.close(fig7)

# %%
# ─────────────────────────────────────────────────────────────────────────────
# Case 8 — nested relative effects with two named variants, MCP correction,
#           and value labels (replicates the "Variants vs Control" style)
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(7)

# Baseline rates / counts per group (control arm)
BASELINE = {
    "total_leads": 12.5,
    "paid_leads": 6.0,
    "unpaid_leads": 6.5,
    "has_lead": 0.58,
    "has_paid_lead": 0.35,
    "has_unpaid_lead": 0.30,
    "has_multi_leads_per_cat": 0.22,
}

# (relative_lift_v1, relative_lift_v2) per outcome
LIFTS = {
    "total_leads": (0.004, 0.080),
    "paid_leads": (-0.055, 0.085),
    "unpaid_leads": (0.234, 0.057),
    "has_lead": (-0.009, 0.045),
    "has_paid_lead": (-0.056, 0.051),
    "has_unpaid_lead": (0.177, 0.042),
    "has_multi_leads_per_cat": (-0.008, 0.104),
}

COUNT_OUTCOMES = ["total_leads", "paid_leads", "unpaid_leads"]
PROP_OUTCOMES = ["has_lead", "has_paid_lead", "has_unpaid_lead", "has_multi_leads_per_cat"]
ALL_OUTCOMES = COUNT_OUTCOMES + PROP_OUTCOMES

n_per_arm = 4_000
arms = {"control": 0, "V1 (Proximity)": 1, "V2 (ML)": 2}

rows_nested = []
for arm_name, arm_idx in arms.items():
    rng = np.random.default_rng(seed=arm_idx * 100 + 7)
    d: dict = {"treatment": arm_name}
    for outcome in ALL_OUTCOMES:
        base = BASELINE[outcome]
        lift_v1, lift_v2 = LIFTS[outcome]
        if arm_name == "control":
            lift = 0.0
        elif arm_name == "V1 (Proximity)":
            lift = lift_v1
        else:
            lift = lift_v2
        effective = base * (1 + lift)
        if outcome in PROP_OUTCOMES:
            d[outcome] = rng.binomial(1, np.clip(effective, 0, 1), n_per_arm).astype(float)
        else:
            d[outcome] = rng.poisson(lam=max(effective, 0.01), size=n_per_arm).astype(float)
    rows_nested.append(pd.DataFrame(d))

df_nested = pd.concat(rows_nested, ignore_index=True)

analyzer_nested = ExperimentAnalyzer(
    data=df_nested,
    treatment_col="treatment",
    outcomes=ALL_OUTCOMES,
    pvalue_adjustment="bonferroni",
)
analyzer_nested.get_effects()

fig8 = analyzer_nested.plot_effects(
    y="outcome",
    effect="relative",
    color_by="treatment_group",
    show_labels=True,
    sort_by_magnitude=False,
    title="Variants vs Control\n* = significant after multiple comparisons correction",
    panel_titles="",
)
fig8.savefig("examples/plot_effects_nested_relative.png", bbox_inches="tight", dpi=150)
plt.close(fig8)

print("\nAll plots saved to examples/")
print("  plot_effects_basic.png")
print("  plot_effects_country_us.png  /  plot_effects_country_eu.png")
print("  plot_effects_type_email.png  /  plot_effects_type_push.png  /  plot_effects_type_in_app.png")
print("  plot_effects_meta.png")
print("  plot_effects_y_outcome.png")
print("  plot_effects_single_exp.png")
print("  plot_effects_color_by.png")
print("  plot_effects_nested_relative.png")
