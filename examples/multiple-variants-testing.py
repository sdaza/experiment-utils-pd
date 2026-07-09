# %% [markdown]
# # Multiple variants vs a shared control: choosing the correction
#
# Design: control + 3 variants, proportion metric, baseline 10%, true lift +5%
# RELATIVE (0.10 -> 0.105) on every variant. Family = control vs each variant.
#
# Pick the correction that matches the decision the test feeds:
#   * ship every variant that beats control  -> dunnett (FWER, exploits shared control)
#   * pick the best variant (B vs C claims)  -> tukey   (FWER over all pairs)
#   * screening, winners confirmed later     -> fdr     (BH; cheapest, weaker guarantee)
#   * generic / conservative                 -> sidak, bonferroni, holm, hochberg

# %% setup
from experiment_utils import PowerSim

ps = PowerSim(
    metric="proportion",
    relative_effect=True,
    variants=3,
    nsim=10000,
    alpha=0.05,
    correction="dunnett",
    alternative="two-tailed",
    early_stopping=False,
    comparisons=[(0, 1), (0, 2), (0, 3)],  # the family corrections are applied across
)

# %% power at a given design
power_df = ps.get_power(
    baseline=0.10,
    effect=0.05,
    sample_size=75433,
)
print(power_df)

# %% [markdown]
# ## Required sample size under each correction
#
# Same instance, per-call `correction` override. Stronger guarantees cost more
# sample: none < fdr < dunnett < sidak < bonferroni.
#
# Note on search precision: with nsim=5000 the power estimate has SE ~ 0.006,
# so `tolerance` below ~0.01 or `step_size` below ~1000 buys precision the
# simulation cannot resolve.

# %%
results = {}
for corr in ["none", "fdr", "dunnett", "sidak", "bonferroni"]:
    res = ps.find_sample_size(
        baseline=0.10,
        effect=0.05,
        power=0.8,
        min_sample_size=10000,
        max_sample_size=500000,
        step_size=2000,
        tolerance=0.001,        
        correction=corr,
    )
    results[corr] = res.iloc[0]["total_sample_size"]

for corr, n in results.items():
    print(f"{corr:12s}: total {n:,}")

# %% [markdown]
# ## FDR variants
#
# `fdr` is Benjamini-Hochberg: it bounds the SHARE of false discoveries (<= 5%
# of what you ship is a dud, on average), not the probability of any false
# positive — that's why it costs almost nothing here, where all effects are
# real. `fdr_method="negcorr"` switches to Benjamini-Yekutieli (valid under
# arbitrary dependence, notably more conservative).

# %%
ps.find_sample_size(
    baseline=0.10,
    effect=0.05,
    power=0.8,
    min_sample_size=10000,
    max_sample_size=500000,
    # step_size=2000,
    correction="fdr",
    fdr_method="negcorr",  # Benjamini-Yekutieli
)

# %% [markdown]
# ## Tukey: when the decision involves variant-vs-variant claims
#
# Tukey HSD protects ALL pairwise comparisons — needed if the plan is "scale
# the best variant" (an implicit B-vs-C claim). It requires the all-pairwise
# family, so build a separate instance without restricting `comparisons`.
# Watch the variant-vs-variant power: with equal true lifts it is ~alpha, and
# with realistic small gaps it stays low at any affordable sample size — if
# pick-the-best is the real decision, that number should drive the design.

# %%
ps_pairwise = PowerSim(
    metric="proportion",
    relative_effect=True,
    variants=3,
    nsim=5000,
    alpha=0.05,
    correction="tukey",
    alternative="two-tailed",
    early_stopping=False,  # family defaults to all 6 pairwise comparisons
)
print(ps_pairwise.get_power(baseline=0.10, effect=[0.12, 0.10, 0.08], sample_size=75433))
