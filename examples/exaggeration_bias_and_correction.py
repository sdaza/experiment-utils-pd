# %% [markdown]
# # Exaggeration bias (Type M / winner's curse) and its correction
#
# When you only act on *statistically significant* results, the surviving
# estimates are systematically too big: significance selection favours the noisy
# draws that landed far from zero. This is the "winner's curse" / Type M error,
# and it is worst when power is low.
#
# This notebook:
# 1. Uses `PowerSim.simulate_retrodesign()` to show the theoretical exaggeration
#    ratio at different power levels.
# 2. Runs a Monte Carlo of many experiments, keeps the significant winners, and
#    measures the exaggeration empirically.
# 3. Corrects each winner with `winners_curse_estimate()` (single-estimate,
#    median-unbiased) and the whole family with `empirical_bayes_shrinkage()`.

# %%
import numpy as np
import pandas as pd

from experiment_utils import (
    ExperimentAnalyzer,
    PowerSim,
    empirical_bayes_shrinkage,
    winners_curse_estimate,
)

ALPHA = 0.05
Z_STAR = 1.959963984540054  # norm.ppf(1 - alpha/2) for alpha = 0.05

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# %% [markdown]
# ## 1. Theoretical exaggeration ratio vs. power
#
# A binary outcome at a 10% baseline with a true lift of +1pp. We grow the
# per-group sample size to drive power up and watch the exaggeration ratio fall
# back toward 1.0. `exaggeration_ratio = mean(|observed| / |true|)` among the
# significant results — an underpowered study reports effects 2-3x too large.

# %%
p = PowerSim(metric="proportion", variants=1, nsim=1000, alpha=ALPHA)

print(f"  {'n / arm':>8}  {'power':>6}  {'type_s':>7}  {'exagg.':>7}  {'rel_bias':>8}")
for n_per_arm in [800, 2000, 5000, 12000]:
    retro = p.simulate_retrodesign(
        true_effect=0.01,
        sample_size=n_per_arm,
        baseline=0.10,
        random_seed=7,
    )
    r = retro.iloc[0]
    print(
        f"  {n_per_arm:>8}  {r['power']:>6.2f}  {r['type_s_error']:>7.3f}  "
        f"{r['exaggeration_ratio']:>7.2f}  {r['relative_bias']:>8.2f}"
    )

# %% [markdown]
# ## 2. Monte Carlo of winners
#
# Fix a true effect and let each experiment draw a noisy estimate. We mimic an
# underpowered design: true effect ≈ 1.6 SE, so power ≈ 0.36. The mean of ALL
# estimates is unbiased; the mean of the *significant* ones is inflated.

# %%
TRUE_EFFECT = 0.40
SE = 0.25
N_EXPERIMENTS = 50000
rng = np.random.default_rng(123)

estimates = rng.normal(loc=TRUE_EFFECT, scale=SE, size=N_EXPERIMENTS)
significant = np.abs(estimates) >= Z_STAR * SE
winners = estimates[significant]

print(f"  power (fraction significant)        : {significant.mean():6.2%}")
print(f"  mean estimate, ALL experiments      : {estimates.mean():6.3f}  (≈ true {TRUE_EFFECT})")
print(f"  mean estimate, SIGNIFICANT winners  : {winners.mean():6.3f}")
print(f"  empirical exaggeration ratio        : {np.abs(winners).mean() / TRUE_EFFECT:6.2f}x")
print(f"  sign errors among winners (Type S)  : {(winners < 0).mean():6.2%}")

# %% [markdown]
# ## 3. Per-estimate correction with `winners_curse_estimate()`
#
# A median-unbiased estimate that conditions on the estimate having been
# selected by significance. First on a single representative winner, then
# applied across many to compare bias before vs. after.

# %%
example = float(np.median(winners[winners > 0]))
wc = winners_curse_estimate(effect=example, standard_error=SE, alpha=ALPHA, ci=0.95)
print(f"  Example winner: observed = {example:.3f}  (z = {example / SE:.2f})")
print(f"    corrected (median-unbiased) = {wc['corrected']:.3f}")
print(f"    selection-adjusted 95% CI   = [{wc['ci_lower']:.3f}, {wc['ci_upper']:.3f}]")
print(f"    shrinkage factor            = {wc['shrinkage']:.2f}")

# %%
sample = winners[:2000]
corrected = np.array([winners_curse_estimate(effect=e, standard_error=SE, alpha=ALPHA)["corrected"] for e in sample])
print("  Across 2,000 winners:")
print(f"    mean |observed|   = {np.abs(sample).mean():.3f}   (true = {TRUE_EFFECT})")
print(f"    mean |corrected|  = {np.abs(corrected).mean():.3f}")
print(f"    exaggeration before = {np.abs(sample).mean() / TRUE_EFFECT:.2f}x")
print(f"    exaggeration after  = {np.abs(corrected).mean() / TRUE_EFFECT:.2f}x")

# %% [markdown]
# ## 4. Joint shrinkage across a portfolio with `empirical_bayes_shrinkage()`
#
# When you have several significant estimates (a portfolio of experiments),
# learn a shared prior and shrink them jointly. The noisiest winners (largest
# SE) shrink the most. Report the shrunk values, not the raw ones.

# %%
portfolio_effects = np.array([0.62, 0.55, 0.70, 0.48, 0.90])
portfolio_ses = np.array([0.25, 0.24, 0.26, 0.23, 0.30])

eb = empirical_bayes_shrinkage(portfolio_effects, portfolio_ses, prior_mean=0.0, ci=0.95)
print(f"  learned prior variance tau^2 = {eb['tau2']:.4f}\n")
print(f"  {'observed':>9}  {'se':>5}  {'shrunk':>7}  {'shrink_f':>8}  {'95% CI':>20}")
for obs, se, sh, sf, lo, hi in zip(
    portfolio_effects, portfolio_ses, eb["shrunk"], eb["shrinkage_factor"], eb["ci_lower"], eb["ci_upper"], strict=False
):
    print(f"  {obs:>9.3f}  {se:>5.2f}  {sh:>7.3f}  {sf:>8.2f}  [{lo:>6.3f}, {hi:>6.3f}]")

# %% [markdown]
# ## 5. End-to-end with `ExperimentAnalyzer`
#
# The same corrections are built into the analyzer, so you can go straight from
# raw assignment data to de-biased effects. We simulate a *portfolio* of 6
# underpowered experiments — each a binary outcome on a 10% baseline with a true
# +2pp lift (≈3,000 users/arm, so individually underpowered). Significant winners
# will overstate that +2pp.

# %%
TRUE_LIFT = 0.02
BASELINE = 0.10
N_PER_ARM = 3000
rng_exp = np.random.default_rng(2024)

frames = []
for exp_id in range(1, 7):
    n = 2 * N_PER_ARM
    treatment = rng_exp.binomial(1, 0.5, n)
    outcome = rng_exp.binomial(1, BASELINE + TRUE_LIFT * treatment, n)
    frames.append(pd.DataFrame({"experiment_id": exp_id, "treatment": treatment, "converted": outcome}))

ea = ExperimentAnalyzer(
    data=pd.concat(frames, ignore_index=True),
    outcomes=["converted"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    alpha=ALPHA,
)
ea.get_effects()
print(ea.results[["experiment_id", "absolute_effect", "standard_error", "pvalue", "stat_significance"]].round(4))

# %% [markdown]
# ### 5a. `calculate_retrodesign()` — power, Type S, Type M per significant result
#
# Filters to the significant winners and reports the retrodesign metrics, using
# the observed SE. Pass the design truth as `true_effect` (here +2pp); omit it to
# fall back to the observed effect (conservative).

# %%
retro = ea.calculate_retrodesign(true_effect={"converted": TRUE_LIFT}, nsim=2000, seed=1)
if not retro.empty:
    print(retro[["experiment_id", "absolute_effect", "true_effect", "power", "type_s_error", "type_m_error"]].round(4))
else:
    print("  (no statistically significant effects this run)")

# %% [markdown]
# ### 5b. `winners_curse_summary(method="conditional")` — de-bias each winner
#
# Median-unbiased correction with a selection-adjusted CI for every significant
# row. `corrected_effect` pulls each inflated winner back toward the truth.

# %%
wc_cond = ea.winners_curse_summary(method="conditional")
if not wc_cond.empty:
    cols = [
        "experiment_id",
        "absolute_effect",
        "corrected_effect",
        "corrected_ci_lower",
        "corrected_ci_upper",
        "shrinkage",
    ]
    print(wc_cond[cols].round(4))
else:
    print("  (no statistically significant rows to correct)")

# %% [markdown]
# ### 5c. `winners_curse_summary(method="empirical_bayes")` — joint shrinkage
#
# Shrinks ALL six estimates toward 0 within the `(outcome, effect_type)` group
# (needs ≥3 estimates to learn the prior). Unlike the conditional correction,
# this also adjusts non-significant rows and borrows strength across the family.

# %%
wc_eb = ea.winners_curse_summary(method="empirical_bayes", group_by="outcome")
print(wc_eb[["experiment_id", "absolute_effect", "corrected_effect", "shrinkage", "tau2"]].round(4))

# %%
