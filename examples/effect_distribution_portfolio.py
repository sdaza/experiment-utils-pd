# %% [markdown]
# # The effect distribution of an experimentation program
#
# Based on Datadog's "Effect distribution in experimentation"
# (https://www.datadoghq.com/blog/effect-distribution-in-experimentation/):
# the distribution of *true* effects across a portfolio of experiments is not
# the distribution of *observed* effects — observed effects are the truth plus
# sampling noise, so they are systematically wider, and the biggest observed
# winners are almost always exaggerated.
#
# This example simulates a portfolio of experiments and shows how the package's
# tools recover program-level quantities:
#
# 1. **Deconvolution**: estimate the true-effect spread from noisy observed
#    effects (`empirical_bayes_shrinkage`'s `tau2` is exactly
#    var(observed) - mean(se^2), solved by Paule-Mandel).
# 2. **Winner's curse at the portfolio level**: empirical-Bayes shrinkage pulls
#    the top observed effects back toward what is plausible for the program.
# 3. **Contextualizing the win rate**: what fraction of experiments had a real
#    effect, given the observed significance rate (`estimate_true_success_rate`).
# 4. **The value of experimentation (EVSI)**: expected shipped uplift under
#    "ship if significant" vs "ship everything" vs perfect information.

# %%
import numpy as np
from scipy import stats

from experiment_utils import empirical_bayes_shrinkage, winners_curse_estimate
from experiment_utils.utils import estimate_true_success_rate

rng = np.random.default_rng(42)

ALPHA = 0.05
Z_STAR = stats.norm.ppf(1 - ALPHA / 2)

# Portfolio: 400 conversion experiments at a 10% baseline, 8k users per arm.
# True effects are a mixture: 70% do nothing, 30% draw from a zero-centered
# N(0, 1pp) — real winners and real losers in equal measure, mostly nulls.
N_EXPERIMENTS = 400
N_PER_ARM = 8_000
BASELINE = 0.10
PI_NULL = 0.70

se = np.sqrt(2 * BASELINE * (1 - BASELINE) / N_PER_ARM)  # SE of the pp difference

is_null = rng.random(N_EXPERIMENTS) < PI_NULL
true_effects = np.where(is_null, 0.0, rng.normal(0.0, 0.010, N_EXPERIMENTS))
observed = true_effects + rng.normal(0, se, N_EXPERIMENTS)
significant = np.abs(observed) > Z_STAR * se

print(f"SE per experiment:        {se:.4f} ({se * 100:.2f}pp)")
print(f"sd(true effects):         {np.std(true_effects):.4f}")
print(f"sd(observed effects):     {np.std(observed):.4f}  <- wider: truth + noise")
print(f"win rate (significant):   {significant.mean():.1%}")

# %% [markdown]
# ## 1. Deconvolution: how spread out are the *true* effects?
#
# var(observed) = var(true) + mean(se^2), so the true-effect variance can be
# recovered by subtracting the known sampling variance. `empirical_bayes_shrinkage`
# does this robustly (Paule-Mandel) and returns it as `tau2`.

# %%
eb = empirical_bayes_shrinkage(
    effects=observed,
    standard_errors=np.full(N_EXPERIMENTS, se),
    prior_mean=0.0,
)
tau = np.sqrt(eb["tau2"])
print(f"estimated sd of true effects (sqrt tau2): {tau:.4f}")
print(f"actual sd of true effects:                {np.std(true_effects):.4f}")

# %% [markdown]
# ## 2. Winner's curse at the portfolio level
#
# The largest observed effects are almost always exaggerated. Shrinking each
# estimate toward the program mean (weight tau2 / (tau2 + se^2)) gives estimates
# that are much closer to the truth — especially for the "winners".

# %%
rmse_observed = np.sqrt(np.mean((observed - true_effects) ** 2))
rmse_shrunk = np.sqrt(np.mean((eb["shrunk"] - true_effects) ** 2))
print(f"RMSE of observed effects: {rmse_observed:.4f}")
print(f"RMSE of shrunk effects:   {rmse_shrunk:.4f}  <- closer to the truth")

top = np.argsort(-observed)[:5]
print("\nTop-5 observed winners (pp):")
print(f"  {'observed':>9} {'shrunk':>9} {'true':>9}")
for i in top:
    print(f"  {observed[i] * 100:>8.2f}  {eb['shrunk'][i] * 100:>8.2f}  {true_effects[i] * 100:>8.2f}")

# The truncated-normal correction (winners_curse_estimate) targets a different
# regime: it barely moves extreme winners like the ones above (selection hardly
# distorts a z of 6), but it bites hard exactly where selection bias is worst —
# the barely significant winner:
barely = np.argmin(np.where(significant & (observed > 0), observed, np.inf))
wc = winners_curse_estimate(float(observed[barely]), float(se), alpha=ALPHA)["corrected"]
print(
    f"\nSmallest significant winner: observed {observed[barely] * 100:.2f}pp, "
    f"wc-corrected {wc * 100:.2f}pp, true {true_effects[barely] * 100:.2f}pp"
)

# %% [markdown]
# ## 3. Contextualizing the win rate
#
# A 5% significance rate can mean "all noise" or "healthy program" depending on
# the effect distribution. Given the observed win rate, alpha, and the average
# power against *real* effects, `estimate_true_success_rate` backs out the
# fraction of experiments that had a real effect. The answer is sensitive to the
# power assumption — treat it as a scenario analysis, not a point estimate.

# %%
win_rate = float(significant.mean())

# Actual average power against the non-null effects in this simulation, for reference:
nonnull = np.abs(true_effects[~is_null]) / se
actual_avg_power = float(np.mean(stats.norm.sf(Z_STAR - nonnull) + stats.norm.cdf(-Z_STAR - nonnull)))
print(f"actual average power vs real effects: {actual_avg_power:.2f}")
print(f"actual share with real effects:       {(~is_null).mean():.1%}\n")

print(f"  {'assumed power':>13}  {'estimated share with real effects':>34}")
for assumed_power in (0.3, round(actual_avg_power, 2), 0.7):
    est = estimate_true_success_rate(win_rate=win_rate, alpha=ALPHA, power=assumed_power)
    print(f"  {assumed_power:>13.2f}  {est:>33.1%}")

# %% [markdown]
# ## 4. The expected value of experimentation (EVSI)
#
# Simulate the decision layer: what average uplift per initiative do you bank
# under each policy? With a zero-centered effect distribution, shipping
# everything banks ~0 — experimentation earns its keep not by finding winners
# but by *filtering out* the losers you would otherwise ship. (If your program's
# true effects skew positive and power is low, this gap can invert: the filter
# then rejects more real winners than it blocks losers — worth checking per
# experiment category.)

# %%
ship_all = true_effects.mean()
ship_if_sig = np.where(significant & (observed > 0), true_effects, 0.0).mean()
oracle = np.where(true_effects > 0, true_effects, 0.0).mean()  # perfect information

print("Average banked uplift per initiative (pp):")
print(f"  ship everything (no experimentation): {ship_all * 100:>6.3f}")
print(f"  ship if significant & positive:       {ship_if_sig * 100:>6.3f}")
print(f"  perfect information (upper bound):    {oracle * 100:>6.3f}")
print(f"\nEVSI of the program: {(ship_if_sig - ship_all) * 100:.3f}pp per initiative")

# %% [markdown]
# Program-level takeaways:
# - The observed effect distribution overstates the true one; deconvolve before
#   drawing conclusions about "how big our effects are".
# - Report shrunk (or winner's-curse-corrected) estimates for shipped winners;
#   raw observed winners are upper bounds, not expectations.
# - The gap between "ship if significant" and "perfect information" is the room
#   left for better-powered designs — compare it across experiment categories
#   to decide where the next test is worth running.
