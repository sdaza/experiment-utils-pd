# %% [markdown]
# # Cumulative impact of shipped experiments
#
# Summing raw lifts of "winners" overstates program impact (winner's curse).
# Kessler (2024) / Datadog: shrink every experiment with an archive prior, then
# aggregate **only the shipped set**. This script compares naive vs EB vs
# Student-t cumulative impact, and shows how guardrails change the ship set.
#
# Run:
#
#     uv run python examples/cumulative_impact.py

# %%
import numpy as np
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    fit_t_prior,
    joint_metric_shrinkage,
)

rng = np.random.default_rng(42)
N = 400
TAU = 0.015
SE_PRIMARY = 0.025
SE_GUARD = 0.012
RHO = 0.3
Z = norm.ppf(0.975)

delta = TAU * rng.standard_normal(N)
gamma = RHO * delta + TAU * np.sqrt(1.0 - RHO**2) * rng.standard_normal(N)
x = delta + rng.normal(0.0, SE_PRIMARY, N)
g = gamma + rng.normal(0.0, SE_GUARD, N)

primary_win = x > Z * SE_PRIMARY
ship = primary_win & (g >= 0.0)
true_cum = float(delta[ship].sum())
naive_cum = float(x[ship].sum())

print(f"shipped {ship.sum()} / {N}  (primary wins {primary_win.sum()})")
print(f"true cumulative Δ among shipped: {true_cum:+.4f}")
print(f"naive sum of X among shipped:    {naive_cum:+.4f}  (error {100 * (naive_cum / true_cum - 1):+.0f}%)")

# %% [markdown]
# ## Shrink-then-sum with a known / learned prior

# %%
eb = cumulative_impact(
    x,
    np.full(N, SE_PRIMARY),
    shipped=ship,
    tau2=TAU**2,  # oracle prior variance
    aggregation="sum",
)
print(
    f"EB cumulative (known τ²):          {eb['cumulative']:+.4f}  "
    f"[{eb['ci_lower']:+.4f}, {eb['ci_upper']:+.4f}]  "
    f"(error {100 * (eb['cumulative'] / true_cum - 1):+.0f}%)"
)

learned = cumulative_impact(x, np.full(N, SE_PRIMARY), shipped=ship)
print(f"EB cumulative (learned τ²={learned['tau2']:.2e}): {learned['cumulative']:+.4f}")

t_prior = fit_t_prior(x, np.full(N, SE_PRIMARY), df=4.0)
t_cum = cumulative_impact(x, np.full(N, SE_PRIMARY), shipped=ship, prior=t_prior)
print(f"t-prior cumulative:                {t_cum['cumulative']:+.4f}")

# %% [markdown]
# ## Optional: joint primary + guardrail shrinkage before aggregating
#
# When |ρ| is material, update the primary with the guardrail, then sum the
# joint primary posteriors over the same ship set.

# %%
joint = joint_metric_shrinkage(
    x,
    np.full(N, SE_PRIMARY),
    g,
    np.full(N, SE_GUARD),
    rho=RHO,
    prior_sd_primary=TAU,
)
joint_cum = float(joint["primary_shrunk"][ship].sum())
print(f"joint (X,G) cumulative:            {joint_cum:+.4f}  (error {100 * (joint_cum / true_cum - 1):+.0f}%)")
print(
    "\nTakeaway: always define `shipped` with the real launch rule; "
    "shrink with an archive prior; add joint updating only when |ρ| matters."
)

# %%
