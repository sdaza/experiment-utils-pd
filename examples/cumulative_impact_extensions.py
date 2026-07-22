# %% [markdown]
# # Cumulative impact extensions
#
# Compare four ways to estimate the **total effect of launched experiments**:
#
# 1. **Naive** — sum raw lifts of winners (winner's curse).
# 2. **Airbnb process-level** (`process_level_total_effect`) — Lee & Shen \(\hat T_A\).
# 3. **Kessler / Datadog EB** (`cumulative_impact`) — shrink-then-sum; prior from
#    MoM or Half-Cauchy **MAP** (`fit_normal_prior_map` / `prior="map"`).
# 4. **Joint + estimated ρ** — ship with a guardrail rule; shrink primary using
#    `estimate_guardrail_rho` + `joint_metric_shrinkage`.

# %%
import numpy as np
from scipy.stats import norm

from experiment_utils import (
    cumulative_impact,
    estimate_guardrail_rho,
    fit_normal_prior_map,
    joint_metric_shrinkage,
    process_level_total_effect,
)

rng = np.random.default_rng(7)
N = 500
TAU = 0.018
MU = 0.004
SE_P = 0.022
SE_G = 0.012
RHO = 0.35
Z2 = norm.ppf(0.975)

delta = rng.normal(MU, TAU, N)
gamma = RHO * (delta - MU) + MU + TAU * np.sqrt(1 - RHO**2) * rng.standard_normal(N)
x = delta + rng.normal(0.0, SE_P, N)
g = gamma + rng.normal(0.0, SE_G, N)

# Real launch rule: two-sided primary win AND guardrail non-negative
primary_win = np.abs(x) > Z2 * SE_P
ship = (x > Z2 * SE_P) & (g >= 0.0)  # positive primary + guardrail
true_cum = float(delta[ship].sum())

print(f"n={N}  shipped={ship.sum()}  primary_wins={primary_win.sum()}")
print(f"true cumulative Δ (shipped): {true_cum:+.4f}\n")

# %% [markdown]
# ## 1–2. Naive vs Airbnb process-level (one-sided launch on primary only)

# %%
# Process-level uses its own one-sided selection on the primary (paper setup).
airbnb = process_level_total_effect(x, np.full(N, SE_P), alpha=0.05, alternative="greater")
# For a fair process-level comparison, true total under *that* selection:
true_airbnb = float(delta[airbnb["selected_mask"]].sum())
print("Airbnb selection (one-sided primary only):")
print(f"  naive S_A:           {airbnb['naive_total']:+.4f}")
print(f"  conditional total:   {airbnb['conditional_total']:+.4f}")
print(f"  process-level T̂_A:  {airbnb['total']:+.4f}")
print(f"  true under Airbnb A: {true_airbnb:+.4f}")
print(
    f"  errors vs true: naive {100 * (airbnb['naive_total'] / true_airbnb - 1):+.0f}%, "
    f"cond {100 * (airbnb['conditional_total'] / true_airbnb - 1):+.0f}%, "
    f"T̂_A {100 * (airbnb['total'] / true_airbnb - 1):+.0f}%\n"
)

# %% [markdown]
# ## 3. Kessler shrink-then-sum on the *real* ship set (MoM vs MAP prior)

# %%
naive_ship = float(x[ship].sum())
mom = cumulative_impact(x, np.full(N, SE_P), shipped=ship, tau2=TAU**2, prior_mean=MU)
learned = cumulative_impact(x, np.full(N, SE_P), shipped=ship)
mapped = cumulative_impact(x, np.full(N, SE_P), shipped=ship, prior="map")
fit = fit_normal_prior_map(x, np.full(N, SE_P))

print("Kessler / Datadog on real ship set (primary+guardrail):")
print(f"  naive sum(X):        {naive_ship:+.4f}  (err {100 * (naive_ship / true_cum - 1):+.0f}%)")
print(f"  EB known τ,μ:        {mom['cumulative']:+.4f}  (err {100 * (mom['cumulative'] / true_cum - 1):+.0f}%)")
print(
    f"  EB MoM learned:      {learned['cumulative']:+.4f}  (err {100 * (learned['cumulative'] / true_cum - 1):+.0f}%)"
)
print(
    f"  EB Half-Cauchy MAP:  {mapped['cumulative']:+.4f}  "
    f"(τ̂={np.sqrt(fit['tau2']):.4f}, μ̂={fit['prior_mean']:.4f}; "
    f"err {100 * (mapped['cumulative'] / true_cum - 1):+.0f}%)"
)
print(f"  CI (MAP):            [{mapped['ci_lower']:+.4f}, {mapped['ci_upper']:+.4f}]\n")

# %% [markdown]
# ## 4. Joint shrinkage with **estimated** ρ

# %%
rho_hat = estimate_guardrail_rho(x, np.full(N, SE_P), g, np.full(N, SE_G))
print(
    f"estimated ρ={rho_hat['rho']:.3f} (true {RHO}), τ_p={rho_hat['tau_primary']:.4f}, τ_g={rho_hat['tau_guard']:.4f}"
)

joint = joint_metric_shrinkage(
    x,
    np.full(N, SE_P),
    g,
    np.full(N, SE_G),
    rho=rho_hat["rho"],
    prior_sd_primary=rho_hat["tau_primary"],
    prior_sd_guard=rho_hat["tau_guard"],
)
joint_cum = float(joint["primary_shrunk"][ship].sum())
print(f"joint cumulative:      {joint_cum:+.4f}  (err {100 * (joint_cum / true_cum - 1):+.0f}%)")

print(
    "\nTakeaway: define `shipped` with the real rule; prefer shrink-then-sum "
    "(MoM or MAP). Use Airbnb T̂_A when the estimand is process-level E[T_A]. "
    "Estimate ρ when guardrails should inform the primary posterior."
)

# %%
