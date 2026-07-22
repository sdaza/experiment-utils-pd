# %% [markdown]
# # SCALED ship rule — which shrinker is better?
#
# Ship rule (SCALED-style):
#
# ```text
# ship = (primary significant, positive) AND (guardrail estimate >= 0)
# ```
#
# Monte Carlo under a known DGP. For each method we score:
#
# 1. **Average estimate** among shipped — MAE of posterior means vs true Δ
# 2. **Cumulative impact** — MAE / bias of the sum of shipped lifts
#
# Methods compared (all use the same `ship` mask):
#
# | label | what |
# |---|---|
# | naive | raw primary \(X\) |
# | EB | `cumulative_impact` / normal EB (oracle τ²) |
# | t-prior | Student-t (df=4) fit on archive, then sum shipped |
# | MAP | `cumulative_impact(..., prior="map")` |
# | NSS ρ | `nss_adjusted_cumulative_impact` with true ρ |
# | NSS ρ̂ | same with `estimate_guardrail_rho` |
#
# Run:
#
#     uv run python examples/scaled_ship_method_assessment.py

# %%
import numpy as np
import pandas as pd
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    estimate_guardrail_rho,
    fit_t_prior,
    nss_adjusted_cumulative_impact,
    t_prior_shrinkage,
)

# --- DGP / design (edit to match your noise / archive size) ---
N_EXP = 120
N_SIM = 250
TAU = 0.015
SE_P = 0.025
SE_G = 0.010  # precise guardrail → room for joint to help when |ρ| large
Z_ONE = norm.ppf(0.95)  # one-sided α=0.05 primary win
GUARD_FLOOR = 0.0
DF_T = 4.0

SCENARIOS = {
    "rho=0.0": 0.0,
    "rho=+0.3": 0.3,
    "rho=+0.5": 0.5,
    "rho=+0.75": 0.75,
    "rho=-0.3": -0.3,
}

METHODS = ("naive", "EB", "t-prior", "MAP", "NSS ρ", "NSS ρ̂")


def ship_mask(x: np.ndarray, g: np.ndarray, se_p: float) -> np.ndarray:
    """Primary significant (one-sided greater) and guardrail not negative."""
    return (x > Z_ONE * se_p) & (g >= GUARD_FLOOR)


def one_portfolio(rng: np.random.Generator, rho: float) -> dict | None:
    delta = TAU * rng.standard_normal(N_EXP)
    gamma = rho * delta + TAU * np.sqrt(max(1.0 - rho**2, 0.0)) * rng.standard_normal(N_EXP)
    x = delta + rng.normal(0.0, SE_P, N_EXP)
    g = gamma + rng.normal(0.0, SE_G, N_EXP)
    ship = ship_mask(x, g, SE_P)
    if int(ship.sum()) < 2:
        return None

    se_p = np.full(N_EXP, SE_P)
    se_g = np.full(N_EXP, SE_G)
    true = delta[ship]
    true_mean = float(true.mean())
    true_sum = float(true.sum())

    # --- row-wise / cumulative estimates on the same ship set ---
    naive_ship = x[ship]
    eb = cumulative_impact(x, se_p, shipped=ship, tau2=TAU**2, prior_mean=0.0)
    t_fit = fit_t_prior(x, se_p, df=DF_T)
    t_row = t_prior_shrinkage(x, se_p, scale=t_fit["scale"], df=t_fit["df"], prior_mean=0.0)
    t_cum = float(t_row["shrunk"][ship].sum())
    mapped = cumulative_impact(x, se_p, shipped=ship, prior="map")

    nss_true = nss_adjusted_cumulative_impact(
        x,
        se_p,
        g,
        se_g,
        shipped=ship,
        rho=rho,
        prior_sd_primary=TAU,
        prior_sd_guard=TAU,
    )
    mom = estimate_guardrail_rho(x, se_p, g, se_g)
    # MoM can return τ̂=0 on a noisy draw; fall back to a tiny positive floor.
    sd_p = float(mom["tau_primary"]) if mom["tau_primary"] > 0 else TAU
    sd_g = float(mom["tau_guard"]) if mom["tau_guard"] > 0 else TAU
    nss_hat = nss_adjusted_cumulative_impact(
        x,
        se_p,
        g,
        se_g,
        shipped=ship,
        rho=float(mom["rho"]),
        prior_sd_primary=sd_p,
        prior_sd_guard=sd_g,
    )

    means = {
        "naive": float(naive_ship.mean()),
        "EB": float(eb["shrunk"][ship].mean()),
        "t-prior": float(t_row["shrunk"][ship].mean()),
        "MAP": float(mapped["shrunk"][ship].mean()),
        "NSS ρ": float(nss_true["shrunk"][ship].mean()),
        "NSS ρ̂": float(nss_hat["shrunk"][ship].mean()),
    }
    sums = {
        "naive": float(naive_ship.sum()),
        "EB": float(eb["cumulative"]),
        "t-prior": t_cum,
        "MAP": float(mapped["cumulative"]),
        "NSS ρ": float(nss_true["cumulative"]),
        "NSS ρ̂": float(nss_hat["cumulative"]),
    }
    # per-experiment MAE among shipped (row-level)
    mae_rows = {
        "naive": float(np.mean(np.abs(naive_ship - true))),
        "EB": float(np.mean(np.abs(eb["shrunk"][ship] - true))),
        "t-prior": float(np.mean(np.abs(t_row["shrunk"][ship] - true))),
        "MAP": float(np.mean(np.abs(mapped["shrunk"][ship] - true))),
        "NSS ρ": float(np.mean(np.abs(nss_true["shrunk"][ship] - true))),
        "NSS ρ̂": float(np.mean(np.abs(nss_hat["shrunk"][ship] - true))),
    }
    return {
        "n_ship": int(ship.sum()),
        "true_mean": true_mean,
        "true_sum": true_sum,
        "means": means,
        "sums": sums,
        "mae_rows": mae_rows,
        "rho_hat": float(mom["rho"]),
    }


def run_scenario(rho: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(N_SIM):
        out = one_portfolio(rng, rho)
        if out is not None:
            rows.append(out)
    if not rows:
        raise RuntimeError(f"no usable portfolios for rho={rho}")

    true_mean = np.array([r["true_mean"] for r in rows])
    true_sum = np.array([r["true_sum"] for r in rows])
    n_ship = np.array([r["n_ship"] for r in rows], dtype=float)

    summary = {
        "rho": rho,
        "n_ok": len(rows),
        "mean_n_ship": float(n_ship.mean()),
        "mean_rho_hat": float(np.mean([r["rho_hat"] for r in rows])),
    }
    for m in METHODS:
        mean_hat = np.array([r["means"][m] for r in rows])
        sum_hat = np.array([r["sums"][m] for r in rows])
        mae_row = np.array([r["mae_rows"][m] for r in rows])
        summary[f"avg_mae_row_{m}"] = float(mae_row.mean())
        summary[f"avg_abs_err_mean_{m}"] = float(np.mean(np.abs(mean_hat - true_mean)))
        summary[f"avg_bias_mean_{m}"] = float(np.mean(mean_hat - true_mean))
        summary[f"cum_mae_{m}"] = float(np.mean(np.abs(sum_hat - true_sum)))
        summary[f"cum_bias_{m}"] = float(np.mean(sum_hat - true_sum))
    return summary


def best_method(summary: dict, metric_prefix: str) -> str:
    scores = {m: summary[f"{metric_prefix}_{m}"] for m in METHODS}
    return min(scores, key=scores.get)


# %%
print("SCALED ship rule: primary one-sided win (α=0.05) AND guardrail >= 0")
print(f"DGP: n_exp={N_EXP}, n_sim={N_SIM}, τ={TAU}, SE_p={SE_P}, SE_g={SE_G}")
print(f"Methods: {', '.join(METHODS)}\n")

summaries = []
for i, (label, rho) in enumerate(SCENARIOS.items()):
    s = run_scenario(rho, seed=100 + i)
    s["scenario"] = label
    summaries.append(s)

# %% [markdown]
# ## 1. Average estimate among shipped
#
# Row MAE = mean |θ̂ − Δ| over shipped experiments (then averaged over sims).
# Abs err of mean = |mean(θ̂_ship) − mean(Δ_ship)| — program-average accuracy.

# %%
print("=" * 88)
print("1) AVERAGE among shipped  (lower better)")
print("=" * 88)
hdr = f"{'scenario':<12} {'n̄ ship':>7} {'ρ̂':>6}"
for m in METHODS:
    hdr += f" {m:>8}"
print(hdr + "  | best")
print("-" * 88)
for s in summaries:
    line = f"{s['scenario']:<12} {s['mean_n_ship']:7.1f} {s['mean_rho_hat']:6.2f}"
    for m in METHODS:
        line += f" {s[f'avg_mae_row_{m}']:8.4f}"
    winner = best_method(s, "avg_mae_row")
    print(line + f"  | {winner}")
print("\n  cells = mean row-wise MAE among shipped\n")

print("  Abs error of the *mean* shipped lift:")
hdr = f"{'scenario':<12}"
for m in METHODS:
    hdr += f" {m:>8}"
print(hdr + "  | best")
print("-" * 88)
for s in summaries:
    line = f"{s['scenario']:<12}"
    for m in METHODS:
        line += f" {s[f'avg_abs_err_mean_{m}']:8.4f}"
    print(line + f"  | {best_method(s, 'avg_abs_err_mean')}")

# %% [markdown]
# ## 2. Cumulative impact (sum of shipped primary lifts)

# %%
print("\n" + "=" * 88)
print("2) CUMULATIVE impact  (MAE of sum; lower better)")
print("=" * 88)
hdr = f"{'scenario':<12}"
for m in METHODS:
    hdr += f" {m:>8}"
print(hdr + "  | best")
print("-" * 88)
for s in summaries:
    line = f"{s['scenario']:<12}"
    for m in METHODS:
        line += f" {s[f'cum_mae_{m}']:8.4f}"
    print(line + f"  | {best_method(s, 'cum_mae')}")

print("\n  Bias of cumulative (mean est − true; closer to 0 better):")
hdr = f"{'scenario':<12}"
for m in METHODS:
    hdr += f" {m:>8}"
print(hdr)
print("-" * 88)
for s in summaries:
    line = f"{s['scenario']:<12}"
    for m in METHODS:
        line += f" {s[f'cum_bias_{m}']:+8.4f}"
    print(line)

# %% [markdown]
# ## 3. Takeaway

# %%
print("\n" + "=" * 88)
print("3) TAKEAWAY for this ship rule")
print("=" * 88)
print(
    """
  • Naive always loses (winner's curse on significant + guardrail-selected ships).
  • At ρ=0, EB ≡ NSS (no guardrail information); prefer simple EB / MAP.
  • With a precise guardrail (SE_g ≪ SE_p), NSS with true ρ wins on average as
    soon as |ρ| is non-trivial — including mild ±0.3 in this DGP — for both
    row MAE and cumulative MAE; bias of the sum is near zero.
  • NSS with estimated ρ̂ helps vs naive but often trails oracle NSS / EB when
    MoM τ̂ or ρ̂ is noisy — prefer a stable archive estimate of (ρ, τ).
  • EB / MAP / t-prior are similar; they remove most WC. Use them when you do
    not trust ρ or lack paired guardrail lifts.
  • Always pass the real ship mask: primary significant AND guardrail >= 0.
"""
)

# Optional: export a tidy frame for notebooks
df = pd.DataFrame(summaries)
# df.to_csv("scaled_ship_method_assessment.csv", index=False)

# %%
