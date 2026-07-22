# %% [markdown]
# # Guardrail correlation and prior definition
#
# When shipping depends on a primary metric *and* guardrails, two questions get
# mixed up:
#
# 1. **Selection**: which experiments enter the "scaled / cumulative impact" set?
# 2. **Inference**: given that set, how should we estimate \(\Delta\)?
#
# Important: these effects are **different sizes**. Winner's-curse inflation of
# naive \(X\) among shipped tests is large. The extra bias from *ignoring*
# guardrail correlation in the posterior is usually much smaller — so raw tables
# can look "not that different" even when the sign of the residual bias flips.
#
# This script uses the package APIs:
# `cumulative_impact`, `joint_metric_shrinkage`, `estimate_guardrail_rho`.

# %%
import numpy as np
import pandas as pd

from experiment_utils.shrinkage import (
    cumulative_impact,
    estimate_guardrail_rho,
    joint_metric_shrinkage,
)

N_EXPERIMENTS = 20_000
SE_PRIMARY = 0.025  # noisy primary → more room for G to inform Δ
SE_GUARD = 0.012  # relatively precise guardrail
PRIOR_SD = 0.015
SHIP_Z_PRIMARY = 1.96
GUARD_FLOOR = 0.0

# rho here is corr(true primary, true guardrail). Product metrics are often
# weakly aligned (retention/engagement) or mildly opposed (growth vs quality).
# |rho|=0.8 is rare; keep a mild and a strong-but-plausible case.
SCENARIOS = {
    "rho=0 (independent)": 0.0,
    "rho=+0.3 (mild align)": 0.3,
    "rho=+0.5 (strong align)": 0.5,
    "rho=-0.3 (mild tradeoff)": -0.3,
}


def simulate_portfolio(rho, seed):
    local = np.random.default_rng(seed)
    delta = PRIOR_SD * local.standard_normal(N_EXPERIMENTS)
    gamma = rho * delta + PRIOR_SD * np.sqrt(1.0 - rho**2) * local.standard_normal(N_EXPERIMENTS)
    x = delta + local.normal(0.0, SE_PRIMARY, N_EXPERIMENTS)
    g = gamma + local.normal(0.0, SE_GUARD, N_EXPERIMENTS)
    primary_win = x > SHIP_Z_PRIMARY * SE_PRIMARY
    ship = primary_win & (g >= GUARD_FLOOR)
    return pd.DataFrame(
        {
            "delta": delta,
            "gamma": gamma,
            "x": x,
            "g": g,
            "primary_win": primary_win,
            "ship": ship,
        }
    )


def summarize(rho, label, seed):
    df = simulate_portfolio(rho, seed)
    shipped = df.loc[df["ship"]]
    if shipped.empty:
        raise RuntimeError(f"No shipped experiments for {label}")

    x_all = df["x"].to_numpy()
    g_all = df["g"].to_numpy()
    se_p = np.full(N_EXPERIMENTS, SE_PRIMARY)
    se_g = np.full(N_EXPERIMENTS, SE_GUARD)
    ship = df["ship"].to_numpy()

    true = shipped["delta"].to_numpy()
    true_mean = float(true.mean())
    true_sum = float(true.sum())

    # Primary-only: shrink all, sum shipped (known archive prior)
    primary_cum = cumulative_impact(
        x_all,
        se_p,
        shipped=ship,
        tau2=PRIOR_SD**2,
        prior_mean=0.0,
        aggregation="sum",
    )
    primary_only = primary_cum["shrunk"][ship]

    # Joint with *true* rho (oracle correlation)
    joint_true = joint_metric_shrinkage(
        x_all,
        se_p,
        g_all,
        se_g,
        rho=rho,
        prior_sd_primary=PRIOR_SD,
        prior_sd_guard=PRIOR_SD,
    )
    joint_oracle = joint_true["primary_shrunk"][ship]

    # Joint with *estimated* rho from the full archive (realistic)
    rho_hat = estimate_guardrail_rho(x_all, se_p, g_all, se_g)
    joint_est = joint_metric_shrinkage(
        x_all,
        se_p,
        g_all,
        se_g,
        rho=rho_hat["rho"],
        prior_sd_primary=rho_hat["tau_primary"],
        prior_sd_guard=rho_hat["tau_guard"],
    )
    joint_estimated = joint_est["primary_shrunk"][ship]

    naive = shipped["x"].to_numpy()
    return {
        "scenario": label,
        "rho": rho,
        "rho_hat": rho_hat["rho"],
        "primary_win_rate": float(df["primary_win"].mean()),
        "ship_rate": float(df["ship"].mean()),
        "n_shipped": int(len(shipped)),
        "true_mean_shipped": true_mean,
        "pct_err_naive": 100.0 * (naive.mean() / true_mean - 1.0),
        "pct_err_primary_only": 100.0 * (primary_only.mean() / true_mean - 1.0),
        "pct_err_joint": 100.0 * (joint_oracle.mean() / true_mean - 1.0),
        "pct_err_joint_est": 100.0 * (joint_estimated.mean() / true_mean - 1.0),
        "pct_gap_primary_vs_joint": 100.0 * (primary_only.mean() - joint_oracle.mean()) / true_mean,
        "cum_true": true_sum,
        "cum_naive": float(naive.sum()),
        "cum_primary_only": float(primary_cum["cumulative"]),
        "cum_joint": float(joint_oracle.sum()),
        "cum_joint_est": float(joint_estimated.sum()),
    }


rows = [summarize(rho, label, seed=100 + i) for i, (label, rho) in enumerate(SCENARIOS.items())]
out = pd.DataFrame(rows)


def pct(x):
    return f"{x:+.0f}%"


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. Estimation error among shipped tests
#
# Percent error vs true mean primary effect of shipped experiments.
# Focus on the gap column: does ignoring the guardrail change the answer?

# %%
section("1) Estimation error among SHIPPED tests  (% of true mean)")
print(f"  {'scenario':<26} {'naive X':>10} {'primary-only':>14} {'joint ρ':>10} {'joint ρ̂':>10} {'gap (P-J)':>12}")
print(f"  {'-' * 26} {'-' * 10} {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 12}")
for r in out.itertuples(index=False):
    print(
        f"  {r.scenario:<26} {pct(r.pct_err_naive):>10} "
        f"{pct(r.pct_err_primary_only):>14} {pct(r.pct_err_joint):>10} "
        f"{pct(r.pct_err_joint_est):>10} {pct(r.pct_gap_primary_vs_joint):>12}"
    )
print(
    "\n  naive X       = raw observed primary (winner's curse: always huge)\n"
    "  primary-only  = cumulative_impact / EB on primary only\n"
    "  joint ρ       = joint_metric_shrinkage with true rho\n"
    "  joint ρ̂       = joint_metric_shrinkage with estimate_guardrail_rho\n"
    "  gap (P-J)     = primary-only minus joint(true ρ)\n"
    "                  ~0 when rho=0; grows with |rho|; sign flips on tradeoffs"
)

# %% [markdown]
# ## 2. Selection: guardrails change who ships
#
# Even when inference does not need \(G\), cumulative impact does — because
# guardrails change the shipped set.

# %%
section("2) Selection: who ships?  (guardrails change the set)")
print(f"  {'scenario':<26} {'primary win':>12} {'shipped':>10} {'blocked':>10} {'true cum Δ':>12}")
print(f"  {'-' * 26} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 12}")
for r in out.itertuples(index=False):
    blocked = 1.0 - (r.ship_rate / r.primary_win_rate)
    print(f"  {r.scenario:<26} {r.primary_win_rate:>11.1%} {r.ship_rate:>9.1%} {blocked:>9.0%} {r.cum_true:>12.2f}")
print(
    "\n  blocked = share of primary 'wins' stopped by G >= 0\n"
    "  true cum Δ rises when metrics align (more ships) and falls under tradeoffs"
)

# %% [markdown]
# ## 3. Cumulative / scaled impact
#
# Same estimators on the portfolio sum (`cumulative_impact` for primary-only).

# %%
section("3) Cumulative impact  (sum of primary effects among shipped)")
print(
    f"  {'scenario':<26} {'true sum':>10} {'err naive':>12} {'err prim':>10} {'err joint':>10} {'err ρ̂':>8} {'gap':>8}"
)
print(f"  {'-' * 26} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8}")
for r in out.itertuples(index=False):
    err_n = 100.0 * (r.cum_naive / r.cum_true - 1.0)
    err_p = 100.0 * (r.cum_primary_only / r.cum_true - 1.0)
    err_j = 100.0 * (r.cum_joint / r.cum_true - 1.0)
    err_e = 100.0 * (r.cum_joint_est / r.cum_true - 1.0)
    gap = 100.0 * ((r.cum_primary_only - r.cum_joint) / r.cum_true)
    print(
        f"  {r.scenario:<26} {r.cum_true:>10.2f} {pct(err_n):>12} "
        f"{pct(err_p):>10} {pct(err_j):>10} {pct(err_e):>8} {pct(gap):>8}"
    )

# %% [markdown]
# ## 4. How well is ρ recovered?

# %%
section("4) estimate_guardrail_rho  (archive MoM)")
print(f"  {'scenario':<26} {'true ρ':>10} {'ρ̂':>10} {'error':>10}")
print(f"  {'-' * 26} {'-' * 10} {'-' * 10} {'-' * 10}")
for r in out.itertuples(index=False):
    print(f"  {r.scenario:<26} {r.rho:>+10.2f} {r.rho_hat:>+10.2f} {r.rho_hat - r.rho:>+10.2f}")

# %% [markdown]
# ## Takeaway
#
# 1. Winner's curse dominates (naive column).
# 2. At product-like \(|\rho|\), joint vs primary-only is a smaller correction —
#    check the gap column, not absolute means next to naive.
# 3. Always put guardrails in the *shipped set*; put them in the posterior for
#    \(\Delta\) only when correlation is non-trivial (gap not near 0).
# 4. `estimate_guardrail_rho` is good enough here that joint-with-\(\hat\rho\)
#    tracks joint-with-true-\(\rho\).

# %%
section("Takeaway")
print(
    "  • Always define the scaled set with the real ship rule (incl. guardrails).\n"
    "  • primary-only: cumulative_impact(..., tau2=...) fixes most WC when ρ≈0.\n"
    "  • joint: joint_metric_shrinkage when |gap| is material (esp. tradeoffs).\n"
    "  • estimate_guardrail_rho → pass rho= into joint_metric_shrinkage.\n"
)

# %%
