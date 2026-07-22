# %% [markdown]
# # Shrinkage methods — short simulation tour
#
# Six small, seeded simulations that show what each tool in
# `experiment_utils.shrinkage` does. Truth is known in every DGP so you can
# compare naive vs corrected.
#
# Run:
#
#     uv run python examples/shrinkage_methods_tour.py

# %%
import numpy as np
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    empirical_bayes_shrinkage,
    estimate_guardrail_rho,
    fit_t_prior,
    joint_metric_shrinkage,
    nss_adjusted_cumulative_impact,
    process_level_total_effect,
    t_prior_shrinkage,
    winners_curse_estimate,
)

rng = np.random.default_rng(42)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1. Winner's curse on a single significant result
#
# Underpowered true effect; keep only two-sided winners. Observed mean of
# winners is inflated; `winners_curse_estimate` pulls each one back.

# %%
section("1) winners_curse_estimate — one significant lift")

true_beta, se, alpha = 0.40, 0.25, 0.05
z = norm.ppf(1 - alpha / 2)
obs = rng.normal(true_beta, se, 20_000)
winners = obs[np.abs(obs) >= z * se]
example = float(winners[0])
wc = winners_curse_estimate(example, se, alpha=alpha)

print(f"  true effect                         {true_beta:6.3f}")
print(f"  mean of all draws                   {obs.mean():6.3f}  (≈ unbiased)")
print(f"  mean of significant winners         {winners.mean():6.3f}  (inflated)")
print(f"  one winner observed                 {example:6.3f}")
print(f"  corrected (median-unbiased)         {wc['corrected']:6.3f}")
print(f"  shrinkage ratio corrected/observed  {wc['shrinkage']:6.2f}")

# %% [markdown]
# ## 2. Portfolio EB shrinkage (normal prior)
#
# Many true effects ~ N(0, τ²). Big noisy winners shrink most.

# %%
section("2) empirical_bayes_shrinkage — portfolio of effects")

n, tau, se_p = 200, 0.02, 0.03
delta = rng.normal(0.0, tau, n)
y = delta + rng.normal(0.0, se_p, n)
eb = empirical_bayes_shrinkage(y, np.full(n, se_p))

mae_raw = float(np.mean(np.abs(y - delta)))
mae_eb = float(np.mean(np.abs(eb["shrunk"] - delta)))
top = int(np.argmax(np.abs(y)))

print(f"  true τ                              {tau:6.4f}")
print(f"  learned τ̂ (sqrt tau2)               {np.sqrt(eb['tau2']):6.4f}")
print(f"  MAE raw vs truth                    {mae_raw:6.4f}")
print(f"  MAE EB vs truth                     {mae_eb:6.4f}")
print(f"  largest |observed|                  {y[top]:+.4f} → shrunk {eb['shrunk'][top]:+.4f}")

# %% [markdown]
# ## 3. Student-t prior (fatter tails than normal)
#
# Same archive; t prior shrinks extreme winners less than a normal with the
# same scale.

# %%
section("3) fit_t_prior + t_prior_shrinkage")

t_fit = fit_t_prior(y, np.full(n, se_p), df=4.0)
t_shr = t_prior_shrinkage(y, np.full(n, se_p), scale=t_fit["scale"], df=t_fit["df"])
norm_at_extreme = empirical_bayes_shrinkage(y, np.full(n, se_p), tau2=t_fit["tau2"])

print(f"  fitted scale / df                   {t_fit['scale']:.4f} / {t_fit['df']:.0f}")
print(f"  extreme obs                         {y[top]:+.4f}")
print(f"  normal EB (matched τ²)              {norm_at_extreme['shrunk'][top]:+.4f}")
print(f"  Student-t posterior mean            {t_shr['shrunk'][top]:+.4f}  (less shrink)")

# %% [markdown]
# ## 4. Cumulative impact of shipped experiments
#
# Sum of raw winners overstates program lift. Shrink-then-sum recovers.

# %%
section("4) cumulative_impact — shrink then sum shipped")

n4, tau4, se4 = 400, 0.015, 0.025
z2 = norm.ppf(0.975)
delta4 = tau4 * rng.standard_normal(n4)
x4 = delta4 + rng.normal(0.0, se4, n4)
ship = x4 > z2 * se4
true_cum = float(delta4[ship].sum())
naive = float(x4[ship].sum())
eb_cum = cumulative_impact(x4, np.full(n4, se4), shipped=ship, tau2=tau4**2)

print(f"  shipped / N                         {ship.sum()} / {n4}")
print(f"  true cumulative Δ (shipped)         {true_cum:+.4f}")
print(f"  naive sum of X                      {naive:+.4f}  (err {100 * (naive / true_cum - 1):+.0f}%)")
print(
    f"  EB cumulative                       {eb_cum['cumulative']:+.4f}  "
    f"(err {100 * (eb_cum['cumulative'] / true_cum - 1):+.0f}%)"
)
print(f"  95% CI                              [{eb_cum['ci_lower']:+.4f}, {eb_cum['ci_upper']:+.4f}]")

# %% [markdown]
# ## 5. Airbnb process-level total (Lee & Shen \(\hat T_A\))
#
# Debias the **sum of one-sided winners** without an archive prior.

# %%
section("5) process_level_total_effect — Airbnb T̂_A")

n5, true5, se5 = 300, 0.01, 0.012
x5 = rng.normal(true5, se5, n5)
s5 = np.full(n5, se5)
win = x5 > norm.ppf(0.95) * se5
naive_ta = float(x5[win].sum())
airbnb = process_level_total_effect(x5, s5, alpha=0.05, alternative="greater")
true_ta = float(true5 * win.sum())  # E[sum of winners] ≈ β * n_wins under fixed β

print(f"  one-sided winners                   {win.sum()}")
print(f"  naive S_A                           {naive_ta:+.4f}")
print(f"  Airbnb T̂_A                          {airbnb['total']:+.4f}")
print(f"  (reference) β × n_wins              {true_ta:+.4f}")

# %% [markdown]
# ## 6. Joint primary|guardrail + NSS cumulative
#
# When primary and guardrail true effects are correlated, joint shrinkage
# improves the primary posterior; `nss_adjusted_cumulative_impact` aggregates it.
#
# A **single** portfolio with ~10 shipped can easily have NSS slightly worse than
# primary-only (sampling noise). The gain shows up in **average** absolute error
# when ρ is high and the guardrail is informative.

# %%
section("6) joint_metric_shrinkage + nss_adjusted_cumulative_impact")

# --- one draw (can go either way) ---
n6, tau6, se_p6, se_g6, rho = 350, 0.018, 0.022, 0.012, 0.40
delta6 = rng.normal(0.0, tau6, n6)
gamma6 = rho * delta6 + tau6 * np.sqrt(1 - rho**2) * rng.standard_normal(n6)
x6 = delta6 + rng.normal(0.0, se_p6, n6)
g6 = gamma6 + rng.normal(0.0, se_g6, n6)
ship6 = (x6 > norm.ppf(0.975) * se_p6) & (g6 >= 0.0)

rho_hat = estimate_guardrail_rho(x6, np.full(n6, se_p6), g6, np.full(n6, se_g6))
joint = joint_metric_shrinkage(
    x6,
    np.full(n6, se_p6),
    g6,
    np.full(n6, se_g6),
    rho=rho,
    prior_sd_primary=tau6,
    prior_sd_guard=tau6,
)
uni = empirical_bayes_shrinkage(x6, np.full(n6, se_p6), tau2=tau6**2)
mse_uni = float(np.mean((uni["shrunk"] - delta6) ** 2))
mse_j = float(np.mean((np.asarray(joint["primary_shrunk"]) - delta6) ** 2))

true_c6 = float(delta6[ship6].sum())
prim_c = cumulative_impact(x6, np.full(n6, se_p6), shipped=ship6, tau2=tau6**2)
nss = nss_adjusted_cumulative_impact(
    x6,
    np.full(n6, se_p6),
    g6,
    np.full(n6, se_g6),
    shipped=ship6,
    rho=rho,
    prior_sd_primary=tau6,
    prior_sd_guard=tau6,
)
err_prim = abs(prim_c["cumulative"] - true_c6)
err_nss = abs(nss["cumulative"] - true_c6)

print("  one portfolio (ρ=0.40; ~10 ships — noisy):")
print(f"    true ρ / ρ̂                        {rho:.2f} / {rho_hat['rho']:.2f}")
print(f"    MSE primary-only / joint          {mse_uni:.6f} / {mse_j:.6f}")
print(f"    shipped                           {int(ship6.sum())}")
print(f"    true cumulative                   {true_c6:+.4f}")
print(f"    primary-only / |err|              {prim_c['cumulative']:+.4f}  ({err_prim:.4f})")
print(f"    NSS-adjusted / |err|              {nss['cumulative']:+.4f}  ({err_nss:.4f})")
if err_nss > err_prim:
    print("    → NSS worse on this draw (expected sometimes at moderate ρ)")

# --- short MC: high ρ + precise guardrail (where NSS should win on average) ---
n_sim, n_exp = 200, 120
tau_mc, rho_mc, se_p_mc, se_g_mc = 0.015, 0.75, 0.025, 0.010
z2 = norm.ppf(0.975)
abs_naive, abs_prim, abs_nss = [], [], []
for _ in range(n_sim):
    d = tau_mc * rng.standard_normal(n_exp)
    gam = rho_mc * d + tau_mc * np.sqrt(1 - rho_mc**2) * rng.standard_normal(n_exp)
    xx = d + rng.normal(0.0, se_p_mc, n_exp)
    gg = gam + rng.normal(0.0, se_g_mc, n_exp)
    ship = (xx > z2 * se_p_mc) & (gg >= 0.0)
    if ship.sum() < 1:
        continue
    truth = float(d[ship].sum())
    abs_naive.append(abs(float(xx[ship].sum()) - truth))
    abs_prim.append(
        abs(cumulative_impact(xx, np.full(n_exp, se_p_mc), shipped=ship, tau2=tau_mc**2)["cumulative"] - truth)
    )
    abs_nss.append(
        abs(
            nss_adjusted_cumulative_impact(
                xx,
                np.full(n_exp, se_p_mc),
                gg,
                np.full(n_exp, se_g_mc),
                shipped=ship,
                rho=rho_mc,
                prior_sd_primary=tau_mc,
                prior_sd_guard=tau_mc,
            )["cumulative"]
            - truth
        )
    )

print(f"\n  mean |error| over {len(abs_nss)} sims (ρ={rho_mc:.2f}, precise guardrail):")
print(f"    naive sum of X                    {np.mean(abs_naive):.4f}")
print(f"    primary-only cumulative           {np.mean(abs_prim):.4f}")
print(f"    NSS-adjusted cumulative           {np.mean(abs_nss):.4f}")

# %%
