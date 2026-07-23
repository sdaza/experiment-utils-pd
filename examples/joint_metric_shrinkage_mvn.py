# %% [markdown]
# # Multi-guardrail MVN vs one guardrail
#
# Seeded Monte Carlo: does `joint_metric_shrinkage_mvn` (K companions) beat
# bivariate `joint_metric_shrinkage` / `nss_adjusted_cumulative_impact` with
# **just one** companion?
#
# **Magnitude only — not the scale rule.** `shipped` encodes a hard multi-guardrail
# gate (no companion significantly negative at one-sided p<0.1); see section 2.
#
# Run: `uv run python examples/joint_metric_shrinkage_mvn.py`

# %%
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    empirical_bayes_shrinkage,
    fit_t_prior,
    joint_metric_shrinkage,
    joint_metric_shrinkage_mvn,
    nss_adjusted_cumulative_impact,
    nss_adjusted_cumulative_impact_mvn,
)

rng = np.random.default_rng(2026)


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def sim_portfolio(n: int, tau: float, rhos: tuple[float, ...], se_p: float, se_g: float):
    """One-factor DGP: Corr(γ_j, γ_k) = ρ_j ρ_k (matches rho_guardrails='factor')."""
    k = len(rhos)
    delta = tau * rng.standard_normal(n)
    g_true = np.empty((n, k))
    for j, rho in enumerate(rhos):
        g_true[:, j] = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x = delta + rng.normal(0, se_p, n)
    g = g_true + rng.normal(0, se_g, size=(n, k))
    return delta, x, g


def pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# %% [markdown]
# ## 1. Per-experiment MSE: each single NSS vs all three at once

# %%
section("1) Primary MSE: one guardrail vs MVN K=3")

n, tau = 120, 0.015
rhos = (0.8, 0.7, 0.6)
names = ("NSS1 (ρ=0.8)", "NSS2 (ρ=0.7)", "NSS3 (ρ=0.6)")
se_p, se_g = 0.03, 0.01
n_sim = 300

mse_univ = 0.0
mse_one = np.zeros(len(rhos))
mse_mvn = 0.0

for _ in range(n_sim):
    delta, x, g = sim_portfolio(n, tau, rhos, se_p, se_g)
    se_pa = np.full(n, se_p)
    se_ga = np.full_like(g, se_g)

    univ = empirical_bayes_shrinkage(x, se_pa, prior_mean=0.0, tau2=tau**2)
    mse_univ += float(np.mean((univ["shrunk"] - delta) ** 2))

    for j, rho in enumerate(rhos):
        bi = joint_metric_shrinkage(
            x,
            se_pa,
            g[:, j],
            se_ga[:, j],
            rho=rho,
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        mse_one[j] += float(np.mean((bi["primary_shrunk"] - delta) ** 2))

    mvn = joint_metric_shrinkage_mvn(
        x,
        se_pa,
        g,
        se_ga,
        rho_primary=list(rhos),
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    mse_mvn += float(np.mean((mvn["primary_shrunk"] - delta) ** 2))

mse_univ /= n_sim
mse_one /= n_sim
mse_mvn /= n_sim
best_one = float(np.min(mse_one))
best_idx = int(np.argmin(mse_one))

print(f"  n_sim={n_sim}, n_exp={n}, se_p={se_p}, se_g={se_g}")
print(f"  primary-only EB          MSE = {mse_univ:.6f}")
for j, name in enumerate(names):
    print(f"  one guardrail {name:16s} MSE = {mse_one[j]:.6f}")
print(f"  MVN all K=3              MSE = {mse_mvn:.6f}")
print(f"  best single companion    = {names[best_idx]}")
print(f"  MVN vs best one          : {100 * (1 - mse_mvn / best_one):+.1f}% MSE [{pass_fail(mse_mvn < best_one)}]")
print(f"  MVN vs primary-only      : [{pass_fail(mse_mvn < mse_univ)}]")

# %% [markdown]
# ## 2. Cumulative under hard guardrail gate vs MVN K=5
#
# Same `shipped` for everyone: primary significant (two-sided 0.05) and no
# companion significantly negative at one-sided α=0.1
# (``g_k / SE_k >= -z_{0.90}`` for all k; up-metrics). MVN is not the scale rule.

# %%
section("2) Cumulative MAE: best one guardrail vs MVN K=5 (same shipped)")

rhos5 = (0.75, 0.70, 0.65, 0.60, 0.55)
names5 = [f"guard{j + 1} (ρ={r})" for j, r in enumerate(rhos5)]
z_primary = norm.ppf(0.975)  # primary win, two-sided 0.05
z_guard_neg = norm.ppf(0.90)  # one-sided α=0.1: significantly negative
n_exp, n_port = 100, 400

true_s: list[float] = []
naive_s: list[float] = []
univ_s: list[float] = []
one_s = [[] for _ in rhos5]  # type: list[list[float]]
mvn_s: list[float] = []

for _ in range(n_port):
    delta, x, g = sim_portfolio(n_exp, tau, rhos5, se_p, se_g)
    # Hard gate: block if any guardrail significantly negative @ one-sided 0.1
    guard_ok = np.all(g >= -z_guard_neg * se_g, axis=1)
    ship = (x > z_primary * se_p) & guard_ok
    if int(ship.sum()) < 2:
        continue
    true_s.append(float(delta[ship].sum()))
    naive_s.append(float(x[ship].sum()))
    univ_s.append(cumulative_impact(x, np.full(n_exp, se_p), shipped=ship, tau2=tau**2, prior_mean=0.0)["cumulative"])
    for j, rho in enumerate(rhos5):
        one_s[j].append(
            nss_adjusted_cumulative_impact(
                x,
                np.full(n_exp, se_p),
                g[:, j],
                np.full(n_exp, se_g),
                shipped=ship,
                rho=rho,
                prior_sd_primary=tau,
                prior_sd_guard=tau,
            )["cumulative"]
        )
    mvn_s.append(
        nss_adjusted_cumulative_impact_mvn(
            x,
            np.full(n_exp, se_p),
            g,
            np.full_like(g, se_g),
            shipped=ship,
            rho_primary=list(rhos5),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )["cumulative"]
    )

truth = np.asarray(true_s)
naive_a = np.asarray(naive_s)
univ_a = np.asarray(univ_s)
mvn_a = np.asarray(mvn_s)
one_mae = [float(np.mean(np.abs(np.asarray(one_s[j]) - truth))) for j in range(len(rhos5))]
best_one_j = int(np.argmin(one_mae))

naive_mae = float(np.mean(np.abs(naive_a - truth)))
univ_mae = float(np.mean(np.abs(univ_a - truth)))
mvn_mae = float(np.mean(np.abs(mvn_a - truth)))
naive_bias = float(np.mean(naive_a - truth))
univ_bias = float(np.mean(univ_a - truth))
mvn_bias = float(np.mean(mvn_a - truth))
best_one_mae = one_mae[best_one_j]
best_one_bias = float(np.mean(np.asarray(one_s[best_one_j]) - truth))

print(f"  n portfolios used = {len(true_s)} (of {n_port} draws)")
print(f"  {'method':28s}  {'MAE':>8s}  {'bias':>9s}")
print(f"  {'naive shipped sum':28s}  {naive_mae:8.4f}  {naive_bias:+9.4f}")
print(f"  {'primary-only EB':28s}  {univ_mae:8.4f}  {univ_bias:+9.4f}")
for j, name in enumerate(names5):
    bias_j = float(np.mean(np.asarray(one_s[j]) - truth))
    mark = " ← best one" if j == best_one_j else ""
    print(f"  {'one: ' + name:28s}  {one_mae[j]:8.4f}  {bias_j:+9.4f}{mark}")
print(f"  {'MVN all K=5':28s}  {mvn_mae:8.4f}  {mvn_bias:+9.4f}")
print()
print(f"  MVN MAE < best one-guardrail MAE? {mvn_mae:.4f} < {best_one_mae:.4f}  [{pass_fail(mvn_mae < best_one_mae)}]")
print(
    f"  MVN |bias| < naive |bias|?          "
    f"{abs(mvn_bias):.4f} < {abs(naive_bias):.4f}  [{pass_fail(abs(mvn_bias) < abs(naive_bias))}]"
)
print(
    "\n  Same shipped mask for all rows (primary sig + no guardrail significantly\n"
    "  negative @ one-sided 0.1). MVN only changes primary magnitude."
)

# %% [markdown]
# ## 3. prior="map" and Student-t scale plug-in (still normal MVN)
#
# Learn τ from the primary archive (Half-Cauchy MAP or t scale), then run the
# same multi-guardrail MVN. Not a multivariate-t — just easier prior plug-in.

# %%
section("3) MVN with prior='map' / t-scale vs known-τ and one guardrail")

n3, n_sim3 = 120, 250
rhos3 = (0.8, 0.7, 0.6)
mse_known = mse_map = mse_t = mse_one_best = mse_univ3 = 0.0
map_sds: list[float] = []
t_sds: list[float] = []

for _ in range(n_sim3):
    delta, x, g = sim_portfolio(n3, tau, rhos3, se_p, se_g)
    se_pa = np.full(n3, se_p)
    se_ga = np.full_like(g, se_g)

    univ = empirical_bayes_shrinkage(x, se_pa, prior_mean=0.0, tau2=tau**2)
    mse_univ3 += float(np.mean((univ["shrunk"] - delta) ** 2))

    one_mses = [
        float(
            np.mean(
                (
                    joint_metric_shrinkage(
                        x,
                        se_pa,
                        g[:, j],
                        se_ga[:, j],
                        rho=rhos3[j],
                        prior_sd_primary=tau,
                        prior_sd_guard=tau,
                    )["primary_shrunk"]
                    - delta
                )
                ** 2
            )
        )
        for j in range(len(rhos3))
    ]
    mse_one_best += min(one_mses)

    known = joint_metric_shrinkage_mvn(
        x,
        se_pa,
        g,
        se_ga,
        rho_primary=list(rhos3),
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    mse_known += float(np.mean((known["primary_shrunk"] - delta) ** 2))

    mvn_map = joint_metric_shrinkage_mvn(
        x,
        se_pa,
        g,
        se_ga,
        rho_primary=list(rhos3),
        prior="map",
    )
    mse_map += float(np.mean((mvn_map["primary_shrunk"] - delta) ** 2))
    map_sds.append(float(mvn_map["prior_sd_primary"]))

    t_fit = fit_t_prior(x, se_pa, df=4.0, prior_mean=0.0)
    mvn_t = joint_metric_shrinkage_mvn(
        x,
        se_pa,
        g,
        se_ga,
        rho_primary=list(rhos3),
        prior={"scale": t_fit["scale"], "df": t_fit["df"], "prior_mean": 0.0},
    )
    mse_t += float(np.mean((mvn_t["primary_shrunk"] - delta) ** 2))
    t_sds.append(float(mvn_t["prior_sd_primary"]))

mse_known /= n_sim3
mse_map /= n_sim3
mse_t /= n_sim3
mse_one_best /= n_sim3
mse_univ3 /= n_sim3

print(f"  true τ = {tau:.4f}")
print(f"  mean MAP prior_sd  = {np.mean(map_sds):.4f}")
print(f"  mean t-scale SD    = {np.mean(t_sds):.4f}  (plugged into normal MVN)")
print(f"  {'method':32s}  MSE")
print(f"  {'primary-only EB (known τ)':32s}  {mse_univ3:.6f}")
print(f"  {'best one guardrail (known τ)':32s}  {mse_one_best:.6f}")
print(f"  {'MVN K=3 known τ':32s}  {mse_known:.6f}")
print(f"  {'MVN K=3 prior=map':32s}  {mse_map:.6f}")
print(f"  {'MVN K=3 prior=t-scale':32s}  {mse_t:.6f}")
print()
print(f"  MAP MVN beats best one-guardrail?  [{pass_fail(mse_map < mse_one_best)}]")
print(f"  t-scale MVN beats best one-guardrail? [{pass_fail(mse_t < mse_one_best)}]")
print(f"  MAP MVN beats primary-only?        [{pass_fail(mse_map < mse_univ3)}]")
print(f"  t-scale MVN beats primary-only?    [{pass_fail(mse_t < mse_univ3)}]")
print(
    f"  MAP / known-τ MSE ratio = {mse_map / mse_known:.2f}x; "
    f"t / known-τ = {mse_t / mse_known:.2f}x "
    f"(oracle τ is best; learned priors a bit noisier)"
)
print(
    "\n  Note: joint remains normal–normal; MAP/t only set prior_sd_primary\n"
    "  (and mean for MAP). Not a multivariate-t posterior."
)
