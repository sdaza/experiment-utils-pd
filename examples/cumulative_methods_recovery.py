# %% [markdown]
# # Monte Carlo recovery checks for cumulative-impact methods
#
# Seeded simulations under known DGPs. Each claim prints **PASS** / **FAIL**.
# These mirror the CI tests in `tests/test_cumulative_methods_recovery.py`.
#
# Claims:
# 1. `cumulative_impact` (known prior) debiases the sum of shipped lifts
# 2. Plug-in sum CI covers near nominally under a fixed prior
# 3. `prior="map"` cumulative recovers similarly to known-τ EB
# 4. `process_level_total_effect` (Airbnb \(\hat T_A\)) beats naive \(S_A\)
# 5. `estimate_guardrail_rho` recovers true ρ
# 6. `joint_metric_shrinkage` (true / estimated ρ) beats primary-only MSE
# 7. One-sided `winners_curse_estimate` is median-unbiased among winners
#
# Run:
#
#     uv run python examples/cumulative_methods_recovery.py

# %%
import numpy as np
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    empirical_bayes_shrinkage,
    estimate_guardrail_rho,
    fit_normal_prior_map,
    joint_metric_shrinkage,
    process_level_total_effect,
    winners_curse_estimate,
)

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def record(claim: str, detail: str, ok: bool) -> None:
    status = PASS if ok else FAIL
    results.append((status, claim, detail))
    print(f"  [{status}] {claim}")
    print(f"         {detail}")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# %% [markdown]
# ## 1–2. Kessler cumulative impact: bias + CI coverage

# %%
section("1) cumulative_impact — naive bias vs EB recovery")
rng = np.random.default_rng(42)
n_exp, n_sim = 80, 300
tau, se = 0.02, 0.025
z = norm.ppf(0.975)
true_sums, naive_sums, eb_sums, map_sums = [], [], [], []
for _ in range(n_sim):
    theta = rng.normal(0.0, tau, n_exp)
    y = theta + rng.normal(0.0, se, n_exp)
    ship = y > z * se
    if ship.sum() < 1:
        continue
    true_sums.append(float(theta[ship].sum()))
    naive_sums.append(float(y[ship].sum()))
    eb_sums.append(cumulative_impact(y, np.full(n_exp, se), shipped=ship, tau2=tau**2)["cumulative"])
    map_sums.append(cumulative_impact(y, np.full(n_exp, se), shipped=ship, prior="map")["cumulative"])

true_sums = np.asarray(true_sums)
naive_bias = float(np.mean(np.asarray(naive_sums) - true_sums))
eb_bias = float(np.mean(np.asarray(eb_sums) - true_sums))
map_bias = float(np.mean(np.asarray(map_sums) - true_sums))
scale = float(np.mean(np.abs(true_sums))) + 1e-12

record(
    "naive sum of winners is biased high",
    f"mean(naive−true)={naive_bias:+.4f}  (> 5% of mean|true|={0.05 * scale:.4f})",
    naive_bias > 0.05 * scale,
)
record(
    "EB cumulative_impact (known τ²) recovers",
    f"mean(eb−true)={eb_bias:+.4f}  (target |bias| < {max(0.03 * scale, 0.02):.4f})",
    abs(eb_bias) < max(0.03 * scale, 0.02),
)
record(
    "MAP prior cumulative recovers vs naive",
    f"mean(map−true)={map_bias:+.4f}; |map| < |naive| and |map| < 0.08",
    abs(map_bias) < abs(naive_bias) and abs(map_bias) < 0.08,
)

section("2) cumulative_impact — 95% CI coverage (fixed prior)")
rng = np.random.default_rng(99)
n_exp, n_sim = 60, 400
covered = n_used = 0
for _ in range(n_sim):
    theta = rng.normal(0.0, tau, n_exp)
    y = theta + rng.normal(0.0, se, n_exp)
    ship = y > z * se
    if ship.sum() < 2:
        continue
    out = cumulative_impact(y, np.full(n_exp, se), shipped=ship, tau2=tau**2, ci=0.95)
    n_used += 1
    covered += int(out["ci_lower"] <= float(theta[ship].sum()) <= out["ci_upper"])
rate = covered / n_used
record(
    "plug-in sum CI covers ~95% under known prior",
    f"coverage={rate:.3f} on {n_used} reps (accept 0.90–0.99)",
    0.90 <= rate <= 0.99,
)

# %% [markdown]
# ## 3. Airbnb process-level \(\hat T_A\)

# %%
section("3) process_level_total_effect (Lee & Shen / Airbnb)")
rng = np.random.default_rng(21)
n_exp, n_sim = 30, 400
true_a, naive_a, ta, cond = [], [], [], []
for _ in range(n_sim):
    a = rng.normal(0.2, 0.5, n_exp)
    s = np.sqrt(rng.choice([0.3, 0.5, 0.8], size=n_exp))
    x = rng.normal(a, s)
    sel = x > norm.ppf(0.95) * s
    if sel.sum() < 1:
        continue
    true_a.append(float(a[sel].sum()))
    out = process_level_total_effect(x, s, alpha=0.05, alternative="greater")
    naive_a.append(out["naive_total"])
    ta.append(out["total"])
    cond.append(out["conditional_total"])

true_a = np.asarray(true_a)
naive_bias_a = float(np.mean(np.asarray(naive_a) - true_a))
ta_bias = float(np.mean(np.asarray(ta) - true_a))
cond_bias = float(np.mean(np.asarray(cond) - true_a))
record(
    "S_A biased high",
    f"mean(S_A−T_A)={naive_bias_a:+.3f}",
    naive_bias_a > 0.15,
)
record(
    "T̂_A less biased than S_A and near 0",
    f"mean(T̂_A−T_A)={ta_bias:+.3f}; |T̂_A| < |S_A| and |T̂_A| < 0.35",
    abs(ta_bias) < abs(naive_bias_a) and abs(ta_bias) < 0.35,
)
record(
    "conditional total still overstates vs T̂_A",
    f"mean(T_cond−T_A)={cond_bias:+.3f} > mean(T̂_A−T_A)={ta_bias:+.3f}",
    cond_bias > ta_bias,
)

# %% [markdown]
# ## 4–5. Guardrail ρ and joint shrinkage

# %%
section("4) estimate_guardrail_rho")
rng = np.random.default_rng(5)
n = 2500
tau_g = 0.02
se_p, se_g = 0.015, 0.012
rho_ok = True
details = []
for rho_true in (-0.5, 0.0, 0.5):
    delta = tau_g * rng.standard_normal(n)
    gamma = rho_true * delta + tau_g * np.sqrt(1 - rho_true**2) * rng.standard_normal(n)
    x = delta + rng.normal(0, se_p, n)
    g = gamma + rng.normal(0, se_g, n)
    hat = estimate_guardrail_rho(x, np.full(n, se_p), g, np.full(n, se_g))["rho"]
    err = abs(hat - rho_true)
    details.append(f"ρ={rho_true:+.1f} → ρ̂={hat:+.3f} (|err|={err:.3f})")
    rho_ok = rho_ok and err < 0.12
record("MoM ρ recovers within 0.12", "; ".join(details), rho_ok)

section("5) joint_metric_shrinkage vs primary-only")
rng = np.random.default_rng(11)
n = 4000
tau_j, rho = 0.015, 0.75
se_p, se_g = 0.03, 0.01
delta = tau_j * rng.standard_normal(n)
gamma = rho * delta + tau_j * np.sqrt(1 - rho**2) * rng.standard_normal(n)
x = delta + rng.normal(0, se_p, n)
g = gamma + rng.normal(0, se_g, n)
uni = empirical_bayes_shrinkage(x, np.full(n, se_p), tau2=tau_j**2)
joint = joint_metric_shrinkage(x, np.full(n, se_p), g, np.full(n, se_g), rho=rho, prior_sd_primary=tau_j)
mse_u = float(np.mean((uni["shrunk"] - delta) ** 2))
mse_j = float(np.mean((joint["primary_shrunk"] - delta) ** 2))
record(
    "joint (true ρ) lowers primary MSE vs EB",
    f"MSE_uni={mse_u:.6f}, MSE_joint={mse_j:.6f} (joint < 0.85×uni)",
    mse_j < 0.85 * mse_u,
)

est = estimate_guardrail_rho(x, np.full(n, se_p), g, np.full(n, se_g))
joint_hat = joint_metric_shrinkage(
    x,
    np.full(n, se_p),
    g,
    np.full(n, se_g),
    rho=est["rho"],
    prior_sd_primary=est["tau_primary"],
    prior_sd_guard=est["tau_guard"],
)
mse_h = float(np.mean((joint_hat["primary_shrunk"] - delta) ** 2))
record(
    "joint (estimated ρ) still beats primary-only",
    f"MSE_uni={mse_u:.6f}, MSE_jointρ̂={mse_h:.6f}; ρ̂={est['rho']:.3f}",
    mse_h < mse_u,
)

# %% [markdown]
# ## 6. One-sided winner's-curse MUE

# %%
section("6) winners_curse_estimate(alternative='greater') median-unbiased")
rng = np.random.default_rng(7)
true, se_w, alpha = 0.50, 0.25, 0.05
z1 = norm.ppf(1.0 - alpha)
y = rng.normal(true, se_w, 40_000)
winners = y[y >= z1 * se_w]
corrected = np.array(
    [winners_curse_estimate(float(w), se_w, alpha=alpha, alternative="greater")["corrected"] for w in winners[::8]]
)
med = float(np.median(corrected))
raw_mean = float(np.mean(winners))
record(
    "median of one-sided corrected ≈ true",
    f"median(corrected)={med:.3f}, true={true:.3f}, mean(raw winners)={raw_mean:.3f}",
    abs(med - true) < 0.08 and abs(raw_mean - true) > 0.10,
)

# %% [markdown]
# ## 7. MAP τ recovery

# %%
section("7) fit_normal_prior_map recovers τ, μ")
rng = np.random.default_rng(8)
n = 400
tau_m, mu_m, se_m = 0.025, 0.005, 0.02
theta = rng.normal(mu_m, tau_m, n)
y = theta + rng.normal(0, se_m, n)
fit = fit_normal_prior_map(y, np.full(n, se_m))
record(
    "Half-Cauchy MAP recovers τ and μ",
    f"τ̂={np.sqrt(fit['tau2']):.4f} (true {tau_m}), μ̂={fit['prior_mean']:.4f} (true {mu_m})",
    abs(np.sqrt(fit["tau2"]) - tau_m) < 0.012 and abs(fit["prior_mean"] - mu_m) < 0.01,
)

# %%
section("Summary")
n_pass = sum(1 for s, _, _ in results if s == PASS)
n_fail = sum(1 for s, _, _ in results if s == FAIL)
print(f"  {n_pass} PASS, {n_fail} FAIL  (of {len(results)} claims)\n")
if n_fail:
    print("  Failed claims:")
    for status, claim, detail in results:
        if status == FAIL:
            print(f"    - {claim}: {detail}")
    raise SystemExit(1)
print("  All recovery claims passed.")
