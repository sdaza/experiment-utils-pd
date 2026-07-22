"""CI-enforced Monte Carlo recovery for cumulative-impact method suite.

Mirrors ``examples/cumulative_methods_recovery.py``. If a claim fails: diagnose;
do not loosen tolerances without understanding why.
"""

import numpy as np
from scipy.stats import norm

from experiment_utils import (
    cumulative_impact,
    estimate_guardrail_rho,
    fit_normal_prior_map,
    joint_metric_shrinkage,
    process_level_total_effect,
    winners_curse_estimate,
)
from experiment_utils.utils import empirical_bayes_shrinkage


def test_recovery_cumulative_eb_and_map():
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
    assert naive_bias > 0.05 * scale
    assert abs(eb_bias) < max(0.03 * scale, 0.02)
    assert abs(map_bias) < abs(naive_bias)
    assert abs(map_bias) < 0.08


def test_recovery_cumulative_ci_coverage():
    rng = np.random.default_rng(99)
    n_exp, n_sim = 60, 400
    tau, se = 0.02, 0.025
    z = norm.ppf(0.975)
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
    assert n_used > 200
    assert 0.90 <= covered / n_used <= 0.99


def test_recovery_process_level_airbnb():
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
    naive_bias = float(np.mean(np.asarray(naive_a) - true_a))
    ta_bias = float(np.mean(np.asarray(ta) - true_a))
    cond_bias = float(np.mean(np.asarray(cond) - true_a))
    assert naive_bias > 0.15
    assert abs(ta_bias) < abs(naive_bias)
    assert abs(ta_bias) < 0.35
    assert cond_bias > ta_bias


def test_recovery_rho_and_joint():
    rng = np.random.default_rng(5)
    n = 2500
    tau = 0.02
    se_p, se_g = 0.015, 0.012
    for rho_true in (-0.5, 0.0, 0.5):
        delta = tau * rng.standard_normal(n)
        gamma = rho_true * delta + tau * np.sqrt(1 - rho_true**2) * rng.standard_normal(n)
        x = delta + rng.normal(0, se_p, n)
        g = gamma + rng.normal(0, se_g, n)
        hat = estimate_guardrail_rho(x, np.full(n, se_p), g, np.full(n, se_g))["rho"]
        assert abs(hat - rho_true) < 0.12

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
    assert mse_j < 0.85 * mse_u

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
    assert mse_h < mse_u


def test_recovery_one_sided_mue_and_map():
    rng = np.random.default_rng(7)
    true, se_w, alpha = 0.50, 0.25, 0.05
    z1 = norm.ppf(1.0 - alpha)
    y = rng.normal(true, se_w, 40_000)
    winners = y[y >= z1 * se_w]
    corrected = np.array(
        [winners_curse_estimate(float(w), se_w, alpha=alpha, alternative="greater")["corrected"] for w in winners[::8]]
    )
    assert abs(float(np.median(corrected)) - true) < 0.08
    assert abs(float(np.mean(winners)) - true) > 0.10

    rng = np.random.default_rng(8)
    n = 400
    tau_m, mu_m, se_m = 0.025, 0.005, 0.02
    theta = rng.normal(mu_m, tau_m, n)
    y = theta + rng.normal(0, se_m, n)
    fit = fit_normal_prior_map(y, np.full(n, se_m))
    assert abs(np.sqrt(fit["tau2"]) - tau_m) < 0.012
    assert abs(fit["prior_mean"] - mu_m) < 0.01
