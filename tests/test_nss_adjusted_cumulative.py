"""Tests + Monte Carlo recovery for nss_adjusted_cumulative_impact."""

import numpy as np
import pytest
from scipy.stats import norm

from experiment_utils import (
    aggregate_shrunk_cumulative,
    cumulative_impact,
    joint_metric_shrinkage,
    nss_adjusted_cumulative_impact,
)


def test_aggregate_matches_manual_sum():
    shrunk = np.array([0.02, 0.01, 0.03])
    sd = np.array([0.01, 0.01, 0.01])
    ship = np.array([True, False, True])
    out = aggregate_shrunk_cumulative(shrunk, sd, shipped=ship, observed=np.array([0.05, 0.02, 0.08]))
    assert out["cumulative"] == pytest.approx(0.05)
    assert out["n_shipped"] == 2
    assert out["naive_sum"] == pytest.approx(0.13)
    assert out["retain_vs_naive"] == pytest.approx(0.05 / 0.13)


def test_nss_equals_compose_joint_then_aggregate():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0.01, 0.03, n)
    g = rng.normal(0.0, 0.02, n)
    se_p = np.full(n, 0.02)
    se_g = np.full(n, 0.012)
    ship = x > 1.5 * se_p
    if ship.sum() < 2:
        ship[:3] = True
    rho, tau = 0.4, 0.015
    nss = nss_adjusted_cumulative_impact(
        x,
        se_p,
        g,
        se_g,
        shipped=ship,
        rho=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    joint = joint_metric_shrinkage(
        x, se_p, g, se_g, rho=rho, prior_sd_primary=tau, prior_sd_guard=tau
    )
    agg = aggregate_shrunk_cumulative(
        joint["primary_shrunk"],
        joint["primary_posterior_sd"],
        shipped=ship,
        observed=x,
    )
    assert nss["cumulative"] == pytest.approx(agg["cumulative"])
    assert nss["ci_lower"] == pytest.approx(agg["ci_lower"])
    assert nss["ci_upper"] == pytest.approx(agg["ci_upper"])
    assert np.allclose(nss["shrunk"], joint["primary_shrunk"])


def test_nss_rho0_matches_univariate_cumulative():
    """With ρ=0 and normal prior, NSS primary posteriors = univ EB; aggregate matches."""
    rng = np.random.default_rng(1)
    n = 50
    tau = 0.02
    se = 0.025
    x = rng.normal(0.0, 0.03, n)
    g = rng.normal(0.0, 0.02, n)  # independent noise
    se_p = np.full(n, se)
    se_g = np.full(n, 0.015)
    ship = np.abs(x) > 1.5 * se
    if ship.sum() < 2:
        ship[:5] = True
    nss = nss_adjusted_cumulative_impact(
        x,
        se_p,
        g,
        se_g,
        shipped=ship,
        rho=0.0,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    univ = cumulative_impact(x, se_p, shipped=ship, tau2=tau**2, prior_mean=0.0)
    assert nss["cumulative"] == pytest.approx(univ["cumulative"], abs=1e-10)
    assert np.allclose(nss["shrunk"], univ["shrunk"], atol=1e-10)


def test_nss_validation():
    with pytest.raises(ValueError, match="same length"):
        nss_adjusted_cumulative_impact(
            [0.1, 0.2],
            [0.05, 0.05],
            [0.0],
            [0.05],
            rho=0.0,
            prior_sd_primary=0.02,
            prior_sd_guard=0.02,
        )
    with pytest.raises(ValueError, match="at least 5"):
        nss_adjusted_cumulative_impact([0.1, 0.2], [0.05, 0.05], [0.0, 0.1], [0.05, 0.05])


def test_nss_recovery_beats_naive_and_primary_when_rho_large():
    """Among shipped, NSS cumulative closer to true sum than naive; beats primary-only when ρ high."""
    rng = np.random.default_rng(42)
    n_exp, n_sim = 100, 250
    tau, rho = 0.015, 0.75
    se_p, se_g = 0.03, 0.01
    z = norm.ppf(0.975)
    true_sums, naive_sums, prim_sums, nss_sums = [], [], [], []
    for _ in range(n_sim):
        delta = tau * rng.standard_normal(n_exp)
        gamma = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n_exp)
        x = delta + rng.normal(0, se_p, n_exp)
        g = gamma + rng.normal(0, se_g, n_exp)
        # SCALED-like: positive primary win AND guardrail floor
        ship = (x > z * se_p) & (g >= 0.0)
        if ship.sum() < 2:
            continue
        true_sums.append(float(delta[ship].sum()))
        naive_sums.append(float(x[ship].sum()))
        prim = cumulative_impact(
            x, np.full(n_exp, se_p), shipped=ship, tau2=tau**2, prior_mean=0.0
        )
        prim_sums.append(prim["cumulative"])
        nss = nss_adjusted_cumulative_impact(
            x,
            np.full(n_exp, se_p),
            g,
            np.full(n_exp, se_g),
            shipped=ship,
            rho=rho,
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        nss_sums.append(nss["cumulative"])

    true_sums = np.asarray(true_sums)
    naive_err = float(np.mean(np.abs(np.asarray(naive_sums) - true_sums)))
    prim_err = float(np.mean(np.abs(np.asarray(prim_sums) - true_sums)))
    nss_err = float(np.mean(np.abs(np.asarray(nss_sums) - true_sums)))
    naive_bias = float(np.mean(np.asarray(naive_sums) - true_sums))
    nss_bias = float(np.mean(np.asarray(nss_sums) - true_sums))

    assert naive_bias > 0.05  # winner's curse
    assert nss_err < naive_err
    assert abs(nss_bias) < abs(naive_bias)
    # With precise guardrail + high ρ, NSS should beat primary-only on MAE
    assert nss_err < prim_err * 0.95


def test_nss_ci_coverage_fixed_prior():
    rng = np.random.default_rng(7)
    n_exp, n_sim = 80, 350
    tau, rho = 0.02, 0.5
    se_p, se_g = 0.025, 0.012
    z = norm.ppf(0.975)
    covered = n_used = 0
    for _ in range(n_sim):
        delta = tau * rng.standard_normal(n_exp)
        gamma = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n_exp)
        x = delta + rng.normal(0, se_p, n_exp)
        g = gamma + rng.normal(0, se_g, n_exp)
        ship = (x > z * se_p) & (g >= 0.0)
        if ship.sum() < 2:
            continue
        out = nss_adjusted_cumulative_impact(
            x,
            np.full(n_exp, se_p),
            g,
            np.full(n_exp, se_g),
            shipped=ship,
            rho=rho,
            prior_sd_primary=tau,
            prior_sd_guard=tau,
            ci=0.95,
        )
        truth = float(delta[ship].sum())
        n_used += 1
        covered += int(out["ci_lower"] <= truth <= out["ci_upper"])
    assert n_used > 150
    rate = covered / n_used
    assert 0.88 <= rate <= 0.99
