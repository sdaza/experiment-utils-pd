"""Tests for cumulative_impact and joint_metric_shrinkage."""

import numpy as np
import pytest
from scipy.stats import norm

from experiment_utils.shrinkage import (
    cumulative_impact,
    empirical_bayes_shrinkage,
    joint_metric_shrinkage,
    t_prior_shrinkage,
)

# ---------------------------------------------------------------------------
# joint_metric_shrinkage
# ---------------------------------------------------------------------------


def test_joint_rho0_matches_univariate():
    x = np.array([0.05, 0.02, -0.01])
    sx = np.array([0.02, 0.03, 0.015])
    g = np.array([0.01, -0.02, 0.03])
    sg = np.array([0.01, 0.01, 0.02])
    tau = 0.02
    joint = joint_metric_shrinkage(x, sx, g, sg, rho=0.0, prior_sd_primary=tau, prior_sd_guard=tau)
    uni = empirical_bayes_shrinkage(x, sx, tau2=tau**2)
    assert joint["primary_shrunk"] == pytest.approx(uni["shrunk"], abs=1e-10)
    assert joint["primary_posterior_sd"] == pytest.approx(uni["posterior_sd"], abs=1e-10)


def test_joint_validation():
    with pytest.raises(ValueError, match="rho"):
        joint_metric_shrinkage([0.1], [0.05], [0.0], [0.05], rho=1.0, prior_sd_primary=0.02)
    with pytest.raises(ValueError, match="prior_sd_primary"):
        joint_metric_shrinkage([0.1], [0.05], [0.0], [0.05], rho=0.0, prior_sd_primary=0.0)
    with pytest.raises(ValueError, match="same length"):
        joint_metric_shrinkage([0.1, 0.2], [0.05, 0.05], [0.0], [0.05], rho=0.0, prior_sd_primary=0.02)


def test_joint_recovery_when_rho_large():
    """With strong alignment and a precise guardrail, joint beats primary-only."""
    rng = np.random.default_rng(11)
    n = 5_000
    tau = 0.015
    rho = 0.8
    se_p, se_g = 0.03, 0.01
    delta = tau * rng.standard_normal(n)
    gamma = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x = delta + rng.normal(0.0, se_p, n)
    g = gamma + rng.normal(0.0, se_g, n)

    joint = joint_metric_shrinkage(x, np.full(n, se_p), g, np.full(n, se_g), rho=rho, prior_sd_primary=tau)
    primary_only = empirical_bayes_shrinkage(x, np.full(n, se_p), tau2=tau**2)
    err_joint = np.mean((joint["primary_shrunk"] - delta) ** 2)
    err_primary = np.mean((primary_only["shrunk"] - delta) ** 2)
    assert err_joint < err_primary * 0.85


# ---------------------------------------------------------------------------
# cumulative_impact — API
# ---------------------------------------------------------------------------


def test_cumulative_sum_matches_manual_shrink():
    y = np.array([0.05, 0.02, 0.08, -0.01])
    s = np.full(4, 0.025)
    shipped = np.array([True, False, True, False])
    tau2 = 0.015**2
    out = cumulative_impact(y, s, shipped=shipped, tau2=tau2, aggregation="sum")
    eb = empirical_bayes_shrinkage(y, s, tau2=tau2)
    assert out["cumulative"] == pytest.approx(float(eb["shrunk"][shipped].sum()))
    assert out["n_shipped"] == 2
    assert out["n_total"] == 4
    assert out["prior_family"] == "normal"
    assert out["ci_lower"] < out["cumulative"] < out["ci_upper"]


def test_cumulative_product_and_coverage():
    y = np.array([0.02, 0.03, 0.01])
    s = np.full(3, 0.01)
    cov = np.array([1.0, 0.5, 1.0])
    out = cumulative_impact(y, s, tau2=0.02**2, aggregation="product", coverage=cov, shipped=[True, True, True])
    eb = empirical_bayes_shrinkage(y, s, tau2=0.02**2)
    expected = float(np.prod(1.0 + cov * eb["shrunk"]) - 1.0)
    assert out["cumulative"] == pytest.approx(expected)
    assert out["aggregation"] == "product"


def test_cumulative_t_prior():
    y = np.array([0.04, 0.01, 0.06, 0.02, 0.03])
    s = np.full(5, 0.02)
    prior = {"scale": 0.02, "df": 4.0, "prior_mean": 0.0}
    out = cumulative_impact(y, s, prior=prior, shipped=np.ones(5, dtype=bool))
    tp = t_prior_shrinkage(y, s, scale=0.02, df=4.0)
    assert out["prior_family"] == "student_t"
    assert out["cumulative"] == pytest.approx(float(tp["shrunk"].sum()))


def test_cumulative_validation():
    y = np.array([0.01, 0.02])
    s = np.array([0.01, 0.01])
    with pytest.raises(ValueError, match="aggregation"):
        cumulative_impact(y, s, aggregation="mean", tau2=0.01)
    with pytest.raises(ValueError, match="shipped"):
        cumulative_impact(y, s, shipped=[True], tau2=0.01)
    with pytest.raises(ValueError, match="at least 1 shipped"):
        cumulative_impact(y, s, shipped=[False, False], tau2=0.01, min_shipped=1)
    with pytest.raises(ValueError, match="coverage"):
        cumulative_impact(y, s, coverage=[1.5, 0.5], tau2=0.01)
    with pytest.raises(ValueError, match="prior"):
        cumulative_impact(y, s, prior={"foo": 1})


def test_cumulative_learns_tau2_when_unspecified():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.03, 30)
    s = np.full(30, 0.02)
    out = cumulative_impact(y, s)
    assert out["tau2"] >= 0
    assert out["n_shipped"] == 30


# ---------------------------------------------------------------------------
# Monte Carlo recovery
# ---------------------------------------------------------------------------


def test_cumulative_naive_biased_eb_recovers():
    """Kessler claim: sum of raw winners overstates; EB cumulative recovers under correct prior."""
    rng = np.random.default_rng(42)
    n_exp = 80
    n_sim = 250
    tau = 0.02
    se = 0.025
    z = norm.ppf(0.975)
    true_sums = []
    naive_sums = []
    eb_sums = []
    for _ in range(n_sim):
        theta = rng.normal(0.0, tau, n_exp)
        y = theta + rng.normal(0.0, se, n_exp)
        ship = y > z * se
        if ship.sum() < 1:
            continue
        true_sums.append(float(theta[ship].sum()))
        naive_sums.append(float(y[ship].sum()))
        out = cumulative_impact(
            y,
            np.full(n_exp, se),
            shipped=ship,
            tau2=tau**2,
            prior_mean=0.0,
        )
        eb_sums.append(out["cumulative"])

    true_sums = np.asarray(true_sums)
    naive_sums = np.asarray(naive_sums)
    eb_sums = np.asarray(eb_sums)
    mean_abs_true = float(np.mean(np.abs(true_sums))) + 1e-12
    naive_bias = float(np.mean(naive_sums - true_sums))
    eb_bias = float(np.mean(eb_sums - true_sums))
    assert naive_bias > 0.05 * mean_abs_true
    assert abs(eb_bias) < 0.03 * mean_abs_true or abs(eb_bias) < 0.02


def test_cumulative_sum_ci_coverage_fixed_prior():
    """With known prior, Kessler plug-in CI for the sum covers near nominally."""
    rng = np.random.default_rng(99)
    n_exp = 60
    n_sim = 400
    tau = 0.02
    se = 0.025
    z = norm.ppf(0.975)
    covered = 0
    n_used = 0
    for _ in range(n_sim):
        theta = rng.normal(0.0, tau, n_exp)
        y = theta + rng.normal(0.0, se, n_exp)
        ship = y > z * se
        if ship.sum() < 2:
            continue
        truth = float(theta[ship].sum())
        out = cumulative_impact(y, np.full(n_exp, se), shipped=ship, tau2=tau**2, ci=0.95)
        n_used += 1
        if out["ci_lower"] <= truth <= out["ci_upper"]:
            covered += 1
    rate = covered / n_used
    assert n_used > 200
    # Plug-in CI; allow MC noise around 0.95
    assert 0.90 <= rate <= 0.99


def test_cumulative_product_positive_lifts():
    """Product aggregation recovers multiplicative portfolio impact under fixed prior."""
    rng = np.random.default_rng(3)
    n_exp = 40
    n_sim = 200
    tau = 0.01
    se = 0.008
    z = norm.ppf(0.975)
    err_naive = []
    err_eb = []
    for _ in range(n_sim):
        theta = rng.normal(0.005, tau, n_exp)  # mild positive mean so products stay > -1
        y = theta + rng.normal(0.0, se, n_exp)
        ship = y > z * se
        if ship.sum() < 1:
            continue
        true_prod = float(np.prod(1.0 + theta[ship]) - 1.0)
        naive = float(np.prod(1.0 + y[ship]) - 1.0)
        out = cumulative_impact(y, np.full(n_exp, se), shipped=ship, tau2=tau**2, aggregation="product")
        err_naive.append(naive - true_prod)
        err_eb.append(out["cumulative"] - true_prod)
    assert np.mean(err_naive) > np.mean(np.abs(err_eb)) * 0.5 or abs(np.mean(err_eb)) < abs(np.mean(err_naive))
    assert abs(np.mean(err_eb)) < abs(np.mean(err_naive))
