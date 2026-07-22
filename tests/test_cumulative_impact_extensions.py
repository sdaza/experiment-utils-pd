"""Tests for Airbnb process-level total, rho estimation, MAP prior, analyzer wrapper."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from experiment_utils import (
    ExperimentAnalyzer,
    cumulative_impact,
    estimate_guardrail_rho,
    fit_normal_prior_map,
    joint_metric_shrinkage,
    process_level_total_effect,
)
from experiment_utils.utils import empirical_bayes_shrinkage


def _fitted_analyzer(results: pd.DataFrame) -> ExperimentAnalyzer:
    ea = ExperimentAnalyzer(
        data=pd.DataFrame({"treatment": [0, 1], "y": [0.0, 1.0]}),
        outcomes=["y"],
        treatment_col="treatment",
    )
    ea._results = results
    return ea


# ---------------------------------------------------------------------------
# process_level_total_effect (Airbnb)
# ---------------------------------------------------------------------------


def test_process_level_keys_and_selected():
    y = np.array([0.05, 0.01, -0.02, 0.08])
    s = np.full(4, 0.02)
    out = process_level_total_effect(y, s, alpha=0.05, alternative="greater")
    z = norm.ppf(0.95)
    expected_sel = y > z * s
    assert np.array_equal(out["selected_mask"], expected_sel)
    assert out["n_selected"] == int(expected_sel.sum())
    assert out["naive_total"] == pytest.approx(float(y[expected_sel].sum()))
    assert out["total"] < out["naive_total"]  # positive bias subtracted
    assert set(out) >= {
        "total",
        "naive_total",
        "conditional_total",
        "bias_estimate",
        "n_selected",
        "selected_mask",
    }


def test_process_level_less_sign_symmetry():
    y = np.array([0.05, 0.01, -0.06, 0.02])
    s = np.full(4, 0.02)
    pos = process_level_total_effect(y, s, alternative="greater")
    neg = process_level_total_effect(-y, s, alternative="less")
    assert neg["total"] == pytest.approx(-pos["total"], abs=1e-8)
    assert neg["naive_total"] == pytest.approx(-pos["naive_total"], abs=1e-8)


def test_process_level_bootstrap_ci():
    rng = np.random.default_rng(0)
    y = rng.normal(0.02, 0.03, 40)
    s = np.full(40, 0.025)
    out = process_level_total_effect(y, s, n_bootstrap=400, random_seed=1)
    assert "ci_lower" in out and "ci_upper" in out
    assert out["ci_lower"] < out["ci_upper"]
    assert np.isfinite([out["ci_lower"], out["ci_upper"], out["total"]]).all()


def test_process_level_rejects_two_sided():
    with pytest.raises(ValueError, match="greater"):
        process_level_total_effect([0.1], [0.05], alternative="two-sided")


def test_process_level_recovers_better_than_naive():
    """Lee & Shen claim: hat T_A is closer to E[T_A] than S_A; cond total still high."""
    rng = np.random.default_rng(21)
    n_exp, n_sim = 30, 400
    # Mildly positive true-effect distribution (as in their sim spirit)
    true_sums, naive_sums, ta_sums, cond_sums = [], [], [], []
    for _ in range(n_sim):
        a = rng.normal(0.2, 0.5, n_exp)
        s = np.sqrt(rng.choice([0.3, 0.5, 0.8], size=n_exp))
        x = rng.normal(a, s)
        z = norm.ppf(0.95)
        sel = x > z * s
        if sel.sum() < 1:
            continue
        true_sums.append(float(a[sel].sum()))
        out = process_level_total_effect(x, s, alpha=0.05, alternative="greater")
        naive_sums.append(out["naive_total"])
        ta_sums.append(out["total"])
        cond_sums.append(out["conditional_total"])

    true_sums = np.asarray(true_sums)
    naive_bias = float(np.mean(np.asarray(naive_sums) - true_sums))
    ta_bias = float(np.mean(np.asarray(ta_sums) - true_sums))
    cond_bias = float(np.mean(np.asarray(cond_sums) - true_sums))
    assert naive_bias > 0.15
    assert abs(ta_bias) < abs(naive_bias)
    assert abs(ta_bias) < 0.35
    assert cond_bias > ta_bias  # conditional still overstates the process total


# ---------------------------------------------------------------------------
# estimate_guardrail_rho
# ---------------------------------------------------------------------------


def test_estimate_rho_recovers():
    rng = np.random.default_rng(5)
    n = 2500
    tau = 0.02
    for rho_true in (-0.5, 0.5):
        delta = tau * rng.standard_normal(n)
        gamma = rho_true * delta + tau * np.sqrt(1 - rho_true**2) * rng.standard_normal(n)
        se_p, se_g = 0.015, 0.012
        x = delta + rng.normal(0, se_p, n)
        g = gamma + rng.normal(0, se_g, n)
        out = estimate_guardrail_rho(x, np.full(n, se_p), g, np.full(n, se_g))
        assert abs(out["rho"] - rho_true) < 0.12


def test_estimate_rho_validation():
    with pytest.raises(ValueError, match="at least 5"):
        estimate_guardrail_rho([0.1] * 3, [0.05] * 3, [0.0] * 3, [0.05] * 3)


# ---------------------------------------------------------------------------
# fit_normal_prior_map
# ---------------------------------------------------------------------------


def test_map_prior_recovers_tau():
    rng = np.random.default_rng(8)
    n = 400
    tau, mu = 0.025, 0.005
    se = 0.02
    theta = rng.normal(mu, tau, n)
    y = theta + rng.normal(0, se, n)
    fit = fit_normal_prior_map(y, np.full(n, se))
    assert fit["method"] == "half_cauchy_map"
    assert abs(np.sqrt(fit["tau2"]) - tau) < 0.012
    assert abs(fit["prior_mean"] - mu) < 0.01


def test_map_usable_as_cumulative_prior():
    rng = np.random.default_rng(2)
    y = rng.normal(0, 0.03, 50)
    s = np.full(50, 0.02)
    ship = y > 1.96 * s
    if ship.sum() < 1:
        ship[np.argmax(y)] = True
    out = cumulative_impact(y, s, shipped=ship, prior="map")
    assert out["prior_family"] == "normal"
    assert out["tau2"] >= 0
    fit = fit_normal_prior_map(y, s)
    out2 = cumulative_impact(y, s, shipped=ship, prior=fit)
    assert out["cumulative"] == pytest.approx(out2["cumulative"], rel=1e-6)


def test_map_requires_five():
    with pytest.raises(ValueError, match="at least 5"):
        fit_normal_prior_map([0.1, 0.2], [0.05, 0.05])


# ---------------------------------------------------------------------------
# joint with estimated rho
# ---------------------------------------------------------------------------


def test_joint_with_estimated_rho_beats_primary_only():
    rng = np.random.default_rng(11)
    n = 3000
    tau, rho = 0.015, 0.75
    se_p, se_g = 0.03, 0.01
    delta = tau * rng.standard_normal(n)
    gamma = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x = delta + rng.normal(0, se_p, n)
    g = gamma + rng.normal(0, se_g, n)
    est = estimate_guardrail_rho(x, np.full(n, se_p), g, np.full(n, se_g))
    joint = joint_metric_shrinkage(
        x,
        np.full(n, se_p),
        g,
        np.full(n, se_g),
        rho=est["rho"],
        prior_sd_primary=est["tau_primary"],
        prior_sd_guard=est["tau_guard"],
    )
    uni = empirical_bayes_shrinkage(x, np.full(n, se_p), tau2=est["tau_primary"] ** 2)
    assert np.mean((joint["primary_shrunk"] - delta) ** 2) < np.mean((uni["shrunk"] - delta) ** 2)


# ---------------------------------------------------------------------------
# Analyzer wrapper
# ---------------------------------------------------------------------------


def test_analyzer_cumulative_impact_summary_matches_standalone():
    df = pd.DataFrame(
        {
            "outcome": ["m"] * 6,
            "absolute_effect": [0.05, 0.02, 0.08, 0.01, -0.01, 0.04],
            "standard_error": [0.02] * 6,
            "shipped": [True, False, True, True, False, True],
            "effect_type": ["mean_difference"] * 6,
            "control_value": [1.0] * 6,
            "stat_significance": [1] * 6,
        }
    )
    ea = _fitted_analyzer(df)
    tau2 = 0.015**2
    out = ea.cumulative_impact_summary(shipped="shipped", tau2=tau2)
    direct = cumulative_impact(
        df["absolute_effect"],
        df["standard_error"],
        shipped=df["shipped"].to_numpy(),
        tau2=tau2,
    )
    assert out["cumulative"] == pytest.approx(direct["cumulative"])
    assert "details" in out
    assert len(out["details"]) == 6


def test_analyzer_cumulative_requires_get_effects():
    ea = ExperimentAnalyzer(
        data=pd.DataFrame({"treatment": [0, 1], "y": [0.0, 1.0]}),
        outcomes=["y"],
        treatment_col="treatment",
    )
    with pytest.raises(ValueError, match="get_effects"):
        ea.cumulative_impact_summary()
