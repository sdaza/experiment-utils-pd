"""Tests + Monte Carlo recovery for joint_metric_shrinkage_mvn."""

import numpy as np
import pytest
from scipy.stats import norm

from experiment_utils.shrinkage import (
    aggregate_shrunk_cumulative,
    cumulative_impact,
    empirical_bayes_shrinkage,
    joint_metric_shrinkage,
    joint_metric_shrinkage_mvn,
    nss_adjusted_cumulative_impact_mvn,
)


def _as_guard_list(g: np.ndarray, sg: np.ndarray):
    """(n, K) -> list of columns."""
    return [g[:, k] for k in range(g.shape[1])], [sg[:, k] for k in range(sg.shape[1])]


def test_k1_matches_bivariate():
    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(0.01, 0.03, n)
    g = rng.normal(0.0, 0.02, n)
    se_p = np.full(n, 0.02)
    se_g = np.full(n, 0.012)
    rho, tau = 0.55, 0.018
    bi = joint_metric_shrinkage(x, se_p, g, se_g, rho=rho, prior_sd_primary=tau, prior_sd_guard=tau)
    mvn = joint_metric_shrinkage_mvn(
        x,
        se_p,
        [g],
        [se_g],
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    assert np.allclose(mvn["primary_shrunk"], bi["primary_shrunk"], atol=1e-10)
    assert np.allclose(mvn["primary_posterior_sd"], bi["primary_posterior_sd"], atol=1e-10)
    assert np.allclose(mvn["guard_shrunk"][:, 0], bi["guard_shrunk"], atol=1e-10)


def test_rho0_matches_univariate_eb():
    rng = np.random.default_rng(1)
    n = 50
    tau = 0.02
    x = rng.normal(0.0, 0.03, n)
    se_p = np.full(n, 0.025)
    g = np.column_stack([rng.normal(0, 0.02, n) for _ in range(3)])
    se_g = np.full_like(g, 0.015)
    gl, gsl = _as_guard_list(g, se_g)
    mvn = joint_metric_shrinkage_mvn(
        x,
        se_p,
        gl,
        gsl,
        rho_primary=[0.0, 0.0, 0.0],
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    univ = empirical_bayes_shrinkage(x, se_p, prior_mean=0.0, tau2=tau**2)
    assert np.allclose(mvn["primary_shrunk"], univ["shrunk"], atol=1e-10)
    assert np.allclose(mvn["primary_posterior_sd"], univ["posterior_sd"], atol=1e-10)


def test_nss_mvn_equals_compose():
    rng = np.random.default_rng(2)
    n = 40
    x = rng.normal(0.01, 0.03, n)
    se_p = np.full(n, 0.02)
    g = np.column_stack([rng.normal(0, 0.02, n) for _ in range(2)])
    se_g = np.full_like(g, 0.012)
    gl, gsl = _as_guard_list(g, se_g)
    ship = x > 1.5 * se_p
    if ship.sum() < 2:
        ship[:3] = True
    rho = [0.4, 0.3]
    tau = 0.015
    nss = nss_adjusted_cumulative_impact_mvn(
        x,
        se_p,
        gl,
        gsl,
        shipped=ship,
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    joint = joint_metric_shrinkage_mvn(
        x,
        se_p,
        gl,
        gsl,
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    agg = aggregate_shrunk_cumulative(
        joint["primary_shrunk"],
        joint["primary_posterior_sd"],
        shipped=ship,
        observed=x,
    )
    assert nss["cumulative"] == pytest.approx(agg["cumulative"])
    assert nss["ci_lower"] == pytest.approx(agg["ci_lower"])
    assert np.allclose(nss["shrunk"], joint["primary_shrunk"])


def test_accepts_2d_array_and_flexible_k():
    rng = np.random.default_rng(3)
    n = 20
    x = rng.normal(0, 0.02, n)
    se_p = np.full(n, 0.02)
    for k in (1, 3, 5, 10):
        g = rng.normal(0, 0.02, size=(n, k))
        se_g = np.full((n, k), 0.015)
        out = joint_metric_shrinkage_mvn(
            x,
            se_p,
            g,
            se_g,
            rho_primary=0.2,
            prior_sd_primary=0.015,
        )
        assert out["primary_shrunk"].shape == (n,)
        assert out["guard_shrunk"].shape == (n, k)
        assert np.all(np.isfinite(out["primary_shrunk"]))


def test_validation():
    with pytest.raises(ValueError, match="rho"):
        joint_metric_shrinkage_mvn(
            [0.1],
            [0.05],
            [[0.0]],
            [[0.05]],
            rho_primary=1.0,
            prior_sd_primary=0.02,
        )
    with pytest.raises(ValueError, match="positive definite|correlation|PD"):
        R = np.array([[1.0, 0.99], [0.99, 1.0]])
        joint_metric_shrinkage_mvn(
            [0.1, 0.2, 0.0],
            [0.05, 0.05, 0.05],
            np.zeros((3, 2)),
            np.full((3, 2), 0.05),
            rho_primary=[0.99, -0.99],
            prior_sd_primary=0.02,
            prior_sd_guard=0.02,
            rho_guardrails=R,
        )


def test_prior_map_and_t_scale_plug_in():
    """prior='map' / t-scale resolve to prior_sd; match explicit SD when plugged."""
    from experiment_utils.shrinkage import fit_t_prior, resolve_mvn_prior_sd

    rng = np.random.default_rng(9)
    n = 60
    x = rng.normal(0.0, 0.03, n)
    se_p = np.full(n, 0.025)
    g = np.column_stack([rng.normal(0, 0.02, n) for _ in range(2)])
    se_g = np.full_like(g, 0.015)

    resolved_map = resolve_mvn_prior_sd(x, se_p, prior="map")
    mvn_map = joint_metric_shrinkage_mvn(x, se_p, g, se_g, rho_primary=[0.5, 0.4], prior="map")
    assert mvn_map["prior_source"] == "half_cauchy_map"
    assert mvn_map["prior_sd_primary"] == pytest.approx(resolved_map["prior_sd"])

    t_fit = fit_t_prior(x, se_p, df=4.0, prior_mean=0.0)
    mvn_t = joint_metric_shrinkage_mvn(
        x,
        se_p,
        g,
        se_g,
        rho_primary=[0.5, 0.4],
        prior={"scale": t_fit["scale"], "df": 4.0, "prior_mean": 0.0},
    )
    assert mvn_t["prior_source"] == "t_scale_plug_in"
    assert mvn_t["prior_sd_primary"] == pytest.approx(t_fit["scale"])

    # Same τ via prior= vs prior_sd_primary → identical primary
    tau = 0.02
    a = joint_metric_shrinkage_mvn(x, se_p, g, se_g, rho_primary=[0.5, 0.4], prior={"tau2": tau**2})
    b = joint_metric_shrinkage_mvn(x, se_p, g, se_g, rho_primary=[0.5, 0.4], prior_sd_primary=tau)
    assert np.allclose(a["primary_shrunk"], b["primary_shrunk"], atol=1e-10)

    with pytest.raises(ValueError, match="only one of prior"):
        joint_metric_shrinkage_mvn(x, se_p, g, se_g, rho_primary=0.3, prior="map", prior_sd_primary=0.02)


def _sim_independent_guards(rng, n, tau, rhos, se_p, se_g):
    k = len(rhos)
    delta = tau * rng.standard_normal(n)
    g_true = np.empty((n, k))
    for j, rho in enumerate(rhos):
        g_true[:, j] = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x = delta + rng.normal(0, se_p, n)
    g = g_true + rng.normal(0, se_g, size=(n, k))
    return delta, x, g


def test_r1_mvn_mse_beats_primary_only():
    """R1: K=3 informative independent guards → MVN primary MSE < univariate EB."""
    rng = np.random.default_rng(42)
    n_exp, n_sim = 100, 250
    tau = 0.015
    rhos = (0.8, 0.7, 0.6)
    se_p, se_g = 0.03, 0.01
    mse_univ, mse_mvn = [], []
    for _ in range(n_sim):
        delta, x, g = _sim_independent_guards(rng, n_exp, tau, rhos, se_p, se_g)
        se_p_a = np.full(n_exp, se_p)
        se_g_a = np.full_like(g, se_g)
        univ = empirical_bayes_shrinkage(x, se_p_a, prior_mean=0.0, tau2=tau**2)
        mvn = joint_metric_shrinkage_mvn(
            x,
            se_p_a,
            g,
            se_g_a,
            rho_primary=list(rhos),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        mse_univ.append(float(np.mean((univ["shrunk"] - delta) ** 2)))
        mse_mvn.append(float(np.mean((mvn["primary_shrunk"] - delta) ** 2)))
    assert float(np.mean(mse_mvn)) < float(np.mean(mse_univ))


def test_r2_mvn_mse_le_best_bivariate():
    """R2: MVN MSE ≤ best single-companion bivariate (+ small tol)."""
    rng = np.random.default_rng(43)
    n_exp, n_sim = 100, 200
    tau = 0.015
    rhos = (0.8, 0.7, 0.6)
    se_p, se_g = 0.03, 0.01
    mse_mvn, mse_best_bi = [], []
    for _ in range(n_sim):
        delta, x, g = _sim_independent_guards(rng, n_exp, tau, rhos, se_p, se_g)
        se_p_a = np.full(n_exp, se_p)
        se_g_a = np.full_like(g, se_g)
        mvn = joint_metric_shrinkage_mvn(
            x,
            se_p_a,
            g,
            se_g_a,
            rho_primary=list(rhos),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        bi_mses = []
        for j, rho in enumerate(rhos):
            bi = joint_metric_shrinkage(
                x,
                se_p_a,
                g[:, j],
                se_g_a[:, j],
                rho=rho,
                prior_sd_primary=tau,
                prior_sd_guard=tau,
            )
            bi_mses.append(float(np.mean((bi["primary_shrunk"] - delta) ** 2)))
        mse_mvn.append(float(np.mean((mvn["primary_shrunk"] - delta) ** 2)))
        mse_best_bi.append(min(bi_mses))
    assert float(np.mean(mse_mvn)) <= float(np.mean(mse_best_bi)) * 1.02


def test_r3_primary_ci_coverage_fixed_sigma():
    """R3: primary CI coverage ≈ 0.95 under known Sigma."""
    rng = np.random.default_rng(7)
    n_exp, n_sim = 80, 350
    tau = 0.02
    rhos = (0.6, 0.5, 0.4)
    se_p, se_g = 0.025, 0.012
    covered = 0
    total = 0
    for _ in range(n_sim):
        delta, x, g = _sim_independent_guards(rng, n_exp, tau, rhos, se_p, se_g)
        mvn = joint_metric_shrinkage_mvn(
            x,
            np.full(n_exp, se_p),
            g,
            np.full_like(g, se_g),
            rho_primary=list(rhos),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
            ci=0.95,
        )
        lo, hi = mvn["primary_ci_lower"], mvn["primary_ci_upper"]
        covered += int(np.sum((lo <= delta) & (delta <= hi)))
        total += n_exp
    rate = covered / total
    assert 0.88 <= rate <= 0.99


def test_r4_redundant_guards_still_beat_primary():
    """R4: highly correlated guards via R; MVN still beats primary-only MSE."""
    rng = np.random.default_rng(11)
    n_exp, n_sim = 100, 200
    tau, rho = 0.015, 0.75
    se_p, se_g = 0.03, 0.01
    R = np.array([[1.0, 0.9], [0.9, 1.0]])
    mse_univ, mse_mvn = [], []
    for _ in range(n_sim):
        sd = np.array([tau, tau, tau])
        corr = np.array(
            [
                [1.0, rho, rho],
                [rho, 1.0, 0.9],
                [rho, 0.9, 1.0],
            ]
        )
        sigma = np.outer(sd, sd) * corr
        m = rng.multivariate_normal(np.zeros(3), sigma, size=n_exp)
        delta, g1, g2 = m[:, 0], m[:, 1], m[:, 2]
        x = delta + rng.normal(0, se_p, n_exp)
        g = np.column_stack([g1 + rng.normal(0, se_g, n_exp), g2 + rng.normal(0, se_g, n_exp)])
        univ = empirical_bayes_shrinkage(x, np.full(n_exp, se_p), prior_mean=0.0, tau2=tau**2)
        mvn = joint_metric_shrinkage_mvn(
            x,
            np.full(n_exp, se_p),
            g,
            np.full_like(g, se_g),
            rho_primary=[rho, rho],
            prior_sd_primary=tau,
            prior_sd_guard=tau,
            rho_guardrails=R,
        )
        mse_univ.append(float(np.mean((univ["shrunk"] - delta) ** 2)))
        mse_mvn.append(float(np.mean((mvn["primary_shrunk"] - delta) ** 2)))
        assert np.all(np.isfinite(mvn["primary_shrunk"]))
    assert float(np.mean(mse_mvn)) < float(np.mean(mse_univ))


def test_r5_cumulative_recovers_vs_naive():
    """R5: MVN cumulative |bias| much smaller than naive shipped sum."""
    rng = np.random.default_rng(21)
    n_exp, n_sim = 100, 250
    tau = 0.015
    rhos = (0.75, 0.7, 0.65)
    se_p, se_g = 0.03, 0.01
    z = norm.ppf(0.975)
    true_sums, naive_sums, mvn_sums = [], [], []
    for _ in range(n_sim):
        delta, x, g = _sim_independent_guards(rng, n_exp, tau, rhos, se_p, se_g)
        ship = (x > z * se_p) & np.all(g >= 0.0, axis=1)
        if ship.sum() < 2:
            continue
        true_sums.append(float(delta[ship].sum()))
        naive_sums.append(float(x[ship].sum()))
        nss = nss_adjusted_cumulative_impact_mvn(
            x,
            np.full(n_exp, se_p),
            g,
            np.full_like(g, se_g),
            shipped=ship,
            rho_primary=list(rhos),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        mvn_sums.append(nss["cumulative"])
    true_sums = np.asarray(true_sums)
    naive_bias = float(np.mean(np.asarray(naive_sums) - true_sums))
    mvn_bias = float(np.mean(np.asarray(mvn_sums) - true_sums))
    assert naive_bias > 0.05
    assert abs(mvn_bias) < abs(naive_bias)
    assert abs(mvn_bias) < 0.5 * abs(naive_bias)


def test_r6_plugin_rho_beats_primary_only():
    """R6: MoM rho per pair (known τ) still beats primary-only MSE on average."""
    rng = np.random.default_rng(33)
    n_exp, n_sim = 120, 200
    tau = 0.018
    rhos = (0.75, 0.7, 0.65)
    se_p, se_g = 0.028, 0.012
    mse_univ, mse_mvn = [], []
    for _ in range(n_sim):
        delta, x, g = _sim_independent_guards(rng, n_exp, tau, rhos, se_p, se_g)
        se_p_a = np.full(n_exp, se_p)
        se_g_a = np.full_like(g, se_g)
        univ = cumulative_impact(x, se_p_a, tau2=tau**2, prior_mean=0.0)
        nss = nss_adjusted_cumulative_impact_mvn(
            x,
            se_p_a,
            g,
            se_g_a,
            shipped=np.ones(n_exp, dtype=bool),
            # estimate rho only; fix τ to avoid PM→0 edge cases
            prior_sd_primary=tau,
            prior_sd_guard=tau,
        )
        mse_univ.append(float(np.mean((univ["shrunk"] - delta) ** 2)))
        mse_mvn.append(float(np.mean((nss["shrunk"] - delta) ** 2)))
    assert float(np.mean(mse_mvn)) < float(np.mean(mse_univ))
