"""Tests for MVN self-slot exclusion via primary_metric=."""

import numpy as np
import pytest

from experiment_utils.shrinkage import (
    aggregate_shrunk_cumulative,
    empirical_bayes_shrinkage,
    estimate_guardrail_rho,
    exclude_index_from_targets,
    joint_metric_shrinkage_mvn,
    nss_adjusted_cumulative_impact_mvn,
)


def test_exclude_index_from_targets():
    names = ["a", "b", "c"]
    idx = exclude_index_from_targets(["b", "other", "a"], names)
    assert list(idx) == [1, -1, 0]


def test_primary_metric_none_matches_legacy():
    rng = np.random.default_rng(0)
    n, k = 40, 3
    x = rng.normal(0.01, 0.03, n)
    se_p = np.full(n, 0.02)
    g = rng.normal(0.0, 0.02, size=(n, k))
    se_g = np.full_like(g, 0.012)
    rho = [0.5, 0.4, 0.3]
    tau = 0.018
    a = joint_metric_shrinkage_mvn(x, se_p, g, se_g, rho_primary=rho, prior_sd_primary=tau, prior_sd_guard=tau)
    b = joint_metric_shrinkage_mvn(
        x,
        se_p,
        g,
        se_g,
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
        guardrail_names=["a", "b", "c"],
        primary_metric=["other"] * n,
    )
    assert np.allclose(a["primary_shrunk"], b["primary_shrunk"], atol=1e-10)
    assert b["primary_metric_excluded"] is False


def test_primary_metric_drops_self_slot():
    rng = np.random.default_rng(7)
    n_non, n_a = 20, 25
    n = n_non + n_a
    k = 3
    names = ["a", "b", "c"]
    x = rng.normal(0.01, 0.02, n)
    se_p = np.full(n, 0.01)
    g = rng.normal(0.005, 0.02, (n, k))
    se_g = np.full((n, k), 0.01)
    primary_metric = ["other"] * n_non + ["a"] * n_a
    g[n_non:, 0] = np.nan
    se_g[n_non:, 0] = np.nan
    out = joint_metric_shrinkage_mvn(
        x,
        se_p,
        g,
        se_g,
        rho_primary=0.4,
        prior_sd_primary=0.02,
        prior_sd_guard=0.02,
        guardrail_names=names,
        primary_metric=primary_metric,
    )
    assert out["primary_metric_excluded"] is True
    assert out["strata_n"]["all_k"] == n_non
    assert out["strata_n"]["drop:a"] == n_a
    assert np.isfinite(out["primary_shrunk"]).all()


def test_companion_only_rho_not_pulled_by_self_copies():
    rng = np.random.default_rng(11)
    n, n_self = 100, 70
    true_rho = 0.45
    tau, se = 0.02, 0.008
    z = rng.standard_normal(n)
    delta = tau * z
    gamma0 = tau * (true_rho * z + np.sqrt(1.0 - true_rho**2) * rng.standard_normal(n))
    y = delta + se * rng.standard_normal(n)
    g = np.column_stack(
        [
            gamma0 + se * rng.standard_normal(n),
            tau * rng.standard_normal(n) + se * rng.standard_normal(n),
            tau * rng.standard_normal(n) + se * rng.standard_normal(n),
        ]
    )
    se_y = np.full(n, se)
    se_g = np.full((n, 3), se)
    primary_metric = ["a"] * n_self + ["other"] * (n - n_self)
    g_filled = g.copy()
    g_filled[:n_self, 0] = y[:n_self]

    rho_naive = float(estimate_guardrail_rho(y, se_y, g_filled[:, 0], se_g[:, 0])["rho"])
    excl = nss_adjusted_cumulative_impact_mvn(
        y,
        se_y,
        g_filled,
        se_g,
        shipped=np.ones(n, dtype=bool),
        guardrail_names=["a", "b", "c"],
        primary_metric=primary_metric,
    )
    rho_excl = float(np.asarray(excl["rho_primary"])[0])
    assert rho_naive > 0.8
    assert rho_naive > rho_excl + 0.2
    assert abs(rho_excl - true_rho) < abs(rho_naive - true_rho)
    assert abs(rho_excl - true_rho) < 0.25
    assert excl["rho_info"]["source"] == "mom_companion_only"


def test_nss_primary_metric_compose():
    rng = np.random.default_rng(2)
    n, k = 40, 3
    names = ["a", "b", "c"]
    x = rng.normal(0.01, 0.03, n)
    se_p = np.full(n, 0.02)
    g = rng.normal(0, 0.02, size=(n, k))
    se_g = np.full_like(g, 0.012)
    primary_metric = ["other"] * 20 + ["b"] * 20
    g[20:, 1] = np.nan
    se_g[20:, 1] = np.nan
    ship = np.ones(n, dtype=bool)
    rho = [0.4, 0.3, 0.35]
    tau = 0.015
    nss = nss_adjusted_cumulative_impact_mvn(
        x,
        se_p,
        g,
        se_g,
        shipped=ship,
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
        guardrail_names=names,
        primary_metric=primary_metric,
    )
    joint = joint_metric_shrinkage_mvn(
        x,
        se_p,
        g,
        se_g,
        rho_primary=rho,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
        guardrail_names=names,
        primary_metric=primary_metric,
    )
    agg = aggregate_shrunk_cumulative(
        joint["primary_shrunk"],
        joint["primary_posterior_sd"],
        shipped=ship,
        observed=x,
    )
    assert nss["cumulative"] == pytest.approx(agg["cumulative"])
    assert np.allclose(nss["shrunk"], joint["primary_shrunk"])


def test_r_primary_metric_mse_beats_primary_only():
    rng = np.random.default_rng(42)
    n_exp, n_sim = 100, 200
    tau = 0.015
    rhos = (0.8, 0.7, 0.6)
    se_p, se_g = 0.03, 0.01
    names = ["a", "b", "c"]
    mse_univ, mse_excl = [], []
    for _ in range(n_sim):
        k = len(rhos)
        delta = tau * rng.standard_normal(n_exp)
        g_true = np.empty((n_exp, k))
        for j, rho in enumerate(rhos):
            g_true[:, j] = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n_exp)
        x = delta + rng.normal(0, se_p, n_exp)
        g = g_true + rng.normal(0, se_g, size=(n_exp, k))
        half = n_exp // 2
        primary_metric = ["other"] * half + ["a"] * (n_exp - half)
        g[half:, 0] = np.nan
        se_g_a = np.full_like(g, se_g)
        se_g_a[half:, 0] = np.nan
        se_p_a = np.full(n_exp, se_p)
        univ = empirical_bayes_shrinkage(x, se_p_a, prior_mean=0.0, tau2=tau**2)
        excl = joint_metric_shrinkage_mvn(
            x,
            se_p_a,
            g,
            se_g_a,
            rho_primary=list(rhos),
            prior_sd_primary=tau,
            prior_sd_guard=tau,
            guardrail_names=names,
            primary_metric=primary_metric,
        )
        mse_univ.append(float(np.mean((univ["shrunk"] - delta) ** 2)))
        mse_excl.append(float(np.mean((excl["primary_shrunk"] - delta) ** 2)))
    assert float(np.mean(mse_excl)) < float(np.mean(mse_univ))


def test_mixed_primary_metrics_variable_companion_count():
    """Different primaries → different K used (full K vs K−1 per dropped NSS).

    Panel mixes non-NSS primary (all 5 companions) with each NSS-as-primary
    (other 4). Joint output must match per-stratum fixed-K MVN oracles and
    remain finite; strata sizes reflect the mix.
    """
    rng = np.random.default_rng(99)
    names = [
        "cashflow_gm_cuped",
        "payment_amount_cuped",
        "payment_amount_new_tutorings",
        "hours_confirmed_cuped",
        "new_tutoring_subscribers",
    ]
    k = len(names)
    # 12 non-NSS + 8 per each of 5 NSS primaries = 52 rows
    n_non = 12
    n_per_nss = 8
    blocks = [("non_nss", n_non)] + [(m, n_per_nss) for m in names]
    primary_metric = []
    for label, n_b in blocks:
        primary_metric.extend([label] * n_b)
    primary_metric = np.asarray(primary_metric, dtype=object)
    n = len(primary_metric)

    tau = 0.02
    rhos = np.array([0.55, 0.50, 0.45, 0.40, 0.35])
    se_p, se_g = 0.025, 0.012
    delta = tau * rng.standard_normal(n)
    g_true = np.empty((n, k))
    for j, rho in enumerate(rhos):
        g_true[:, j] = rho * delta + tau * np.sqrt(1 - rho**2) * rng.standard_normal(n)
    x = delta + rng.normal(0, se_p, n)
    g = g_true + rng.normal(0, se_g, size=(n, k))
    se_p_a = np.full(n, se_p)
    se_g_a = np.full((n, k), se_g)

    # Self-slots NaN when primary is that NSS metric
    for j, m in enumerate(names):
        mask = primary_metric == m
        g[mask, j] = np.nan
        se_g_a[mask, j] = np.nan

    out = joint_metric_shrinkage_mvn(
        x,
        se_p_a,
        g,
        se_g_a,
        rho_primary=rhos,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
        guardrail_names=names,
        primary_metric=primary_metric,
    )

    assert out["primary_metric_excluded"] is True
    assert out["strata_n"]["all_k"] == n_non
    for m in names:
        assert out["strata_n"][f"drop:{m}"] == n_per_nss
    assert np.isfinite(out["primary_shrunk"]).all()
    assert np.isfinite(out["primary_posterior_sd"]).all()

    # Oracle: each stratum matches a fixed-K call on that companion subset
    for label, _n_b in blocks:
        idxs = np.flatnonzero(primary_metric == label)
        if label == "non_nss":
            cols = list(range(k))
        else:
            j = names.index(label)
            cols = [c for c in range(k) if c != j]
        oracle = joint_metric_shrinkage_mvn(
            x[idxs],
            se_p_a[idxs],
            g[idxs][:, cols],
            se_g_a[idxs][:, cols],
            rho_primary=rhos[cols],
            prior_sd_primary=tau,
            prior_sd_guard=tau,
            guardrail_names=[names[c] for c in cols],
        )
        assert np.allclose(out["primary_shrunk"][idxs], oracle["primary_shrunk"], atol=1e-10)
        assert np.allclose(
            out["primary_posterior_sd"][idxs],
            oracle["primary_posterior_sd"],
            atol=1e-10,
        )
        # Companion count in the oracle path
        assert oracle["n_guardrails"] == (k if label == "non_nss" else k - 1)

    # K−1 rows should have weakly larger posterior SD than the same row
    # shrunk with all K (fill self with a distinct noisy companion) — drop
    # information ⇒ less precise primary. Use a non-self fill for the counterfactual.
    g_full = g.copy()
    se_full = se_g_a.copy()
    for j, m in enumerate(names):
        mask = primary_metric == m
        # fill with independent noise (not the primary) so K companions are usable
        g_full[mask, j] = rng.normal(0.0, 0.02, int(mask.sum()))
        se_full[mask, j] = se_g
    full_k = joint_metric_shrinkage_mvn(
        x,
        se_p_a,
        g_full,
        se_full,
        rho_primary=rhos,
        prior_sd_primary=tau,
        prior_sd_guard=tau,
    )
    nss_rows = primary_metric != "non_nss"
    # Mean posterior SD among NSS-primary rows: excluding self ≥ using a fake extra companion
    assert (
        float(np.mean(out["primary_posterior_sd"][nss_rows]))
        >= float(np.mean(full_k["primary_posterior_sd"][nss_rows])) - 1e-12
    )


def test_mixed_primary_metrics_cumulative_and_mom_pair_counts():
    """MoM companion-only pair counts reflect which primaries were excluded."""
    rng = np.random.default_rng(5)
    names = ["a", "b", "c"]
    n = 60
    # 20 each: primary=a, primary=b, primary=other
    primary_metric = np.array(["a"] * 20 + ["b"] * 20 + ["other"] * 20, dtype=object)
    y = rng.normal(0.01, 0.02, n)
    se_y = np.full(n, 0.01)
    g = rng.normal(0.0, 0.02, (n, 3))
    se_g = np.full((n, 3), 0.01)
    g[:20, 0] = np.nan
    se_g[:20, 0] = np.nan
    g[20:40, 1] = np.nan
    se_g[20:40, 1] = np.nan

    out = nss_adjusted_cumulative_impact_mvn(
        y,
        se_y,
        g,
        se_g,
        shipped=np.ones(n, dtype=bool),
        guardrail_names=names,
        primary_metric=primary_metric,
        prior_sd_primary=0.02,
        prior_sd_guard=0.02,
        # force MoM for rho only
        rho_primary=None,
    )
    # Column a: excluded on 20 rows → MoM uses 40; same for b; c uses all 60
    assert out["rho_info"]["n_pair"] == [40, 40, 60]
    assert out["rho_info"]["source"] == "mom_companion_only"
    assert out["strata_n"]["drop:a"] == 20
    assert out["strata_n"]["drop:b"] == 20
    assert out["strata_n"]["all_k"] == 20
    assert np.isfinite(out["cumulative"])
