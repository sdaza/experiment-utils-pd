import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from experiment_utils import ExperimentAnalyzer
from experiment_utils.utils import empirical_bayes_shrinkage, winners_curse_estimate


def _fitted_analyzer(results: pd.DataFrame) -> ExperimentAnalyzer:
    # Build an analyzer and inject a synthetic results frame (bypassing get_effects).
    ea = ExperimentAnalyzer(
        data=pd.DataFrame({"treatment": [0, 1], "y": [0.0, 1.0]}),
        outcomes=["y"],
        treatment_col="treatment",
    )
    ea._results = results
    return ea


def _results_frame():
    return pd.DataFrame(
        {
            "outcome": ["rev", "rev", "conv"],
            "treatment_group": [1, 1, 1],
            "control_group": [0, 0, 0],
            "effect_type": ["mean_difference", "mean_difference", "log_odds"],
            "absolute_effect": [5.0, 0.2, 0.4],
            "standard_error": [2.0, 2.0, 0.18],
            "control_value": [50.0, 50.0, 0.1],
            "stat_significance": [1, 0, 1],
        }
    )


def test_summary_requires_get_effects():
    ea = ExperimentAnalyzer(
        data=pd.DataFrame({"treatment": [0, 1], "y": [0.0, 1.0]}),
        outcomes=["y"],
        treatment_col="treatment",
    )
    with pytest.raises(ValueError, match="get_effects"):
        ea.winners_curse_summary()


def test_summary_bad_method():
    ea = _fitted_analyzer(_results_frame())
    with pytest.raises(ValueError, match="method"):
        ea.winners_curse_summary(method="nope")


def test_conditional_zero_control_value_relative_is_nan():
    df = _results_frame()
    df.loc[0, "control_value"] = 0.0  # significant mean_difference row
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="conditional")
    rev = out[out["outcome"] == "rev"].iloc[0]
    assert not pd.isna(rev["corrected_effect"])  # point estimate still computed
    assert pd.isna(rev["corrected_relative_effect"])
    assert pd.isna(rev["corrected_rel_ci_lower"])
    assert pd.isna(rev["corrected_rel_ci_upper"])


def test_summary_bad_alpha_ci():
    ea = _fitted_analyzer(_results_frame())
    with pytest.raises(ValueError):
        ea.winners_curse_summary(method="conditional", alpha=1.5)
    with pytest.raises(ValueError):
        ea.winners_curse_summary(method="conditional", ci=0.0)


def test_conditional_filters_to_significant():
    ea = _fitted_analyzer(_results_frame())
    out = ea.winners_curse_summary(method="conditional")
    # only the 2 significant rows survive
    assert len(out) == 2
    assert {"corrected_effect", "corrected_ci_lower", "corrected_ci_upper", "shrinkage"} <= set(out.columns)
    # mean_difference row: |corrected| < |observed|
    rev = out[out["outcome"] == "rev"].iloc[0]
    assert abs(rev["corrected_effect"]) < 5.0


def test_conditional_log_odds_relative_is_exp():
    ea = _fitted_analyzer(_results_frame())
    out = ea.winners_curse_summary(method="conditional")
    conv = out[out["outcome"] == "conv"].iloc[0]
    assert np.isclose(conv["corrected_relative_effect"], np.exp(conv["corrected_effect"]) - 1)
    assert np.isclose(conv["corrected_rel_ci_lower"], np.exp(conv["corrected_ci_lower"]) - 1)


def test_conditional_additive_relative_divides_control():
    ea = _fitted_analyzer(_results_frame())
    out = ea.winners_curse_summary(method="conditional")
    rev = out[out["outcome"] == "rev"].iloc[0]
    assert np.isclose(rev["corrected_relative_effect"], rev["corrected_effect"] / 50.0)


def test_conditional_degenerate_se_is_nan():
    df = _results_frame()
    df.loc[0, "standard_error"] = 0.0
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="conditional")
    rev = out[out["outcome"] == "rev"].iloc[0]
    assert pd.isna(rev["corrected_effect"])


def test_empirical_bayes_groups_by_effect_type():
    # 4 mean_difference + 3 log_odds under one outcome -> two separate tau2.
    df = pd.DataFrame(
        {
            "outcome": ["m"] * 7,
            "effect_type": ["mean_difference"] * 4 + ["log_odds"] * 3,
            "absolute_effect": [5.0, 1.0, 8.0, 0.5, 0.4, -0.2, 0.9],
            "standard_error": [2.0, 1.0, 3.0, 0.8, 0.18, 0.2, 0.3],
            "control_value": [50.0] * 4 + [0.1] * 3,
            "stat_significance": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="empirical_bayes")
    assert out["tau2"].nunique() == 2  # one tau2 per effect_type group


def test_empirical_bayes_too_few_is_nan():
    df = _results_frame()  # conv group has only 1 log_odds row
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="empirical_bayes")
    conv = out[out["outcome"] == "conv"].iloc[0]
    assert pd.isna(conv["corrected_effect"])


def test_returns_documented_keys():
    out = winners_curse_estimate(effect=5.0, standard_error=2.0, alpha=0.05)
    assert set(out) == {"corrected", "ci_lower", "ci_upper", "observed_z", "shrinkage"}


def test_no_shrink_far_above_threshold():
    # z = 10 >> 1.96: correction should be negligible
    out = winners_curse_estimate(effect=10.0, standard_error=1.0, alpha=0.05)
    assert abs(out["corrected"] - 10.0) < 0.05
    assert out["shrinkage"] > 0.99


def test_strong_shrink_near_threshold():
    # just-significant: z just above 1.96 -> heavy shrink toward 0
    s = 1.0
    b = norm.ppf(0.975) * s + 1e-3
    out = winners_curse_estimate(effect=b, standard_error=s, alpha=0.05)
    assert abs(out["corrected"]) < abs(b)
    assert out["shrinkage"] < 0.6


def test_sign_symmetry():
    pos = winners_curse_estimate(effect=4.0, standard_error=1.5, alpha=0.05)
    neg = winners_curse_estimate(effect=-4.0, standard_error=1.5, alpha=0.05)
    assert np.isclose(pos["corrected"], -neg["corrected"], atol=1e-6)
    assert np.isclose(pos["ci_lower"], -neg["ci_upper"], atol=1e-6)


def test_monotone_in_effect():
    s = 1.0
    vals = [winners_curse_estimate(b, s, alpha=0.05)["corrected"] for b in [2.1, 3.0, 4.0, 6.0]]
    assert all(x < y for x, y in zip(vals, vals[1:], strict=False))


def test_ci_ordering():
    out = winners_curse_estimate(effect=4.0, standard_error=1.5, alpha=0.05, ci=0.95)
    assert out["ci_lower"] < out["corrected"] < out["ci_upper"]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_raises_on_bad_se(bad):
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=bad)


def test_raises_on_bad_alpha_ci():
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, alpha=1.5)
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, ci=0.0)
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, alpha=1.0)
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, ci=1.0)


def test_warns_below_threshold():
    with pytest.warns(RuntimeWarning):
        winners_curse_estimate(effect=1.0, standard_error=1.0, alpha=0.05)  # z=1.0 < 1.96


def test_below_threshold_corrects_toward_zero():
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        out = winners_curse_estimate(effect=1.0, standard_error=1.0, alpha=0.05)  # z=1.0 < 1.96 (gap)
    # gap input must yield a finite, valid correction shrunk toward null, with an ordered CI
    assert np.isfinite(out["corrected"])
    assert abs(out["corrected"]) < 1.0  # heavily shrunk relative to the observed 1.0
    assert abs(out["corrected"]) < 0.2  # gap maps essentially to ~0
    assert out["ci_lower"] < out["corrected"] < out["ci_upper"]


def _selected_draws(beta_true, s, alpha, n, seed):
    rng = np.random.default_rng(seed)
    c = norm.ppf(1 - alpha / 2) * s
    draws = rng.normal(beta_true, s, size=n)
    return draws[np.abs(draws) >= c]


def test_sim_median_unbiased():
    # MUE solves F_beta(b)=0.5; F_beta(B)~U(0,1) => median of corrected ~ beta_true.
    beta_true, s, alpha = 1.0, 1.0, 0.05
    sel = _selected_draws(beta_true, s, alpha, n=400_000, seed=7)[:4000]
    corrected = np.array([winners_curse_estimate(b, s, alpha)["corrected"] for b in sel])
    assert abs(np.median(corrected) - beta_true) < 0.12


def test_sim_conditional_ci_coverage():
    # P(gamma/2 <= F_beta_true(B) <= 1-gamma/2) = 0.95 exactly in theory.
    beta_true, s, alpha = 1.0, 1.0, 0.05
    sel = _selected_draws(beta_true, s, alpha, n=400_000, seed=11)[:3000]
    cov = np.mean(
        [
            (lambda o: o["ci_lower"] <= beta_true <= o["ci_upper"])(winners_curse_estimate(b, s, alpha, ci=0.95))
            for b in sel
        ]
    )
    assert 0.93 <= cov <= 0.97


def test_eb_returns_keys_and_shapes():
    out = empirical_bayes_shrinkage([1.0, -0.5, 0.8, 0.2], [0.4, 0.4, 0.4, 0.4])
    assert set(out) == {"shrunk", "shrinkage_factor", "posterior_sd", "ci_lower", "ci_upper", "tau2", "prior_mean"}
    assert out["shrunk"].shape == (4,)


def test_eb_shrinkage_factor_formula():
    se = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    out = empirical_bayes_shrinkage([1.0, -0.5, 0.8, 0.2, -1.1], se)
    expected = out["tau2"] / (out["tau2"] + se**2)
    assert np.allclose(out["shrinkage_factor"], expected)
    assert np.allclose(out["shrunk"], out["shrinkage_factor"] * np.array([1.0, -0.5, 0.8, 0.2, -1.1]))


def test_eb_nonzero_prior_mean_offset():
    se = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    effects = np.array([1.0, -0.5, 0.8, 0.2, -1.1])
    out = empirical_bayes_shrinkage(effects, se, prior_mean=0.5)
    expected = 0.5 + out["shrinkage_factor"] * (effects - 0.5)
    assert np.allclose(out["shrunk"], expected)
    assert out["prior_mean"] == 0.5


def test_eb_high_variance_shrinks_more():
    out = empirical_bayes_shrinkage([1.0, 1.0, 1.0, 1.0], [0.1, 0.5, 1.0, 2.0])
    assert out["shrinkage_factor"][0] > out["shrinkage_factor"][-1]


def test_eb_requires_three():
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([1.0, 2.0], [0.5, 0.5])


def test_eb_raises_on_bad_se():
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([1.0, 2.0, 3.0], [0.5, -0.5, 0.5])
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([1.0, 2.0, 3.0], [0.5, np.nan, 0.5])


def test_eb_sim_mse_reduction():
    rng = np.random.default_rng(0)
    k, tau2_true = 200, 0.5
    s = np.sqrt(rng.uniform(0.2, 2.0, size=k))
    beta = rng.normal(0.0, np.sqrt(tau2_true), size=k)
    y = rng.normal(beta, s)
    out = empirical_bayes_shrinkage(y, s, prior_mean=0.0)
    assert np.mean((out["shrunk"] - beta) ** 2) < np.mean((y - beta) ** 2)
    assert abs(out["tau2"] - tau2_true) < 0.2


def test_top_level_exports():
    import experiment_utils as eu

    assert hasattr(eu, "winners_curse_estimate")
    assert hasattr(eu, "empirical_bayes_shrinkage")
