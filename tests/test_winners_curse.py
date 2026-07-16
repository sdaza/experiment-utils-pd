import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from experiment_utils import ExperimentAnalyzer
from experiment_utils.utils import (
    empirical_bayes_shrinkage,
    fit_t_prior,
    fit_t_prior_with_estimated_mean,
    t_prior_shrinkage,
    winners_curse_estimate,
)


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


def test_empirical_bayes_fixed_tau2_single_experiment():
    # a single-row group is shrunk when an external prior variance is supplied
    df = _results_frame()  # conv group has only 1 log_odds row
    ea = _fitted_analyzer(df)
    tau2 = 0.04
    out = ea.winners_curse_summary(method="empirical_bayes", tau2=tau2)
    conv = out[out["outcome"] == "conv"].iloc[0]
    f = tau2 / (tau2 + 0.18**2)
    assert conv["tau2"] == tau2
    assert conv["corrected_effect"] == pytest.approx(f * 0.4)
    # relative effect for log_odds derives from exp(corrected)
    assert conv["corrected_relative_effect"] == pytest.approx(np.exp(f * 0.4) - 1)


def test_prior_dict_tau2_equals_tau2_param():
    ea = _fitted_analyzer(_results_frame())
    out_a = ea.winners_curse_summary(method="empirical_bayes", tau2=0.04)
    out_b = ea.winners_curse_summary(method="empirical_bayes", prior={"tau2": 0.04})
    assert np.allclose(out_a["corrected_effect"], out_b["corrected_effect"], equal_nan=True)


def test_prior_dict_t_single_experiment():
    # single-row group shrunk with an external t prior; relative effect derived
    df = _results_frame()  # conv group has only 1 log_odds row
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="empirical_bayes", prior={"scale": 0.15, "df": 4.0})
    conv = out[out["outcome"] == "conv"].iloc[0]
    expected = t_prior_shrinkage([0.4], [0.18], scale=0.15, df=4.0, ci=0.95)
    assert conv["corrected_effect"] == pytest.approx(expected["shrunk"][0])
    assert 0 < conv["corrected_effect"] < 0.4  # shrunk toward 0
    assert conv["corrected_relative_effect"] == pytest.approx(np.exp(conv["corrected_effect"]) - 1)
    # fit_t_prior's dict (with extra keys) is accepted directly
    out2 = ea.winners_curse_summary(
        method="empirical_bayes", prior={"scale": 0.15, "df": 4.0, "tau2": 9.9, "loglik": -1.0, "n": 50}
    )
    assert np.allclose(out2["corrected_effect"], out["corrected_effect"], equal_nan=True)


def test_prior_dict_validation():
    ea = _fitted_analyzer(_results_frame())
    with pytest.raises(ValueError, match="not both"):
        ea.winners_curse_summary(method="empirical_bayes", tau2=0.1, prior={"tau2": 0.1})
    with pytest.raises(ValueError, match="empirical_bayes"):
        ea.winners_curse_summary(method="conditional", prior={"tau2": 0.1})
    with pytest.raises(ValueError, match="prior"):
        ea.winners_curse_summary(method="empirical_bayes", prior={"sigma": 0.1})
    with pytest.raises(ValueError, match="scale"):
        ea.winners_curse_summary(method="empirical_bayes", prior={"scale": -1.0, "df": 4.0})


def test_fixed_tau2_requires_empirical_bayes():
    ea = _fitted_analyzer(_results_frame())
    with pytest.raises(ValueError, match="empirical_bayes"):
        ea.winners_curse_summary(method="conditional", tau2=0.1)


def test_fixed_tau2_validated_by_analyzer():
    ea = _fitted_analyzer(_results_frame())
    with pytest.raises(ValueError, match="tau2"):
        ea.winners_curse_summary(method="empirical_bayes", tau2=-1.0)


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


def test_eb_fixed_tau2_single_estimate():
    # posterior mean = tau2/(tau2+se^2) * y, posterior var = tau2*se^2/(tau2+se^2)
    tau2, y, se = 0.5, 1.0, 0.5
    out = empirical_bayes_shrinkage([y], [se], tau2=tau2)
    f = tau2 / (tau2 + se**2)
    assert out["tau2"] == tau2
    assert np.isclose(out["shrunk"][0], f * y)
    assert np.isclose(out["posterior_sd"][0], np.sqrt(tau2 * se**2 / (tau2 + se**2)))


def test_eb_fixed_tau2_skips_estimation():
    # identical inputs, wildly different fixed tau2 -> tau2 is used as-is
    out = empirical_bayes_shrinkage([1.0, -0.5, 0.8], [0.4, 0.4, 0.4], tau2=9.0)
    assert out["tau2"] == 9.0
    assert np.all(out["shrinkage_factor"] > 0.9)


def test_eb_fixed_tau2_zero_shrinks_to_prior_mean():
    out = empirical_bayes_shrinkage([1.0], [0.5], prior_mean=0.2, tau2=0.0)
    assert np.isclose(out["shrunk"][0], 0.2)
    assert np.isclose(out["shrinkage_factor"][0], 0.0)


def test_eb_fixed_tau2_validation():
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([1.0], [0.5], tau2=-0.1)
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([1.0], [0.5], tau2=np.nan)
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage([], [], tau2=0.5)


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


def test_t_prior_large_df_matches_normal():
    # as df -> inf the t prior tends to N(0, scale^2): posterior means converge
    scale, df = 0.5, 500.0
    y, se = [1.0, -0.4, 2.5], [0.6, 0.6, 0.6]
    out_t = t_prior_shrinkage(y, se, scale=scale, df=df)
    tau2 = scale**2 * df / (df - 2)
    out_n = empirical_bayes_shrinkage(y, se, tau2=tau2)
    assert np.allclose(out_t["shrunk"], out_n["shrunk"], atol=0.01)
    assert np.allclose(out_t["posterior_sd"], out_n["posterior_sd"], atol=0.01)


def test_t_prior_fat_tails_release_big_winners():
    # same prior variance: t(df=3) shrinks a big estimate LESS than the normal,
    # and its shrinkage is nonlinear (big z keeps a larger fraction than small z)
    df = 3.0
    tau2 = 0.25
    scale = np.sqrt(tau2 * (df - 2) / df)
    se = 0.5
    out_t = t_prior_shrinkage([0.8, 4.0], [se, se], scale=scale, df=df)
    out_n = empirical_bayes_shrinkage([0.8, 4.0], [se, se], tau2=tau2)
    assert out_t["shrunk"][1] > out_n["shrunk"][1]  # big winner passes through more
    assert out_t["shrinkage_factor"][1] > out_t["shrinkage_factor"][0]  # nonlinear


def test_t_prior_shrinks_toward_prior_mean():
    out = t_prior_shrinkage([1.0], [0.5], scale=0.3, df=4.0, prior_mean=0.2)
    assert 0.2 < out["shrunk"][0] < 1.0
    assert out["ci_lower"][0] < out["shrunk"][0] < out["ci_upper"][0]


def test_t_prior_single_estimate_and_keys():
    out = t_prior_shrinkage([1.0], [0.5], scale=0.3, df=4.0)
    assert set(out) == {
        "shrunk",
        "shrinkage_factor",
        "posterior_sd",
        "ci_lower",
        "ci_upper",
        "scale",
        "df",
        "tau2",
        "prior_mean",
    }
    assert out["shrunk"].shape == (1,)
    assert np.isclose(out["tau2"], 0.3**2 * 4.0 / 2.0)


def test_t_prior_validation():
    with pytest.raises(ValueError):
        t_prior_shrinkage([1.0], [0.5], scale=0.0, df=4.0)
    with pytest.raises(ValueError):
        t_prior_shrinkage([1.0], [0.5], scale=0.3, df=0.0)
    with pytest.raises(ValueError):
        t_prior_shrinkage([], [], scale=0.3, df=4.0)
    with pytest.raises(ValueError):
        t_prior_shrinkage([1.0], [-0.5], scale=0.3, df=4.0)


def test_fit_t_prior_recovers_scale():
    rng = np.random.default_rng(3)
    k, df_true, scale_true = 400, 4.0, 0.5
    beta = scale_true * rng.standard_t(df_true, size=k)
    se = np.full(k, 0.3)
    y = rng.normal(beta, se)
    fit = fit_t_prior(y, se, df=df_true)  # fix df, learn scale
    assert abs(fit["scale"] - scale_true) < 0.1
    assert fit["df"] == df_true
    assert fit["n"] == k
    assert np.isfinite(fit["loglik"])


def test_fit_t_prior_fits_df_above_two():
    rng = np.random.default_rng(4)
    beta = 0.5 * rng.standard_t(4.0, size=300)
    se = np.full(300, 0.3)
    y = rng.normal(beta, se)
    fit = fit_t_prior(y, se)
    assert fit["df"] > 2.0
    assert fit["scale"] > 0.0
    assert np.isfinite(fit["tau2"])


def test_fit_t_prior_validation():
    with pytest.raises(ValueError):
        fit_t_prior([1.0, 2.0], [0.5, 0.5])  # <3 estimates
    with pytest.raises(ValueError):
        fit_t_prior([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], df=-1.0)


def test_top_level_exports():
    import experiment_utils as eu

    assert hasattr(eu, "winners_curse_estimate")
    assert hasattr(eu, "empirical_bayes_shrinkage")
    assert hasattr(eu, "fit_t_prior")
    assert hasattr(eu, "fit_t_prior_with_estimated_mean")
    assert hasattr(eu, "t_prior_shrinkage")


def test_fit_t_prior_with_estimated_mean_validation():
    with pytest.raises(ValueError, match="at least 3"):
        fit_t_prior_with_estimated_mean([1.0, 2.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="df"):
        fit_t_prior_with_estimated_mean([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], df=1.0)
    with pytest.raises(ValueError, match="mean_ci"):
        fit_t_prior_with_estimated_mean([1.0, 2.0, 3.0], [0.5, 0.5, 0.5], mean_ci=1.0)
    with pytest.raises(ValueError, match="standard_errors"):
        fit_t_prior_with_estimated_mean([1.0, 2.0, 3.0], [0.5, 0.5, -0.1])


def test_fit_t_prior_with_estimated_mean_matches_conditional_fit():
    rng = np.random.default_rng(11)
    k, df, scale_true, mu_true = 120, 4.0, 0.4, 0.2
    beta = mu_true + scale_true * rng.standard_t(df, size=k)
    se = np.full(k, 0.2)
    y = rng.normal(beta, se)
    fit = fit_t_prior_with_estimated_mean(y, se, df=df)
    conditional = fit_t_prior(y, se, prior_mean=fit["prior_mean"], df=df)
    assert fit["scale"] == pytest.approx(conditional["scale"], rel=1e-6)
    assert fit["loglik"] == pytest.approx(conditional["loglik"], rel=1e-6)
    assert fit["prior_mean_method"] == "profile_likelihood"
    assert fit["prior_mean_ci_level"] == 0.95
    assert fit["prior_mean_ci_lower"] < fit["prior_mean"] < fit["prior_mean_ci_upper"]


def test_fit_t_prior_with_estimated_mean_recovers_mean_and_scale():
    """Seeded recovery: profile MLE recovers known prior mean and scale (tol absorbs MC noise).

    Tolerance 0.08 at k=400 is ~3× the Monte Carlo MAE (~0.025) under this DGP;
    the estimator is approximately unbiased (mean error ≈ 0 across seeds).
    """
    rng = np.random.default_rng(7)
    k, df, scale_true, mu_true = 400, 4.0, 0.5, 0.3
    beta = mu_true + scale_true * rng.standard_t(df, size=k)
    se = np.full(k, 0.25)
    y = rng.normal(beta, se)
    fit = fit_t_prior_with_estimated_mean(y, se, df=df)
    assert abs(fit["prior_mean"] - mu_true) < 0.08
    assert abs(fit["scale"] - scale_true) < 0.12


def test_fit_t_prior_with_estimated_mean_ci_excludes_zero_when_mean_large():
    rng = np.random.default_rng(8)
    k, df, scale_true, mu_true = 250, 4.0, 0.3, 0.4
    beta = mu_true + scale_true * rng.standard_t(df, size=k)
    se = np.full(k, 0.15)
    y = rng.normal(beta, se)
    fit = fit_t_prior_with_estimated_mean(y, se, df=df)
    assert fit["prior_mean_ci_lower"] > 0.0


def test_fit_t_prior_with_estimated_mean_lr_ci_coverage():
    """Empirical coverage of the profile LR interval ≈ nominal 0.95 (loose band for nsim)."""
    rng = np.random.default_rng(9)
    nsim, k, df, scale_true, se_val = 40, 80, 4.0, 0.45, 0.22
    covers = []
    for mu_true in (0.0, 0.25):
        for _ in range(nsim):
            beta = mu_true + scale_true * rng.standard_t(df, size=k)
            se = np.full(k, se_val)
            y = rng.normal(beta, se)
            fit = fit_t_prior_with_estimated_mean(y, se, df=df, mean_ci=0.95)
            covers.append(fit["prior_mean_ci_lower"] <= mu_true <= fit["prior_mean_ci_upper"])
    coverage = float(np.mean(covers))
    assert 0.80 <= coverage <= 1.0, f"coverage={coverage}"


def test_prior_dict_honors_t_prior_mean():
    ea = _fitted_analyzer(_results_frame())
    prior = {"scale": 0.15, "df": 4.0, "prior_mean": 0.2}
    out = ea.winners_curse_summary(method="empirical_bayes", prior=prior)
    conv = out[out["outcome"] == "conv"].iloc[0]
    expected = t_prior_shrinkage([0.4], [0.18], scale=0.15, df=4.0, prior_mean=0.2, ci=0.95)
    assert conv["corrected_effect"] == pytest.approx(expected["shrunk"][0])
    # default remains shrink-toward-zero when prior_mean omitted
    out0 = ea.winners_curse_summary(method="empirical_bayes", prior={"scale": 0.15, "df": 4.0})
    expected0 = t_prior_shrinkage([0.4], [0.18], scale=0.15, df=4.0, prior_mean=0.0, ci=0.95)
    assert out0[out0["outcome"] == "conv"].iloc[0]["corrected_effect"] == pytest.approx(expected0["shrunk"][0])


def test_prior_dict_honors_normal_prior_mean():
    ea = _fitted_analyzer(_results_frame())
    prior = {"tau2": 0.04, "prior_mean": 0.5}
    out = ea.winners_curse_summary(method="empirical_bayes", prior=prior)
    rev = out[(out["outcome"] == "rev") & (out["absolute_effect"] == 5.0)].iloc[0]
    expected = empirical_bayes_shrinkage([5.0], [2.0], prior_mean=0.5, tau2=0.04, ci=0.95)
    assert rev["corrected_effect"] == pytest.approx(expected["shrunk"][0])


def test_conditional_uses_mcp_selection_alpha():
    """MCP-selected rows must be de-biased at their per-comparison alpha_mcp."""
    df = _results_frame()
    df["pvalue_mcp"] = [0.001, 0.5, 0.002]
    df["stat_significance_mcp"] = [1, 0, 1]
    df["alpha_mcp"] = 0.05 / 3
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="conditional")
    rev = out[out["outcome"] == "rev"].iloc[0]
    expected = winners_curse_estimate(5.0, 2.0, alpha=0.05 / 3, ci=0.95)["corrected"]
    assert rev["corrected_effect"] == pytest.approx(expected, rel=1e-9)
    # stricter selection threshold -> stronger correction than at nominal alpha
    nominal = winners_curse_estimate(5.0, 2.0, alpha=0.05, ci=0.95)["corrected"]
    assert abs(rev["corrected_effect"]) < abs(nominal)


def test_conditional_explicit_alpha_overrides_alpha_mcp():
    """A user-supplied alpha wins over the stored alpha_mcp."""
    df = _results_frame()
    df["pvalue_mcp"] = [0.001, 0.5, 0.002]
    df["stat_significance_mcp"] = [1, 0, 1]
    df["alpha_mcp"] = 0.05 / 3
    ea = _fitted_analyzer(df)
    out = ea.winners_curse_summary(method="conditional", alpha=0.05)
    rev = out[out["outcome"] == "rev"].iloc[0]
    expected = winners_curse_estimate(5.0, 2.0, alpha=0.05, ci=1 - 0.05)["corrected"]
    assert rev["corrected_effect"] == pytest.approx(expected, rel=1e-9)
