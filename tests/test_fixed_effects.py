import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

from experiment_utils.estimators import Estimators
from experiment_utils.experiment_analyzer import ExperimentAnalyzer
from experiment_utils.utils import suppress_fit_warnings


def make_panel(n_units=60, n_periods=6, effect=1.5, seed=0, between_unit=False):
    """Tiny within-unit panel. between_unit=True makes treatment constant per unit."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        fe = rng.normal()
        unit_treat = int(u % 2 == 0)
        for t in range(n_periods):
            treat = unit_treat if between_unit else int((u + t) % 2 == 0)
            z = rng.normal()
            y = 2.0 + effect * treat + 0.4 * z + fe + rng.normal(0, 1)
            rows.append((u, t, treat, z, y))
    df = pd.DataFrame(rows, columns=["unit", "period", "treatment", "cov", "y"])
    # analyzer standardizes covariates to z_ before calling estimators
    df["z_cov"] = (df["cov"] - df["cov"].mean()) / df["cov"].std()
    return df


def est():
    return Estimators(treatment_col="treatment", alpha=0.05)


def test_ols_fe_recovers_within_unit_effect():
    df = make_panel(effect=1.5, seed=1)
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    assert out["model_type"] == "ols"
    assert out["effect_type"] == "mean_difference"
    assert out["absolute_effect"] == pytest.approx(1.5, abs=0.25)


def test_ols_fe_ci_matches_pyfixest_exactly():
    """df_resid=float(fit._df_t) => analyzer's Wald reconstruction == pyfixest CI."""
    df = make_panel(seed=2)
    e = est()
    out = e.fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    fit = pf.feols("y ~ z_cov + treatment | unit", data=df, vcov="hetero")
    ci = fit.confint(alpha=0.05).loc["treatment"]
    from scipy import stats

    t_crit = stats.t.ppf(0.975, out["df_resid"])
    lo = out["absolute_effect"] - t_crit * out["standard_error"]
    hi = out["absolute_effect"] + t_crit * out["standard_error"]
    assert lo == pytest.approx(float(ci.iloc[0]), abs=1e-6)
    assert hi == pytest.approx(float(ci.iloc[1]), abs=1e-6)


def test_switcher_diagnostics_exact():
    df = make_panel(n_units=60, seed=3)  # within-unit => all units switch
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    assert out["n_units"] == 60
    assert out["n_switchers"] == 60
    assert out["pct_switchers"] == 100.0
    assert out["fe_absorbed"] == "unit"


def test_relative_effect_is_control_mean_delta():
    df = make_panel(seed=4)
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    control_mean = df.loc[df["treatment"] == 0, "y"].mean()
    assert out["control_value"] == pytest.approx(control_mean)
    assert out["relative_effect"] == pytest.approx(out["absolute_effect"] / control_mean)
    assert np.isnan(out["se_intercept"])
    assert np.isnan(out["cov_coef_intercept"])


def test_collinear_treatment_returns_nan_with_diagnostics(capsys):
    """Between-unit treatment has 0% switchers -> switcher gate skips the fit."""
    df = make_panel(seed=5, between_unit=True)
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    assert "switchers" in capsys.readouterr().err
    assert np.isnan(out["absolute_effect"])
    assert out["n_switchers"] == 0
    assert out["pct_switchers"] == 0.0


def test_clustered_vcov_differs_from_hetero():
    df = make_panel(seed=6)
    e = est()
    hetero = e.fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    clustered = e.fixed_effects_regression(
        data=df,
        outcome_variable="y",
        fixed_effects=["unit"],
        covariates=["cov"],
        cluster_col="unit",
    )
    assert hetero["absolute_effect"] == pytest.approx(clustered["absolute_effect"])
    assert hetero["standard_error"] != pytest.approx(clustered["standard_error"])


def test_compute_relative_ci_false_nans_relative_bounds():
    """compute_relative_ci=False (bootstrap path) NaNs relative bounds but keeps point/SE."""
    df = make_panel(seed=7)
    out = est().fixed_effects_regression(
        data=df,
        outcome_variable="y",
        fixed_effects=["unit"],
        covariates=["cov"],
        compute_relative_ci=False,
    )
    assert np.isnan(out["rel_effect_lower"])
    assert np.isnan(out["rel_effect_upper"])
    assert np.isfinite(out["absolute_effect"])
    assert np.isfinite(out["standard_error"])


def make_count_panel(n_units=60, n_periods=6, log_irr=0.3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        fe = rng.normal(0, 0.3)
        for t in range(n_periods):
            treat = int((u + t) % 2 == 0)
            z = rng.normal()
            mu = np.exp(0.2 + log_irr * treat + 0.1 * z + fe)
            rows.append((u, t, treat, z, rng.poisson(mu)))
    df = pd.DataFrame(rows, columns=["unit", "period", "treatment", "cov", "count"])
    df["z_cov"] = (df["cov"] - df["cov"].mean()) / df["cov"].std()
    return df


def test_poisson_fe_recovers_irr():
    df = make_count_panel(log_irr=0.3, seed=7)
    out = est().fixed_effects_regression(
        data=df,
        outcome_variable="count",
        fixed_effects=["unit"],
        covariates=["cov"],
        model_type="poisson",
    )
    assert out["model_type"] == "poisson"
    assert out["effect_type"] == "log_rate_ratio"
    assert out["incidence_rate_ratio"] == pytest.approx(np.exp(0.3), abs=0.25)
    assert out["relative_effect"] == pytest.approx(out["incidence_rate_ratio"] - 1)
    assert out["n_switchers"] == 60


def analyzer_df(seed=10):
    df = make_panel(seed=seed)
    return df


def test_init_stores_fixed_effects_as_list():
    df = analyzer_df()
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        fixed_effects="unit",  # string normalized to list
    )
    assert a._fixed_effects == ["unit"]
    assert a._fixed_effects_min_switcher_pct == 10.0


def test_init_missing_fe_column_warns_and_drops(caplog):
    df = analyzer_df()
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        fixed_effects=["unit", "does_not_exist"],
    )
    assert a._fixed_effects == ["unit"]


def test_fe_column_retained_in_data_after_init():
    df = analyzer_df()
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        fixed_effects=["unit"],
    )
    assert "unit" in a._data.columns


def test_get_effects_uses_fe_and_adds_diagnostic_columns():
    df = analyzer_df(seed=11)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        regression_covariates=["cov"],
        fixed_effects=["unit"],
        cluster_col="unit",
    )
    a.get_effects()
    res = a._results
    assert {"n_units", "n_switchers", "pct_switchers", "fe_absorbed"}.issubset(res.columns)
    row = res.iloc[0]
    assert row["fe_absorbed"] == "unit"
    assert row["n_switchers"] > 0
    assert row["absolute_effect"] == pytest.approx(1.5, abs=0.3)


def test_get_effects_fe_warns_and_falls_back_for_logistic(caplog):
    df = analyzer_df(seed=12)
    df["bin"] = (df["y"] > df["y"].median()).astype(int)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["bin"],
        treatment_col="treatment",
        outcome_models={"bin": "logistic"},
        fixed_effects=["unit"],
    )
    a.get_effects()
    res = a._results
    # logistic ran (no FE), so FE diagnostics are absent or NaN for this row
    assert res.iloc[0]["model_type"] == "logistic"
    if "n_switchers" in res.columns:
        assert pd.isna(res.iloc[0]["n_switchers"])
    if "fe_absorbed" in res.columns:
        assert res.iloc[0]["fe_absorbed"] in ("", None) or pd.isna(res.iloc[0]["fe_absorbed"])


def test_non_fe_run_schema_has_no_fe_columns():
    """Without fixed_effects, FE diagnostic columns are absent (schema unchanged)."""
    df = analyzer_df(seed=13)
    a = ExperimentAnalyzer(data=df, outcomes=["y"], treatment_col="treatment")
    a.get_effects()
    res = a.results
    for col in ["n_units", "n_switchers", "pct_switchers", "fe_absorbed"]:
        assert col not in res.columns


def test_fe_diagnostic_columns_present_after_standard_columns():
    """With fixed_effects, FE diagnostic columns are present and appear after the effect columns."""
    df = analyzer_df(seed=14)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        regression_covariates=["cov"],
        fixed_effects=["unit"],
    )
    a.get_effects()
    res = a.results
    for col in ["n_units", "n_switchers", "pct_switchers", "fe_absorbed"]:
        assert col in res.columns
    cols = list(res.columns)
    assert cols.index("fe_absorbed") > cols.index("absolute_effect")
    assert res.iloc[0]["fe_absorbed"] == "unit"


def test_fe_composes_with_interactions_and_clustering():
    """FE + regression covariates + CUPED interactions + clustered SEs runs end-to-end."""
    df = analyzer_df(seed=15)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        regression_covariates=["cov"],
        interaction_covariates=["cov"],
        fixed_effects=["unit"],
        cluster_col="unit",
    )
    a.get_effects()
    res = a.results
    assert len(res) >= 1
    assert res.iloc[0]["fe_absorbed"] == "unit"
    assert np.isfinite(res.iloc[0]["absolute_effect"])
    assert res.iloc[0]["n_switchers"] > 0


def test_fe_estimator_dropna_alignment():
    """dropna happens first, so switcher counts match the rows pyfixest fits."""
    df = make_panel(seed=16)
    df.loc[df.index[:20], "y"] = np.nan  # introduce missingness in the outcome
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
    cleaned = df.dropna(subset=["y", "treatment", "z_cov", "unit"])
    assert out["n_units"] == cleaned["unit"].nunique()
    expected_switchers = int((cleaned.groupby("unit")["treatment"].nunique() > 1).sum())
    assert out["n_switchers"] == expected_switchers


def test_bootstrap_with_fe_keeps_analytic_se_and_warns(capsys):
    """bootstrap=True + fixed_effects: FE rows keep analytic SE/p-value (no NaN), with a warning."""
    df = analyzer_df(seed=21)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        regression_covariates=["cov"],
        fixed_effects=["unit"],
        bootstrap=True,
    )
    a.get_effects()
    captured = capsys.readouterr()
    res = a.results
    row = res.iloc[0]
    assert np.isfinite(row["standard_error"])
    assert np.isfinite(row["pvalue"])
    assert np.isfinite(row["absolute_effect"])
    assert "fixed-effects" in captured.err.lower()


def test_ols_fe_ci_matches_pyfixest_clustered():
    """The df_resid coupling reproduces pyfixest's CI under CRV1 clustering too."""
    df = make_panel(seed=22)
    out = est().fixed_effects_regression(
        data=df,
        outcome_variable="y",
        fixed_effects=["unit"],
        covariates=["cov"],
        cluster_col="unit",
    )
    fit = pf.feols("y ~ z_cov + treatment | unit", data=df, vcov={"CRV1": "unit"})
    ci = fit.confint(alpha=0.05).loc["treatment"]
    from scipy import stats

    t_crit = stats.t.ppf(0.975, out["df_resid"])
    lo = out["absolute_effect"] - t_crit * out["standard_error"]
    hi = out["absolute_effect"] + t_crit * out["standard_error"]
    assert lo == pytest.approx(float(ci.iloc[0]), abs=1e-6)
    assert hi == pytest.approx(float(ci.iloc[1]), abs=1e-6)


# --- numeric covariate finiteness/variance guard + matmul warning suppression ---


def test_numeric_covariate_status_classifies_ok_nonfinite_degenerate():
    status = ExperimentAnalyzer._numeric_covariate_status
    rng = np.random.default_rng(0)
    assert status(pd.Series(rng.normal(size=200))) == "ok"
    assert status(pd.Series([5.0] * 100)) == "degenerate"  # exactly constant
    # near-constant: std tiny relative to scale -> standardizing is meaningless
    # and risks overflow; must be rejected even though std != 0.
    near_constant = pd.Series([1e6 + (i % 2) * 1e-5 for i in range(100)])
    assert status(near_constant) == "degenerate"
    assert status(pd.Series([1.0, 2.0, np.inf, 4.0])) == "nonfinite"
    assert status(pd.Series([1.0, 2.0, np.nan, 4.0])) == "nonfinite"


def test_suppress_fit_warnings_silences_only_known_noise():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with suppress_fit_warnings():
            warnings.warn("divide by zero encountered in matmul", RuntimeWarning, stacklevel=2)
            warnings.warn("29855 singleton fixed effect(s) dropped from the model.", UserWarning, stacklevel=2)
        warnings.warn("a genuine unrelated warning", RuntimeWarning, stacklevel=2)  # must survive
    messages = [str(w.message) for w in rec]
    assert not any("matmul" in m for m in messages)
    assert not any("singleton" in m for m in messages)
    assert any("unrelated" in m for m in messages)


def test_fe_reports_effective_sample_size_and_drops_singletons():
    """Singleton FE units are dropped silently; n_obs/n_singletons_dropped report the gap."""
    df = make_panel(n_units=40, seed=24)  # 40 units x 6 periods, all switchers
    # Append 10 singleton units (one observation each) -> pyfixest drops them.
    extra = pd.DataFrame(
        {
            "unit": range(1000, 1010),
            "period": 0,
            "treatment": [i % 2 for i in range(10)],
            "cov": np.linspace(-1, 1, 10),
            "y": np.linspace(0, 1, 10),
        }
    )
    extra["z_cov"] = (extra["cov"] - extra["cov"].mean()) / extra["cov"].std()
    df = pd.concat([df, extra], ignore_index=True)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])

    assert not any("singleton" in str(w.message) for w in rec)  # warning suppressed
    assert out["n_obs"] == 40 * 6  # 10 singleton rows dropped from 250
    assert out["n_singletons_dropped"] == 10
    assert np.isfinite(out["absolute_effect"])


def test_near_constant_numeric_covariate_dropped_from_propensity_fit():
    """A near-constant numeric covariate is dropped (not fed to the ps-logistic fit)."""
    df = analyzer_df(seed=23)
    rng = np.random.default_rng(23)
    # >2 unique values => treated as numeric; std tiny relative to ~1e6 scale.
    df["dust"] = 1e6 + rng.normal(0, 1e-5, size=len(df))
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        balance_covariates=["cov", "dust"],
        adjustment="balance",
        balance_method="ps-logistic",
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a.get_effects()
    assert "dust" not in a._final_covariates
    assert "cov" in a._final_covariates
    assert not any("matmul" in str(w.message) for w in rec)


def test_fe_fit_does_not_leak_matmul_warnings():
    """Near-singular FE design (interactions + clustering) must not leak matmul warnings."""
    df = analyzer_df(seed=15)
    a = ExperimentAnalyzer(
        data=df,
        outcomes=["y"],
        treatment_col="treatment",
        regression_covariates=["cov"],
        interaction_covariates=["cov"],
        fixed_effects=["unit"],
        cluster_col="unit",
    )
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        a.get_effects()
    assert not any("matmul" in str(w.message) for w in rec)
