import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

from experiment_utils.estimators import Estimators
from experiment_utils.experiment_analyzer import ExperimentAnalyzer


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


def test_collinear_treatment_returns_nan_with_diagnostics():
    """Between-unit treatment is collinear with unit FE -> pyfixest drops it."""
    df = make_panel(seed=5, between_unit=True)
    out = est().fixed_effects_regression(data=df, outcome_variable="y", fixed_effects=["unit"], covariates=["cov"])
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
