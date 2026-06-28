import numpy as np
import pandas as pd
import pyfixest as pf
import pytest

from experiment_utils.estimators import Estimators


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
