"""
Tests for calculate_retrodesign across all supported model types.

Design: each test generates data with a known true effect large enough to be
statistically significant (so the significance filter passes), then calls
calculate_retrodesign with a *smaller* true_effect that gives ~70-80% power.
The retrodesign power estimate is compared against the analytical normal-
approximation power derived from the observed standard error.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from experiment_utils import ExperimentAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def analytical_power(true_effect: float, se: float, alpha: float = 0.05) -> float:
    """Two-sided power via normal approximation: Φ(|δ/SE| − z_{α/2})."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z = abs(true_effect) / se
    return float(stats.norm.sf(z_alpha - z) + stats.norm.cdf(-z_alpha - z))


def run_retrodesign(analyzer, true_effect, nsim=1000, seed=0):
    analyzer.adjust_pvalues(method="bonferroni")
    return analyzer.calculate_retrodesign(true_effect=true_effect, nsim=nsim, seed=seed)


POWER_TOL = 0.15  # acceptable absolute difference vs. analytical power
TYPE_M_MIN = 0.99  # type_m ≥ 1 in theory; allow tiny MC rounding below
NSIM = 1000


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def binary_df():
    """Binary 0/1 outcome, n=10 000 per arm. True effect = 0.025 pp."""
    np.random.seed(42)
    n = 10_000
    p_c, te = 0.25, 0.025
    treatment = np.concatenate([np.ones(n), np.zeros(n)])
    outcome = np.where(
        treatment == 1,
        np.random.binomial(1, p_c + te, n * 2),
        np.random.binomial(1, p_c, n * 2),
    )
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


@pytest.fixture(scope="module")
def count_df():
    """Poisson count outcome, n=10 000 per arm. True effect = +0.2 counts."""
    np.random.seed(42)
    n = 10_000
    lam_c, te = 2.0, 0.2
    treatment = np.concatenate([np.ones(n), np.zeros(n)])
    outcome = np.where(
        treatment == 1,
        np.random.poisson(lam_c + te, n * 2),
        np.random.poisson(lam_c, n * 2),
    )
    return pd.DataFrame({"treatment": treatment, "outcome": outcome})


@pytest.fixture(scope="module")
def survival_df():
    """Survival data, n=10 000 per arm. True log-HR = 0.25."""
    np.random.seed(42)
    n = 10_000
    log_hr = 0.25
    scale_c = 100.0
    treatment = np.concatenate([np.ones(n), np.zeros(n)])
    time_raw = np.where(
        treatment == 1,
        np.random.exponential(scale_c * np.exp(log_hr), n * 2),
        np.random.exponential(scale_c, n * 2),
    )
    max_t = np.percentile(time_raw, 70)
    event = (time_raw <= max_t).astype(int)
    time = np.minimum(time_raw, max_t)
    return pd.DataFrame({"treatment": treatment, "time": time, "event": event})


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------


def test_retrodesign_ols_power(binary_df):
    """OLS on binary outcome: retrodesign power ≈ analytical power."""
    analyzer = ExperimentAnalyzer(binary_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="ols")
    analyzer.get_effects()

    se = analyzer._results["standard_error"].iloc[0]
    true_effect = 0.015  # smaller than data-generating effect → ~70% power
    expected_power = analytical_power(true_effect, se)

    retro = run_retrodesign(analyzer, true_effect=true_effect, nsim=NSIM)

    assert len(retro) > 0, "No retrodesign results (significance filter removed all rows)"
    row = retro.iloc[0]

    assert row["retrodesign_method"] == "powersim"
    assert 0.0 < row["power"] <= 1.0
    assert abs(row["power"] - expected_power) < POWER_TOL, (
        f"OLS power {row['power']:.3f} too far from analytical {expected_power:.3f}"
    )


def test_retrodesign_ols_type_m(binary_df):
    """OLS retrodesign: type_m ≥ 1 and type_s close to 0 for well-powered study."""
    analyzer = ExperimentAnalyzer(binary_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="ols")
    analyzer.get_effects()

    retro = run_retrodesign(analyzer, true_effect=0.015, nsim=NSIM)
    assert len(retro) > 0
    row = retro.iloc[0]

    assert row["type_m_error"] >= TYPE_M_MIN, f"type_m {row['type_m_error']:.3f} < {TYPE_M_MIN}"
    assert row["type_s_error"] < 0.1, f"type_s {row['type_s_error']:.3f} too high"


# ---------------------------------------------------------------------------
# Logistic
# ---------------------------------------------------------------------------


def test_retrodesign_logistic_power(binary_df):
    """Logistic (probability_change): retrodesign power ≈ analytical power."""
    analyzer = ExperimentAnalyzer(binary_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="logistic")
    analyzer.get_effects()

    se = analyzer._results["standard_error"].iloc[0]
    true_effect = 0.015
    expected_power = analytical_power(true_effect, se)

    retro = run_retrodesign(analyzer, true_effect=true_effect, nsim=NSIM)

    assert len(retro) > 0
    row = retro.iloc[0]

    assert row["retrodesign_method"] == "powersim"
    assert 0.0 < row["power"] <= 1.0
    assert abs(row["power"] - expected_power) < POWER_TOL, (
        f"Logistic power {row['power']:.3f} too far from analytical {expected_power:.3f}"
    )


def test_retrodesign_logistic_type_m(binary_df):
    """Logistic retrodesign: type_m ≥ 1."""
    analyzer = ExperimentAnalyzer(binary_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="logistic")
    analyzer.get_effects()

    retro = run_retrodesign(analyzer, true_effect=0.015, nsim=NSIM)
    assert len(retro) > 0
    assert retro.iloc[0]["type_m_error"] >= TYPE_M_MIN


# ---------------------------------------------------------------------------
# Poisson
# ---------------------------------------------------------------------------


def test_retrodesign_poisson_power(count_df):
    """Poisson (count_change): retrodesign power ≈ analytical power."""
    analyzer = ExperimentAnalyzer(count_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="poisson")
    analyzer.get_effects()

    se = analyzer._results["standard_error"].iloc[0]
    true_effect = 0.1  # smaller than data-generating 0.2 → moderate power
    expected_power = analytical_power(true_effect, se)

    retro = run_retrodesign(analyzer, true_effect=true_effect, nsim=NSIM)

    assert len(retro) > 0
    row = retro.iloc[0]

    assert row["retrodesign_method"] == "powersim"
    assert 0.0 < row["power"] <= 1.0
    assert abs(row["power"] - expected_power) < POWER_TOL, (
        f"Poisson power {row['power']:.3f} too far from analytical {expected_power:.3f}"
    )


def test_retrodesign_poisson_type_m(count_df):
    """Poisson retrodesign: type_m ≥ 1."""
    analyzer = ExperimentAnalyzer(count_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="poisson")
    analyzer.get_effects()

    retro = run_retrodesign(analyzer, true_effect=0.1, nsim=NSIM)
    assert len(retro) > 0
    assert retro.iloc[0]["type_m_error"] >= TYPE_M_MIN


# ---------------------------------------------------------------------------
# Negative Binomial
# ---------------------------------------------------------------------------


def test_retrodesign_negative_binomial_power(count_df):
    """Negative binomial retrodesign power ≈ analytical power."""
    analyzer = ExperimentAnalyzer(
        count_df,
        treatment_col="treatment",
        outcomes=["outcome"],
        outcome_models="negative_binomial",
    )
    analyzer.get_effects()

    se = analyzer._results["standard_error"].iloc[0]
    true_effect = 0.1
    expected_power = analytical_power(true_effect, se)

    retro = run_retrodesign(analyzer, true_effect=true_effect, nsim=NSIM)

    assert len(retro) > 0
    row = retro.iloc[0]

    assert row["retrodesign_method"] == "powersim"
    assert 0.0 < row["power"] <= 1.0
    assert abs(row["power"] - expected_power) < POWER_TOL, (
        f"NegBin power {row['power']:.3f} too far from analytical {expected_power:.3f}"
    )


# ---------------------------------------------------------------------------
# Cox
# ---------------------------------------------------------------------------


def test_retrodesign_cox_power(survival_df):
    """Cox model: retrodesign power ≈ analytical power on log-HR scale."""
    analyzer = ExperimentAnalyzer(
        survival_df,
        treatment_col="treatment",
        outcomes=[("time", "event")],
        outcome_models="cox",
    )
    analyzer.get_effects()

    r = analyzer._results.iloc[0]
    assert r["model_type"] == "cox"

    se = r["standard_error"]
    true_effect = 0.15  # log-HR — smaller than data-generating 0.25
    expected_power = analytical_power(true_effect, se)

    retro = run_retrodesign(analyzer, true_effect=true_effect, nsim=NSIM)

    assert len(retro) > 0, "Cox retrodesign returned no rows — check significance filter or effect direction"
    row = retro.iloc[0]

    assert row["retrodesign_method"] == "powersim"
    assert 0.0 < row["power"] <= 1.0
    assert abs(row["power"] - expected_power) < POWER_TOL, (
        f"Cox power {row['power']:.3f} too far from analytical {expected_power:.3f}"
    )


def test_retrodesign_cox_type_m(survival_df):
    """Cox retrodesign: type_m ≥ 1."""
    analyzer = ExperimentAnalyzer(
        survival_df,
        treatment_col="treatment",
        outcomes=[("time", "event")],
        outcome_models="cox",
    )
    analyzer.get_effects()

    retro = run_retrodesign(analyzer, true_effect=0.15, nsim=NSIM)
    assert len(retro) > 0
    assert retro.iloc[0]["type_m_error"] >= TYPE_M_MIN


# ---------------------------------------------------------------------------
# Cross-model consistency
# ---------------------------------------------------------------------------


def test_retrodesign_powersim_always_used(binary_df):
    """retrodesign_method is always 'powersim' regardless of model type."""
    for model in ("ols", "logistic"):
        analyzer = ExperimentAnalyzer(
            binary_df,
            treatment_col="treatment",
            outcomes=["outcome"],
            outcome_models=model,
        )
        analyzer.get_effects()
        retro = run_retrodesign(analyzer, true_effect=0.015, nsim=500)
        if len(retro) > 0:
            assert (retro["retrodesign_method"] == "powersim").all(), (
                f"{model}: expected powersim, got {retro['retrodesign_method'].unique()}"
            )


def test_retrodesign_no_extra_params(binary_df):
    """calculate_retrodesign works without extra kwargs; method param was removed."""
    analyzer = ExperimentAnalyzer(binary_df, treatment_col="treatment", outcomes=["outcome"], outcome_models="ols")
    analyzer.get_effects()
    analyzer.adjust_pvalues(method="bonferroni")

    retro = analyzer.calculate_retrodesign(true_effect=0.015, nsim=500, seed=0)

    if len(retro) > 0:
        assert (retro["retrodesign_method"] == "powersim").all()


def test_retrodesign_true_effect_dict_skips_missing_outcomes(count_df, binary_df):
    """Outcomes absent from the true_effect dict are excluded from results."""
    combined = pd.DataFrame(
        {
            "treatment": binary_df["treatment"],
            "binary": binary_df["outcome"],
            "count": count_df["outcome"].values,
        }
    )
    analyzer = ExperimentAnalyzer(
        combined,
        treatment_col="treatment",
        outcomes=["binary", "count"],
        outcome_models={"binary": "logistic", "count": "poisson"},
    )
    analyzer.get_effects()
    analyzer.adjust_pvalues(method="bonferroni")

    retro = analyzer.calculate_retrodesign(
        true_effect={"binary": 0.015},  # "count" intentionally omitted
        nsim=500,
        seed=0,
    )

    assert "count" not in retro["outcome"].values, "count should be excluded because it is not in the true_effect dict"
