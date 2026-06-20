import numpy as np
import pandas as pd
import pytest

from experiment_utils import ExperimentAnalyzer
from experiment_utils import false_positive_risk as fpr_from_init
from experiment_utils.utils import estimate_true_success_rate, false_positive_risk


def test_fpr_known_values():
    # alpha=0.10, power=0.80, prior_success_rate=0.12
    # FPR = (0.10*0.88) / (0.10*0.88 + 0.80*0.12) = 0.088 / 0.184 = 0.4783
    result = false_positive_risk(alpha=0.10, power=0.80, prior_success_rate=0.12)
    assert abs(result - 0.478) < 0.01


def test_fpr_known_values_low_alpha():
    # alpha=0.05, power=0.80, prior_success_rate=0.15
    # FPR = (0.05*0.85) / (0.05*0.85 + 0.80*0.15) = 0.0425 / 0.1625 = 0.2615
    result = false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.15)
    assert abs(result - 0.262) < 0.01


def test_fpr_high_success_rate():
    # At 50% success rate, FPR should be lower
    result = false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.50)
    assert result < 0.10


def test_fpr_invalid_prior():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.0)
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=1.0)


def test_fpr_invalid_alpha():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.0, power=0.80, prior_success_rate=0.15)


def test_fpr_invalid_power():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.0, prior_success_rate=0.15)


def test_estimate_true_success_rate_known():
    # win_rate=0.12, alpha=0.10, power=0.80
    # pi = (0.80-0.12)/(0.80-0.10) = 0.68/0.70 = 0.9714 -> true = 1 - pi = 0.0286
    result = estimate_true_success_rate(win_rate=0.12, alpha=0.10, power=0.80)
    assert abs(result - 0.029) < 0.01


def test_estimate_true_success_rate_standard():
    # win_rate=0.15, alpha=0.05, power=0.80
    # pi = (0.80-0.15)/(0.80-0.05) = 0.65/0.75 = 0.8667 -> true = 1 - pi = 0.1333
    result = estimate_true_success_rate(win_rate=0.15, alpha=0.05, power=0.80)
    assert abs(result - 0.14) < 0.02


def test_estimate_true_success_rate_clamps_to_zero():
    # win_rate < alpha -> pi > 1 -> clamp to 0
    result = estimate_true_success_rate(win_rate=0.03, alpha=0.05, power=0.80)
    assert result == 0.0


def test_estimate_true_success_rate_power_equals_alpha():
    with pytest.raises(ValueError):
        estimate_true_success_rate(win_rate=0.15, alpha=0.80, power=0.80)


def test_exported_from_init():
    # Verify the function is importable from the top-level package
    result = fpr_from_init(alpha=0.05, power=0.80, prior_success_rate=0.15)
    assert 0 < result < 1


def _make_analyzer():
    np.random.seed(42)
    n = 2000
    treatment = np.random.binomial(1, 0.5, n)
    outcome = np.random.binomial(1, 0.10 + 0.03 * treatment, n)
    df = pd.DataFrame({"treatment": treatment, "outcome": outcome})
    ea = ExperimentAnalyzer(data=df, outcomes=["outcome"], treatment_col="treatment", alpha=0.05)
    ea.get_effects()
    return ea


def test_fpr_summary_returns_dataframe():
    ea = _make_analyzer()
    summary = ea.fpr_summary(prior_success_rate=0.15)
    assert isinstance(summary, pd.DataFrame)


def test_fpr_summary_columns():
    ea = _make_analyzer()
    summary = ea.fpr_summary(prior_success_rate=0.15)
    for col in [
        "outcome",
        "n_total",
        "n_significant",
        "win_rate",
        "false_positive_risk",
        "estimated_true_success_rate",
    ]:
        assert col in summary.columns, f"Missing column: {col}"


def test_fpr_summary_values_in_range():
    ea = _make_analyzer()
    summary = ea.fpr_summary(prior_success_rate=0.15)
    assert (summary["false_positive_risk"] >= 0).all()
    assert (summary["false_positive_risk"] <= 1).all()
    assert (summary["estimated_true_success_rate"] >= 0).all()
    assert (summary["estimated_true_success_rate"] <= 1).all()


def test_fpr_summary_requires_get_effects():
    np.random.seed(0)
    n = 500
    df = pd.DataFrame({"treatment": np.random.binomial(1, 0.5, n), "outcome": np.random.binomial(1, 0.1, n)})
    ea = ExperimentAnalyzer(data=df, outcomes=["outcome"], treatment_col="treatment")
    with pytest.raises(ValueError, match="get_effects"):
        ea.fpr_summary(prior_success_rate=0.15)


def test_fpr_summary_invalid_prior():
    ea = _make_analyzer()
    with pytest.raises(ValueError):
        ea.fpr_summary(prior_success_rate=0.0)
    with pytest.raises(ValueError):
        ea.fpr_summary(prior_success_rate=1.5)


def _make_multi_experiment_analyzer():
    np.random.seed(7)
    rows = []
    for exp_id in [1, 2, 3]:
        n = 1000
        treatment = np.random.binomial(1, 0.5, n)
        outcome = np.random.binomial(1, 0.10 + 0.02 * treatment, n)
        rows.append(
            pd.DataFrame(
                {
                    "experiment": exp_id,
                    "treatment": treatment,
                    "outcome": outcome,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    ea = ExperimentAnalyzer(
        data=df,
        outcomes=["outcome"],
        treatment_col="treatment",
        experiment_identifier=["experiment"],
        alpha=0.05,
    )
    ea.get_effects()
    return ea


def test_aggregate_effects_fpr_column_present():
    ea = _make_multi_experiment_analyzer()
    agg = ea.aggregate_effects(prior_success_rate=0.15)
    assert "false_positive_risk" in agg.columns


def test_aggregate_effects_fpr_column_absent_by_default():
    ea = _make_multi_experiment_analyzer()
    agg = ea.aggregate_effects()
    assert "false_positive_risk" not in agg.columns


def test_aggregate_effects_fpr_values_valid():
    ea = _make_multi_experiment_analyzer()
    agg = ea.aggregate_effects(prior_success_rate=0.15)
    assert (agg["false_positive_risk"] >= 0).all()
    assert (agg["false_positive_risk"] <= 1).all()
