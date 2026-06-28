import numpy as np
import pandas as pd
import pytest

from experiment_utils import ExperimentAnalyzer


def _make_analyzer(seed=11, n=4000, lift=0.02):
    np.random.seed(seed)
    t = np.random.binomial(1, 0.5, n)
    y = np.random.binomial(1, 0.10 + lift * t, n)
    ea = ExperimentAnalyzer(
        data=pd.DataFrame({"treatment": t, "outcome": y}),
        outcomes=["outcome"],
        treatment_col="treatment",
        alpha=0.05,
    )
    ea.get_effects()
    return ea


def test_returns_dataframe():
    ea = _make_analyzer()
    out = ea.msprt_summary(tau=0.02)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(ea._results)


def test_columns_present():
    ea = _make_analyzer()
    out = ea.msprt_summary(tau=0.02)
    for col in ["outcome", "z", "se", "tau", "llr", "boundary", "should_stop"]:
        assert col in out.columns, f"Missing column: {col}"


def test_boundary_matches_alpha():
    ea = _make_analyzer()
    out = ea.msprt_summary(tau=0.02, alpha=0.05)
    assert abs(out.iloc[0]["boundary"] - np.log(1 / 0.05)) < 1e-9


def test_absolute_and_relative_agree():
    ea = _make_analyzer()
    ctrl = abs(ea._results.iloc[0]["control_value"])
    out_abs = ea.msprt_summary(tau=0.10 * ctrl)
    out_rel = ea.msprt_summary(tau_rel=0.10)
    assert abs(out_abs.iloc[0]["llr"] - out_rel.iloc[0]["llr"]) < 1e-9


def test_strong_signal_stops():
    ea = _make_analyzer(n=40000, lift=0.03)
    out = ea.msprt_summary(tau=0.03, alpha=0.05)
    assert bool(out.iloc[0]["should_stop"]) is True


def test_null_does_not_stop():
    ea = _make_analyzer(n=4000, lift=0.0)
    out = ea.msprt_summary(tau=0.02)
    assert bool(out.iloc[0]["should_stop"]) is False


def test_requires_get_effects():
    np.random.seed(0)
    n = 500
    df = pd.DataFrame({"treatment": np.random.binomial(1, 0.5, n), "outcome": np.random.binomial(1, 0.1, n)})
    ea = ExperimentAnalyzer(data=df, outcomes=["outcome"], treatment_col="treatment")
    with pytest.raises(ValueError, match="get_effects"):
        ea.msprt_summary(tau=0.02)


def test_no_tau_raises():
    ea = _make_analyzer()
    with pytest.raises(ValueError):
        ea.msprt_summary()


def test_both_tau_raises():
    ea = _make_analyzer()
    with pytest.raises(ValueError):
        ea.msprt_summary(tau=0.02, tau_rel=0.10)


def test_negative_tau_raises():
    ea = _make_analyzer()
    with pytest.raises(ValueError):
        ea.msprt_summary(tau=-0.01)


def test_degenerate_se_no_stop():
    ea = _make_analyzer()
    ea._results.loc[ea._results.index[0], "standard_error"] = 0.0
    out = ea.msprt_summary(tau=0.02)
    assert bool(out.iloc[0]["should_stop"]) is False
    assert np.isnan(out.iloc[0]["llr"])
