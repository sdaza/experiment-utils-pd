"""Tests for the critical effect size columns (Perugini et al. 2025).

critical_effect is the smallest absolute effect that would reach statistical
significance at alpha given the standard error; critical_effect_mcp is the same
at the MCP-adjusted per-comparison alpha.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from experiment_utils import ExperimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    rng = np.random.default_rng(11)
    n = 1000
    data = pd.DataFrame(
        {
            "treatment": [0] * n + [1] * n,
            "revenue": np.concatenate([rng.normal(10, 2, n), rng.normal(10.5, 2, n)]),
            "engaged": np.concatenate([rng.binomial(1, 0.30, n), rng.binomial(1, 0.36, n)]),
            "experiment": "exp1",
        }
    )
    ea = ExperimentAnalyzer(
        data=data,
        outcomes=["revenue", "engaged"],
        treatment_col="treatment",
        experiment_identifier=["experiment"],
    )
    ea.get_effects()
    return ea


def test_critical_effect_column_present(analyzer):
    assert "critical_effect" in analyzer.results.columns
    assert analyzer.results["critical_effect"].notna().all()
    assert (analyzer.results["critical_effect"] > 0).all()


def test_critical_effect_matches_closed_form(analyzer):
    """critical_effect = t/z critical value * SE at the analyzer's alpha."""
    # df_resid is internal-only, so check against the private frame
    internal = analyzer._results
    for _, row in internal.iterrows():
        if "df_resid" in internal.columns and pd.notna(row.get("df_resid")):
            crit = stats.t.ppf(1 - 0.05 / 2, row["df_resid"])
        else:
            crit = stats.norm.ppf(1 - 0.05 / 2)
        assert row["critical_effect"] == pytest.approx(crit * row["standard_error"], rel=1e-9)


def test_critical_effect_consistent_with_significance(analyzer):
    """|effect| >= critical_effect exactly when the row is significant."""
    for _, row in analyzer.results.iterrows():
        if row["stat_significance"] == 1:
            assert abs(row["absolute_effect"]) >= row["critical_effect"] * 0.999
        else:
            assert abs(row["absolute_effect"]) <= row["critical_effect"] * 1.001


def test_mcp_adds_alpha_and_critical_effect(analyzer):
    analyzer.adjust_pvalues(method="bonferroni")
    results = analyzer.results
    assert "alpha_mcp" in results.columns
    assert "critical_effect_mcp" in results.columns
    # two outcomes in the family -> Bonferroni alpha/2
    assert np.allclose(results["alpha_mcp"], 0.05 / 2)
    # stricter alpha -> larger critical effect
    assert (results["critical_effect_mcp"] > results["critical_effect"]).all()
