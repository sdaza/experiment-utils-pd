"""Tests for TOST equivalence testing."""

import numpy as np
import pandas as pd
import pytest

from experiment_utils import ExperimentAnalyzer


@pytest.fixture
def simple_experiment_data():
    """Generate simple experiment data with known properties."""
    rng = np.random.default_rng(42)
    n = 500
    control_values = rng.normal(10, 2, n)
    treatment_values = rng.normal(10.1, 2, n)
    outcome = np.concatenate([control_values, treatment_values])
    treatment = np.array([0] * n + [1] * n)
    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "experiment": "exp1",
        }
    )


class TestPooledSd:
    """Tests for pooled_sd column in get_effects() results."""

    def test_pooled_sd_present_in_results(self, simple_experiment_data):
        """pooled_sd column should be present after get_effects()."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        assert "pooled_sd" in analyzer.results.columns

    def test_pooled_sd_value_is_reasonable(self, simple_experiment_data):
        """pooled_sd should be close to the true SD (2.0) for our generated data."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        pooled_sd = analyzer.results["pooled_sd"].iloc[0]
        # True SD is 2.0, should be within 0.3
        assert abs(pooled_sd - 2.0) < 0.3


class TestEquivalenceTOST:
    """Tests for test_equivalence() with test_type='equivalence'."""

    def test_equivalent_with_absolute_bound(self, simple_experiment_data):
        """Near-zero effect with wide bounds should conclude 'equivalent'."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "equivalence"
        assert result["eq_bound_type"] == "absolute"
        assert result["eq_bound_lower"] == pytest.approx(-1.0)
        assert result["eq_bound_upper"] == pytest.approx(1.0)
        assert result["eq_pvalue_lower"] < 0.05
        assert result["eq_pvalue_upper"] < 0.05
        assert result["eq_pvalue"] < 0.05
        assert result["eq_conclusion"] in ("equivalent", "equivalent_with_difference")

    def test_not_equivalent_with_tight_bound(self):
        """Large effect with tight bounds should conclude 'not_equivalent'."""
        rng = np.random.default_rng(123)
        n = 200
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(15, 2, n)]),
                "experiment": "exp1",
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert result["eq_conclusion"] == "not_equivalent"

    def test_90_percent_ci_computed(self, simple_experiment_data):
        """90% CI (1-2*alpha) should be present and narrower than 95% CI."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0, alpha=0.05)
        result = analyzer.results.iloc[0]
        assert pd.notna(result["eq_ci_lower"])
        assert pd.notna(result["eq_ci_upper"])
        eq_width = result["eq_ci_upper"] - result["eq_ci_lower"]
        nhst_width = result["abs_effect_upper"] - result["abs_effect_lower"]
        assert eq_width < nhst_width

    def test_cohens_d_reported(self, simple_experiment_data):
        """eq_cohens_d should be absolute_effect / pooled_sd."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        expected_d = result["absolute_effect"] / result["pooled_sd"]
        assert result["eq_cohens_d"] == pytest.approx(expected_d, rel=1e-6)

    def test_requires_get_effects_first(self, simple_experiment_data):
        """Should raise error if get_effects() not called."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=1.0)

    def test_must_specify_exactly_one_bound(self, simple_experiment_data):
        """Should raise error if zero or multiple bounds specified."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence()  # no bound
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=1.0, relative_bound=0.1)  # two bounds
