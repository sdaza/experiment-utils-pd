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
