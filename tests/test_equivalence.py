"""Tests for TOST equivalence testing."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from experiment_utils import ExperimentAnalyzer, PowerSim, plot_equivalence


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


class TestEquivalenceBoundTypes:
    """Tests for different bound specifications."""

    def test_relative_bound(self, simple_experiment_data):
        """Relative bound should compute delta from control_value."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", relative_bound=0.20)
        result = analyzer.results.iloc[0]
        expected_delta = 0.20 * abs(result["control_value"])
        assert result["eq_bound_upper"] == pytest.approx(expected_delta, rel=1e-6)
        assert result["eq_bound_lower"] == pytest.approx(-expected_delta, rel=1e-6)
        assert result["eq_bound_type"] == "relative"

    def test_cohens_d_bound_ols(self, simple_experiment_data):
        """Cohen's d bound should compute delta from pooled_sd."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", cohens_d_bound=0.5)
        result = analyzer.results.iloc[0]
        expected_delta = 0.5 * result["pooled_sd"]
        assert result["eq_bound_upper"] == pytest.approx(expected_delta, rel=1e-6)
        assert result["eq_bound_type"] == "cohens_d"

    def test_cohens_d_warns_non_ols(self):
        """Cohen's d with logistic model should produce NaN conclusion."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome": rng.binomial(1, 0.5, 2 * n),
                "experiment": "exp1",
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
            outcome_models="logistic",
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", cohens_d_bound=0.3)
        result = analyzer.results.iloc[0]
        assert pd.isna(result["eq_conclusion"])


class TestNonInferiority:
    """Tests for non-inferiority and non-superiority test types."""

    def test_non_inferiority_higher_is_better(self, simple_experiment_data):
        """Non-inferiority with higher_is_better tests lower bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "non_inferiority"
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_lower"])

    def test_non_inferiority_lower_is_better(self, simple_experiment_data):
        """Non-inferiority with lower_is_better tests upper bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="lower_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_upper"])

    def test_non_superiority_higher_is_better(self, simple_experiment_data):
        """Non-superiority with higher_is_better tests upper bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_superiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "non_superiority"
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_upper"])

    def test_non_inferiority_conclusion_labels(self, simple_experiment_data):
        """Non-inferiority should use 'non_inferior' labels, not 'equivalent'."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        valid_conclusions = {
            "non_inferior",
            "not_non_inferior",
            "inconclusive",
            "non_inferior_with_difference",
        }
        assert result["eq_conclusion"] in valid_conclusions


class TestEquivalenceMultiple:
    """Tests for multiple outcomes and experiments."""

    def test_multiple_outcomes(self):
        """TOST should produce results per outcome."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome1": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
                "outcome2": np.concatenate([rng.normal(5, 1, n), rng.normal(8, 1, n)]),
                "experiment": "exp1",
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome1", "outcome2"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        results = analyzer.results
        assert len(results) == 2
        o1 = results[results["outcome"] == "outcome1"].iloc[0]
        o2 = results[results["outcome"] == "outcome2"].iloc[0]
        assert o1["eq_conclusion"] in ("equivalent", "equivalent_with_difference")
        assert o2["eq_conclusion"] == "not_equivalent"

    def test_multiple_experiments(self):
        """TOST should produce results per experiment."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n + [0] * n + [1] * n,
                "outcome": np.concatenate(
                    [
                        rng.normal(10, 2, n),
                        rng.normal(10.1, 2, n),
                        rng.normal(10, 2, n),
                        rng.normal(10.05, 2, n),
                    ]
                ),
                "experiment": ["exp1"] * (2 * n) + ["exp2"] * (2 * n),
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        results = analyzer.results
        assert len(results) == 2
        assert results["eq_pvalue"].notna().all()

    def test_bootstrap_inference(self):
        """TOST should work with bootstrap SE."""
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
                "experiment": "exp1",
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
            bootstrap=True,
            bootstrap_iterations=100,
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert pd.notna(result["eq_pvalue"])
        assert pd.notna(result["eq_ci_lower"])
        assert result["eq_conclusion"] in (
            "equivalent",
            "equivalent_with_difference",
            "inconclusive",
            "not_equivalent",
        )


class TestEquivalenceValidation:
    """Tests for input validation in test_equivalence()."""

    def test_invalid_test_type(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(test_type="invalid", absolute_bound=1.0)

    def test_no_bounds_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(test_type="equivalence")

    def test_two_bounds_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=1.0, relative_bound=0.1)

    def test_negative_bound_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=-1.0)

    def test_invalid_alpha_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=1.0, alpha=0.0)

    def test_invalid_direction_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.test_equivalence(absolute_bound=1.0, direction="invalid")


class TestPlotEquivalence:
    """Tests for the plot_equivalence() standalone function."""

    def _make_equivalence_results(self):
        """Helper: create analyzer with equivalence results."""
        rng = np.random.default_rng(42)
        n = 500
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
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
        analyzer.test_equivalence(absolute_bound=1.0)
        return analyzer.results

    def test_returns_figure(self):
        """plot_equivalence should return a matplotlib Figure."""
        results = self._make_equivalence_results()
        fig = plot_equivalence(data=results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_without_eq_columns(self):
        """Should raise error if eq_ columns are missing."""
        data = pd.DataFrame({"absolute_effect": [0.1], "outcome": ["x"]})
        with pytest.raises(ValueError):
            plot_equivalence(data=data)

    def test_multiple_outcomes(self):
        """Should handle multiple outcomes in panels."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame(
            {
                "treatment": [0] * n + [1] * n,
                "outcome1": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
                "outcome2": np.concatenate([rng.normal(5, 1, n), rng.normal(5.05, 1, n)]),
                "experiment": "exp1",
            }
        )
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome1", "outcome2"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        fig = plot_equivalence(data=analyzer.results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotEquivalenceClassMethod:
    """Tests for ExperimentAnalyzer.plot_equivalence() wrapper."""

    def test_class_method_returns_figure(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        fig = analyzer.plot_equivalence()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_class_method_raises_without_test_equivalence(self, simple_experiment_data):
        """Should raise if test_equivalence() hasn't been called."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(ValueError):
            analyzer.plot_equivalence()


class TestPowerTOST:
    """Tests for PowerSim.power_tost()."""

    def test_high_power_scenario(self):
        """Large N, true_effect=0, wide bounds should give power near 1."""
        ps = PowerSim(metric="average", nsim=200)
        result = ps.power_tost(
            sample_sizes=[500, 1000],
            equivalence_bound=1.0,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        assert isinstance(result, pd.DataFrame)
        assert "sample_size" in result.columns
        assert "power" in result.columns
        high_n_power = result[result["sample_size"] == 1000]["power"].iloc[0]
        assert high_n_power > 0.7

    def test_low_power_scenario(self):
        """Small N should give low power."""
        ps = PowerSim(metric="average", nsim=200)
        result = ps.power_tost(
            sample_sizes=[10],
            equivalence_bound=0.5,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        low_n_power = result[result["sample_size"] == 10]["power"].iloc[0]
        assert low_n_power < 0.3

    def test_power_increases_with_sample_size(self):
        """Power should increase as sample size grows."""
        ps = PowerSim(metric="average", nsim=300)
        result = ps.power_tost(
            sample_sizes=[50, 200, 500],
            equivalence_bound=0.8,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        powers = result.sort_values("sample_size")["power"].tolist()
        assert powers[-1] >= powers[0]

    def test_proportion_metric(self):
        """power_tost should work with proportion metric."""
        ps = PowerSim(metric="proportion", nsim=200)
        result = ps.power_tost(
            sample_sizes=[200, 500],
            equivalence_bound=0.05,
            true_effect=0.0,
            pooled_sd=1.0,
            baseline=0.5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_output_compatible_with_plot_power(self):
        """Output should have columns compatible with plot_power."""
        ps = PowerSim(metric="average", nsim=100)
        result = ps.power_tost(
            sample_sizes=[100, 200],
            equivalence_bound=1.0,
            true_effect=0.0,
            pooled_sd=2.0,
        )
        assert "sample_size" in result.columns
        assert "power" in result.columns
        assert "se" in result.columns
        assert "nsim" in result.columns
