"""
Unit tests for bootstrap functionality in ExperimentAnalyzer
"""

import numpy as np
import pandas as pd
import pytest

from experiment_utils.experiment_analyzer import ExperimentAnalyzer


@pytest.fixture
def sample_data():
    """Generate sample experiment data"""
    np.random.seed(42)
    n = 300

    data = pd.DataFrame(
        {
            "experiment_id": "test_exp",
            "treatment": np.random.binomial(1, 0.5, n),
            "age": np.random.normal(35, 10, n),
            "gender": np.random.binomial(1, 0.5, n),
        }
    )

    # Generate outcome with known treatment effect
    data["outcome"] = (
        100
        + 0.5 * data["age"]
        + 3 * data["gender"]
        + 5 * data["treatment"]  # True effect = 5
        + np.random.normal(0, 10, n)
    )

    return data


class TestBootstrapInference:
    """Test bootstrap inference functionality"""

    def test_bootstrap_initialization(self, sample_data):
        """Test that bootstrap parameters are correctly initialized"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_ci_method="percentile",
            bootstrap_stratify=True,
            bootstrap_seed=123,
        )

        assert analyzer._bootstrap
        assert analyzer._bootstrap_iterations == 100
        assert analyzer._bootstrap_ci_method == "percentile"
        assert analyzer._bootstrap_stratify
        assert analyzer._bootstrap_seed == 123

    def test_bootstrap_results_columns(self, sample_data):
        """Test that bootstrap results include correct columns with inference_method"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=50,
            bootstrap_seed=123,
        )

        analyzer.get_effects()
        results = analyzer.results

        assert "inference_method" in results.columns
        assert results["inference_method"].values[0] == "bootstrap"
        assert "pvalue" in results.columns
        assert "abs_effect_lower" in results.columns
        assert "abs_effect_upper" in results.columns
        assert "standard_error" in results.columns

    def test_bootstrap_vs_asymptotic(self, sample_data):
        """Test that bootstrap and asymptotic results use same column names"""
        # Asymptotic
        analyzer_async = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=False,
        )
        analyzer_async.get_effects()
        results_async = analyzer_async.results

        # Bootstrap
        analyzer_boot = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_seed=123,
        )
        analyzer_boot.get_effects()
        results_boot = analyzer_boot.results

        # Effect estimates should be the same (same data)
        assert results_async["absolute_effect"].values[0] == results_boot["absolute_effect"].values[0]

        # Check inference_method column
        assert results_async["inference_method"].values[0] == "asymptotic"
        assert results_boot["inference_method"].values[0] == "bootstrap"

        # Both should have same columns
        assert "pvalue" in results_async.columns
        assert "pvalue" in results_boot.columns
        assert "abs_effect_lower" in results_async.columns
        assert "abs_effect_lower" in results_boot.columns

        # Both should detect statistical significance (true effect = 5)
        assert results_async["stat_significance"].values[0] == 1
        assert results_boot["stat_significance"].values[0] == 1
        assert results_boot["pvalue"].values[0] < 0.05

        # CIs should overlap but may differ slightly
        async_ci_width = results_async["abs_effect_upper"].values[0] - results_async["abs_effect_lower"].values[0]
        boot_ci_width = results_boot["abs_effect_upper"].values[0] - results_boot["abs_effect_lower"].values[0]

        # Both CI widths should be positive and reasonable
        assert async_ci_width > 0
        assert boot_ci_width > 0

        # They shouldn't be wildly different (allow 50% difference)
        assert 0.5 < boot_ci_width / async_ci_width < 2.0

    def test_bootstrap_override(self, sample_data):
        """Test overriding bootstrap setting in get_effects"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=False,  # Default is False
        )

        # Override to use bootstrap
        analyzer.get_effects(bootstrap=True)
        results = analyzer.results

        assert results["inference_method"].values[0] == "bootstrap"

        # Original setting should be restored
        assert not analyzer._bootstrap

    def test_bootstrap_stratified_resampling(self, sample_data):
        """Test that stratified resampling maintains treatment proportions"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_stratify=True,
            bootstrap_seed=123,
        )

        # Test the resampling function directly
        resampled = analyzer._ExperimentAnalyzer__stratified_resample(sample_data, seed=123)

        original_ratio = sample_data["treatment"].mean()
        resampled_ratio = resampled["treatment"].mean()

        # Ratios should be exactly the same with stratified resampling
        assert original_ratio == resampled_ratio

        # Should have same number of observations
        assert len(resampled) == len(sample_data)

    def test_bootstrap_with_covariates(self, sample_data):
        """Test bootstrap with covariate adjustment"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            covariates=["age", "gender"],
            regression_covariates=["age", "gender"],
            bootstrap=True,
            bootstrap_iterations=50,
            bootstrap_seed=123,
        )

        analyzer.get_effects()
        results = analyzer.results

        assert not results.empty
        assert results["inference_method"].values[0] == "bootstrap"

    def test_bootstrap_ci_methods(self, sample_data):
        """Test different bootstrap CI methods"""
        for method in ["percentile", "basic"]:
            analyzer = ExperimentAnalyzer(
                data=sample_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=True,
                bootstrap_iterations=50,
                bootstrap_ci_method=method,
                bootstrap_seed=123,
            )

            analyzer.get_effects()
            results = analyzer.results

            assert not results.empty
            assert "abs_effect_lower" in results.columns
            assert "abs_effect_upper" in results.columns

            # CI lower should be less than CI upper
            assert results["abs_effect_lower"].values[0] < results["abs_effect_upper"].values[0]

    def test_bootstrap_reproducibility(self, sample_data):
        """Test that bootstrap results are reproducible with same seed"""
        analyzer1 = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_seed=999,
        )
        analyzer1.get_effects()
        results1 = analyzer1.results

        analyzer2 = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_seed=999,
        )
        analyzer2.get_effects()
        results2 = analyzer2.results

        # Results should be identical
        assert results1["pvalue"].values[0] == results2["pvalue"].values[0]
        assert results1["abs_effect_lower"].values[0] == results2["abs_effect_lower"].values[0]
        assert results1["abs_effect_upper"].values[0] == results2["abs_effect_upper"].values[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
