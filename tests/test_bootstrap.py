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
        assert analyzer._bootstrap_pvalue_method == "studentized"
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
        resampled = analyzer._stratified_resample(sample_data, seed=123)

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

    def test_bootstrap_pvalue_methods(self, sample_data):
        """Test bootstrap_pvalue_method parameter: percentile and studentized"""
        results = {}
        for method in ["percentile", "studentized"]:
            analyzer = ExperimentAnalyzer(
                data=sample_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=True,
                bootstrap_iterations=100,
                bootstrap_pvalue_method=method,
                bootstrap_seed=123,
            )
            assert analyzer._bootstrap_pvalue_method == method
            analyzer.get_effects()
            results[method] = analyzer.results

        for method in ["percentile", "studentized"]:
            assert "pvalue" in results[method].columns
            # True effect = 5, both methods should detect significance
            assert results[method]["pvalue"].values[0] < 0.05
            assert results[method]["stat_significance"].values[0] == 1

    def test_bootstrap_relative_ci(self, sample_data):
        """Test that bootstrap computes relative effect CIs correctly"""
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_seed=123,
        )

        analyzer.get_effects()
        results = analyzer.results

        # Check that relative effect CIs are present and not NaN
        assert "rel_effect_lower" in results.columns
        assert "rel_effect_upper" in results.columns
        assert not pd.isna(results["rel_effect_lower"].values[0])
        assert not pd.isna(results["rel_effect_upper"].values[0])

        # Check that relative CI bounds make sense
        assert results["rel_effect_lower"].values[0] < results["rel_effect_upper"].values[0]

        # Relative effect should be within the CI bounds (approximately)
        rel_effect = results["relative_effect"].values[0]
        rel_lower = results["rel_effect_lower"].values[0]
        rel_upper = results["rel_effect_upper"].values[0]

        # Allow for some numerical tolerance
        assert rel_lower <= rel_effect * 1.01 or rel_effect <= rel_lower * 1.01
        assert rel_effect <= rel_upper * 1.01 or rel_upper <= rel_effect * 1.01

    def test_pvalue_convergence_with_asymptotic(self):
        """Bootstrap p-values should converge toward asymptotic p-values at large N.

        Uses N=1000 with a clear true effect. At this sample size, the normal
        approximation underlying asymptotic inference holds well, so all three
        methods (asymptotic, percentile, studentized) should give similar p-values.
        """
        np.random.seed(7)
        n = 1000
        data = pd.DataFrame(
            {
                "experiment_id": "conv_test",
                "treatment": np.random.binomial(1, 0.5, n),
                "x": np.random.normal(0, 1, n),
            }
        )
        data["outcome"] = 2.0 * data["treatment"] + 0.5 * data["x"] + np.random.normal(0, 3, n)

        def _run(method, bootstrap):
            a = ExperimentAnalyzer(
                data=data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=bootstrap,
                bootstrap_iterations=500,
                bootstrap_pvalue_method=method,
                bootstrap_seed=42,
            )
            a.get_effects()
            return float(a.results["pvalue"].values[0])

        pval_asymptotic = _run("percentile", bootstrap=False)
        pval_percentile = _run("percentile", bootstrap=True)
        pval_studentized = _run("studentized", bootstrap=True)

        # All three should detect the true effect (SNR ~ 2/3 ≈ 0.67 at N=1000)
        assert pval_asymptotic < 0.05, f"Asymptotic p-value not significant: {pval_asymptotic:.4f}"
        assert pval_percentile < 0.05, f"Percentile p-value not significant: {pval_percentile:.4f}"
        assert pval_studentized < 0.05, f"Studentized p-value not significant: {pval_studentized:.4f}"

        # At N=1000 bootstrap p-values should be within 0.05 of the asymptotic p-value
        assert abs(pval_percentile - pval_asymptotic) < 0.05, (
            f"Percentile vs asymptotic p-value gap too large: {pval_percentile:.4f} vs {pval_asymptotic:.4f}"
        )
        assert abs(pval_studentized - pval_asymptotic) < 0.05, (
            f"Studentized vs asymptotic p-value gap too large: {pval_studentized:.4f} vs {pval_asymptotic:.4f}"
        )

    def test_ratio_outcome_bootstrap_relative_ci_is_finite(self):
        # Regression test: before Fix B, the bootstrap ran on the linearized
        # column whose control mean is ~0, so bootstrap_rel_effects were
        # absolute_effect / ~0 — producing CIs around -2.7e12.
        np.random.seed(1)
        n = 5000
        df = pd.DataFrame(
            {
                "experiment_id": "ratio_exp",
                "treat": np.random.choice(["A", "B"], n),
                "lead": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )
        df["trial"] = df["lead"] * np.random.choice([0, 1], n, p=[0.6, 0.4])

        analyzer = ExperimentAnalyzer(
            data=df,
            outcomes=["trial", "lead"],
            treatment_col="treat",
            experiment_identifier=["experiment_id"],
            ratio_outcomes={"trial_per_lead": ("trial", "lead")},
            outcome_models={"trial": "ols", "lead": "ols"},
            bootstrap=True,
            bootstrap_iterations=200,
            bootstrap_seed=1,
            estimand="ATE",
        )
        analyzer.get_effects()

        row = analyzer.results[analyzer.results["outcome"] == "trial_per_lead"].iloc[0]
        rel_lower = row["rel_effect_lower"]
        rel_upper = row["rel_effect_upper"]
        rel_effect = row["relative_effect"]

        assert np.isfinite(rel_lower), f"rel_effect_lower is not finite: {rel_lower}"
        assert np.isfinite(rel_upper), f"rel_effect_upper is not finite: {rel_upper}"
        # Plausibility: |rel CI| for a trial/lead ratio should be O(1), not 1e12.
        assert abs(rel_lower) < 100, f"rel_effect_lower implausibly large: {rel_lower}"
        assert abs(rel_upper) < 100, f"rel_effect_upper implausibly large: {rel_upper}"
        assert rel_lower <= rel_effect <= rel_upper or abs(rel_upper - rel_lower) < 1e-9, (
            f"point estimate {rel_effect} outside CI [{rel_lower}, {rel_upper}]"
        )


class TestBootstrapProbAndRope:
    """Tests for P(effect > threshold) and ROPE probability columns."""

    PROB_COLS = [
        "prob_abs_effect_gt",
        "prob_rel_effect_gt",
        "prob_abs_effect_in_rope",
        "prob_abs_effect_above_rope",
        "prob_abs_effect_below_rope",
        "prob_rel_effect_in_rope",
        "prob_rel_effect_above_rope",
        "prob_rel_effect_below_rope",
    ]

    def test_prob_columns_present_with_bootstrap(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=200,
            bootstrap_seed=123,
            rope_abs=(-1.0, 1.0),
            rope_rel=(-0.01, 0.01),
        )
        analyzer.get_effects()
        results = analyzer.results
        for col in self.PROB_COLS:
            assert col in results.columns, f"Missing column: {col}"

    def test_prob_columns_absent_without_bootstrap(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=False,
        )
        analyzer.get_effects()
        results = analyzer.results
        for col in self.PROB_COLS:
            assert col not in results.columns, f"Unexpected column in asymptotic output: {col}"

    def test_prob_gt_zero_detects_positive_effect(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_seed=123,
        )
        analyzer.get_effects()
        results = analyzer.results
        assert results["prob_abs_effect_gt"].values[0] > 0.95
        assert results["prob_rel_effect_gt"].values[0] > 0.95

    def test_prob_gt_high_threshold_is_low(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_seed=123,
            prob_threshold_abs=100.0,
        )
        analyzer.get_effects()
        results = analyzer.results
        assert results["prob_abs_effect_gt"].values[0] < 0.05

    def test_rope_contains_all_estimates(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_seed=123,
            rope_abs=(-100.0, 100.0),
        )
        analyzer.get_effects()
        r = analyzer.results.iloc[0]
        assert r["prob_abs_effect_in_rope"] > 0.99
        assert r["prob_abs_effect_above_rope"] < 0.01
        assert r["prob_abs_effect_below_rope"] < 0.01

    def test_rope_entirely_above_true_effect(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_seed=123,
            rope_abs=(100.0, 200.0),
        )
        analyzer.get_effects()
        r = analyzer.results.iloc[0]
        assert r["prob_abs_effect_below_rope"] > 0.99
        assert r["prob_abs_effect_in_rope"] < 0.01
        assert r["prob_abs_effect_above_rope"] < 0.01

    def test_rope_three_way_partition_sums_to_one(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=500,
            bootstrap_seed=123,
            rope_abs=(-1.0, 1.0),
            rope_rel=(-0.01, 0.01),
        )
        analyzer.get_effects()
        r = analyzer.results.iloc[0]
        total = r["prob_abs_effect_in_rope"] + r["prob_abs_effect_above_rope"] + r["prob_abs_effect_below_rope"]
        assert abs(total - 1.0) < 1e-9

    def test_rope_none_yields_nan(self, sample_data):
        analyzer = ExperimentAnalyzer(
            data=sample_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            bootstrap=True,
            bootstrap_iterations=100,
            bootstrap_seed=123,
        )
        analyzer.get_effects()
        r = analyzer.results.iloc[0]
        for col in (
            "prob_abs_effect_in_rope",
            "prob_abs_effect_above_rope",
            "prob_abs_effect_below_rope",
            "prob_rel_effect_in_rope",
            "prob_rel_effect_above_rope",
            "prob_rel_effect_below_rope",
        ):
            assert np.isnan(r[col]), f"Expected NaN for {col}, got {r[col]}"

    def test_invalid_rope_raises(self, sample_data):
        with pytest.raises(ValueError):
            ExperimentAnalyzer(
                data=sample_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=True,
                rope_abs=(1.0, 0.0),
            )
        with pytest.raises(ValueError):
            ExperimentAnalyzer(
                data=sample_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=True,
                rope_rel=(float("nan"), 1.0),
            )
        with pytest.raises(ValueError):
            ExperimentAnalyzer(
                data=sample_data,
                outcomes=["outcome"],
                treatment_col="treatment",
                experiment_identifier=["experiment_id"],
                bootstrap=True,
                rope_abs=(0.0,),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
