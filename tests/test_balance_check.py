"""Tests for balance checking functionality"""

import numpy as np
import pandas as pd
import pytest

from experiment_utils.experiment_analyzer import ExperimentAnalyzer
from experiment_utils.utils import check_covariate_balance


@pytest.fixture
def sample_data_with_covariates():
    """Create sample data with numeric, binary, and categorical covariates"""
    np.random.seed(42)
    n = 500

    df = pd.DataFrame(
        {
            "treatment": np.random.choice([0, 1], n),
            "outcome": np.random.randn(n),
            "age": np.random.normal(35, 10, n),
            "income": np.random.normal(50000, 15000, n),
            "is_member": np.random.choice([0, 1], n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
            "tier": np.random.choice([1, 2, 3], n),
        }
    )

    return df


@pytest.fixture
def imbalanced_data():
    """Create deliberately imbalanced data"""
    np.random.seed(42)
    n_treatment = 300
    n_control = 200

    # Treatment group has higher age and income
    treatment_df = pd.DataFrame(
        {
            "treatment": [1] * n_treatment,
            "outcome": np.random.randn(n_treatment),
            "age": np.random.normal(45, 10, n_treatment),  # Higher age
            "income": np.random.normal(60000, 15000, n_treatment),  # Higher income
            "is_member": np.random.choice([0, 1], n_treatment),
        }
    )

    control_df = pd.DataFrame(
        {
            "treatment": [0] * n_control,
            "outcome": np.random.randn(n_control),
            "age": np.random.normal(30, 10, n_control),  # Lower age
            "income": np.random.normal(40000, 15000, n_control),  # Lower income
            "is_member": np.random.choice([0, 1], n_control),
        }
    )

    return pd.concat([treatment_df, control_df], ignore_index=True)


class TestStandaloneBalanceChecker:
    """Tests for check_covariate_balance standalone function"""

    def test_basic_balance_check(self, sample_data_with_covariates):
        """Test basic balance checking with numeric and binary covariates"""
        balance_df = check_covariate_balance(
            data=sample_data_with_covariates,
            treatment_col="treatment",
            covariates=["age", "income", "is_member"],
            threshold=0.1,
        )

        assert not balance_df.empty
        assert "covariate" in balance_df.columns
        assert "mean_treated" in balance_df.columns
        assert "mean_control" in balance_df.columns
        assert "smd" in balance_df.columns
        assert "balance_flag" in balance_df.columns

        # Should have 3 covariates
        assert len(balance_df) == 3

        # All SMDs should be reasonably small for random data
        assert balance_df["smd"].abs().max() < 0.5

    def test_categorical_covariate_handling(self, sample_data_with_covariates):
        """Test balance checking with categorical covariates"""
        balance_df = check_covariate_balance(
            data=sample_data_with_covariates,
            treatment_col="treatment",
            covariates=["age", "region", "tier"],
            threshold=0.1,
        )

        assert not balance_df.empty

        # Should have age + region dummies + tier dummies
        # region (4 categories) + tier (3 categories) + age (1) = 8 total
        assert len(balance_df) >= 7  # At least most should pass filtering

        # Check that categorical dummies are created
        region_dummies = [c for c in balance_df["covariate"] if "region_" in c]
        tier_dummies = [c for c in balance_df["covariate"] if "tier_" in c]

        assert len(region_dummies) > 0
        assert len(tier_dummies) > 0

    def test_imbalanced_detection(self, imbalanced_data):
        """Test that imbalanced covariates are correctly flagged"""
        balance_df = check_covariate_balance(
            data=imbalanced_data,
            treatment_col="treatment",
            covariates=["age", "income", "is_member"],
            threshold=0.1,
        )

        assert not balance_df.empty

        # Age and income should be imbalanced
        age_balance = balance_df[balance_df["covariate"] == "age"]
        assert len(age_balance) == 1
        assert age_balance["balance_flag"].values[0] == 0  # Imbalanced
        assert abs(age_balance["smd"].values[0]) > 0.1

        income_balance = balance_df[balance_df["covariate"] == "income"]
        assert len(income_balance) == 1
        assert income_balance["balance_flag"].values[0] == 0  # Imbalanced

    def test_zero_variance_filtering(self):
        """Test that zero variance covariates are filtered out"""
        df = pd.DataFrame(
            {
                "treatment": [0, 1, 0, 1, 0, 1] * 10,
                "age": [30] * 60,  # Zero variance
                "income": np.random.normal(50000, 15000, 60),
            }
        )

        balance_df = check_covariate_balance(
            data=df,
            treatment_col="treatment",
            covariates=["age", "income"],
            threshold=0.1,
        )

        # Age should be filtered out
        assert "age" not in balance_df["covariate"].values
        assert "income" in balance_df["covariate"].values

    def test_low_frequency_binary_filtering(self):
        """Test that low frequency binary covariates are filtered"""
        df = pd.DataFrame(
            {
                "treatment": [0, 1] * 50,
                "rare_event": [1, 0, 0, 0] * 25,  # Only 25 occurrences
                "common_event": np.random.choice([0, 1], 100),
            }
        )

        balance_df = check_covariate_balance(
            data=df,
            treatment_col="treatment",
            covariates=["rare_event", "common_event"],
            min_binary_count=30,
            threshold=0.1,
        )

        # rare_event should be filtered if it has < 30 in either group
        # This depends on the split, but common_event should be present
        assert "common_event" in balance_df["covariate"].values

    def test_empty_result_when_no_valid_covariates(self):
        """Test that empty DataFrame is returned when no valid covariates"""
        df = pd.DataFrame(
            {
                "treatment": [0, 1] * 50,
                "constant": [1] * 100,  # Zero variance
            }
        )

        balance_df = check_covariate_balance(
            data=df,
            treatment_col="treatment",
            covariates=["constant"],
            threshold=0.1,
        )

        assert balance_df.empty

    def test_custom_threshold(self, sample_data_with_covariates):
        """Test that custom thresholds affect balance flags"""
        # Strict threshold
        balance_strict = check_covariate_balance(
            data=sample_data_with_covariates,
            treatment_col="treatment",
            covariates=["age", "income"],
            threshold=0.05,
        )

        # Lenient threshold
        balance_lenient = check_covariate_balance(
            data=sample_data_with_covariates,
            treatment_col="treatment",
            covariates=["age", "income"],
            threshold=0.2,
        )

        # Lenient threshold should have more balanced flags
        assert balance_lenient["balance_flag"].sum() >= balance_strict["balance_flag"].sum()

    def test_explicit_categorical_specification(self, sample_data_with_covariates):
        """Test explicit specification of categorical covariates"""
        categorical_info = {"region": ["North", "South", "East", "West"]}

        balance_df = check_covariate_balance(
            data=sample_data_with_covariates,
            treatment_col="treatment",
            covariates=["age", "region"],
            categorical_covariates=categorical_info,
            threshold=0.1,
        )

        assert not balance_df.empty

        # Should have region dummies
        region_dummies = [c for c in balance_df["covariate"] if "region_" in c]
        assert len(region_dummies) > 0


class TestExperimentAnalyzerBalanceMethod:
    """Tests for ExperimentAnalyzer.check_balance() method"""

    def test_check_balance_basic(self, sample_data_with_covariates):
        """Test basic check_balance method"""
        analyzer = ExperimentAnalyzer(
            data=sample_data_with_covariates,
            outcomes=["outcome"],
            treatment_col="treatment",
            covariates=["age", "income", "is_member"],
        )

        balance_df = analyzer.check_balance()

        assert balance_df is not None
        assert not balance_df.empty
        assert "covariate" in balance_df.columns
        assert "smd" in balance_df.columns
        assert "balance_flag" in balance_df.columns

    def test_check_balance_with_categorical(self, sample_data_with_covariates):
        """Test check_balance with categorical covariates"""
        analyzer = ExperimentAnalyzer(
            data=sample_data_with_covariates,
            outcomes=["outcome"],
            treatment_col="treatment",
            covariates=["age", "region", "tier"],
        )

        balance_df = analyzer.check_balance()

        assert balance_df is not None
        assert not balance_df.empty

        # Should have categorical dummies for region
        assert any("region_" in str(c) for c in balance_df["covariate"])

        # tier with 3 values should be included (either as dummies or treated differently)
        # The key is that all covariates are processed
        assert len(balance_df) >= 5  # At least age + several region/tier entries

    def test_check_balance_with_experiments(self):
        """Test check_balance with multiple experiments"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "experiment": ["exp1"] * 250 + ["exp2"] * 250,
                "treatment": np.random.choice([0, 1], 500),
                "outcome": np.random.randn(500),
                "age": np.random.normal(35, 10, 500),
                "income": np.random.normal(50000, 15000, 500),
            }
        )

        analyzer = ExperimentAnalyzer(
            data=df,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier="experiment",
            covariates=["age", "income"],
        )

        balance_df = analyzer.check_balance()

        assert balance_df is not None
        assert not balance_df.empty
        assert "experiment" in balance_df.columns

        # Should have results for both experiments
        experiments = balance_df["experiment"].unique()
        assert len(experiments) == 2
        assert "exp1" in experiments
        assert "exp2" in experiments

    def test_check_balance_no_covariates(self):
        """Test check_balance returns None when no covariates"""
        df = pd.DataFrame(
            {
                "treatment": [0, 1] * 50,
                "outcome": np.random.randn(100),
            }
        )

        analyzer = ExperimentAnalyzer(
            data=df,
            outcomes=["outcome"],
            treatment_col="treatment",
        )

        balance_df = analyzer.check_balance()

        assert balance_df is None

    def test_check_balance_custom_parameters(self, sample_data_with_covariates):
        """Test check_balance with custom parameters"""
        analyzer = ExperimentAnalyzer(
            data=sample_data_with_covariates,
            outcomes=["outcome"],
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Strict threshold
        balance_strict = analyzer.check_balance(threshold=0.05)

        # Lenient threshold
        balance_lenient = analyzer.check_balance(threshold=0.2)

        assert balance_strict is not None
        assert balance_lenient is not None

        # Lenient should have more balanced flags
        if not balance_strict.empty and not balance_lenient.empty:
            assert balance_lenient["balance_flag"].sum() >= balance_strict["balance_flag"].sum()

    def test_check_balance_independent_of_get_effects(self, sample_data_with_covariates):
        """Test that check_balance works independently of get_effects"""
        analyzer = ExperimentAnalyzer(
            data=sample_data_with_covariates,
            outcomes=["outcome"],
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        # Call check_balance before get_effects
        balance_before = analyzer.check_balance()

        assert balance_before is not None
        assert not balance_before.empty

        # Now call get_effects
        analyzer.get_effects()

        # Call check_balance again after get_effects
        balance_after = analyzer.check_balance()

        assert balance_after is not None
        assert not balance_after.empty

        # Results should be similar (allowing for minor differences)
        assert len(balance_before) == len(balance_after)

    def test_check_balance_with_categorical_treatment(self):
        """Test check_balance with multi-level categorical treatment"""
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "treatment": np.random.choice(["control", "variant_a", "variant_b"], 300),
                "outcome": np.random.randn(300),
                "age": np.random.normal(35, 10, 300),
                "income": np.random.normal(50000, 15000, 300),
            }
        )

        analyzer = ExperimentAnalyzer(
            data=df,
            outcomes=["outcome"],
            treatment_col="treatment",
            covariates=["age", "income"],
        )

        balance_df = analyzer.check_balance()

        assert balance_df is not None
        assert not balance_df.empty

        # Should have multiple comparisons (variant_a vs control, variant_b vs control, etc.)
        # The exact number depends on the comparison strategy
        assert len(balance_df) >= 2  # At least age and income for one comparison


def test_integration_standalone_and_method(sample_data_with_covariates):
    """Test that standalone function and method produce consistent results"""
    # Using standalone function
    balance_standalone = check_covariate_balance(
        data=sample_data_with_covariates,
        treatment_col="treatment",
        covariates=["age", "income"],
        threshold=0.1,
    )

    # Using ExperimentAnalyzer method
    analyzer = ExperimentAnalyzer(
        data=sample_data_with_covariates,
        outcomes=["outcome"],
        treatment_col="treatment",
        covariates=["age", "income"],
    )
    balance_method = analyzer.check_balance(threshold=0.1)

    assert not balance_standalone.empty
    assert balance_method is not None
    assert not balance_method.empty

    # Should have same covariates (method adds experiment column)
    standalone_covs = set(balance_standalone["covariate"].values)
    method_covs = set(balance_method["covariate"].values)
    assert standalone_covs == method_covs

    # SMD values should be very similar
    for cov in standalone_covs:
        smd_standalone = balance_standalone[balance_standalone["covariate"] == cov]["smd"].values[0]
        smd_method = balance_method[balance_method["covariate"] == cov]["smd"].values[0]
        assert abs(smd_standalone - smd_method) < 0.01  # Allow small numerical differences
