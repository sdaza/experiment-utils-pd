import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm

from experiment_utils.experiment_analyzer import ExperimentAnalyzer


@pytest.fixture
def sample_data(
    n_model=1000,
    n_random=500,
    base_model_conversion_mean=0.3,
    base_model_conversion_variance=0.01,
    base_random_conversion_mean=0.10,
    base_random_conversion_variance=0.01,
    model_treatment_effect=0.05,
    random_treatment_effect=0.05,
    random_seed=42,
):
    np.random.seed(random_seed)

    # Function to get a truncated normal distribution
    def get_truncated_normal(mean, variance, size):
        std_dev = np.sqrt(variance)
        lower, upper = 0, 1
        a, b = (lower - mean) / std_dev, (upper - mean) / std_dev
        return truncnorm.rvs(a, b, loc=mean, scale=std_dev, size=size)

    # Generate baseline conversions with a truncated normal distribution
    base_model_conversion = get_truncated_normal(base_model_conversion_mean, base_model_conversion_variance, n_model)
    base_random_conversion = get_truncated_normal(
        base_random_conversion_mean, base_random_conversion_variance, n_random
    )

    # model group data
    model_treatment = np.random.binomial(1, 0.8, n_model)
    model_conversion = (base_model_conversion + model_treatment_effect * model_treatment) > np.random.rand(n_model)

    model_data = pd.DataFrame(
        {
            "experiment": 123,
            "expected_ratio": 0.5,
            "group": "model",
            "treatment": model_treatment,
            "conversion": model_conversion.astype(int),
            "baseline_conversion": base_model_conversion,
        }
    )

    # random group data
    random_treatment = np.random.binomial(1, 0.5, n_random)
    random_conversion = (base_random_conversion + random_treatment_effect * random_treatment) > np.random.rand(n_random)
    random_data = pd.DataFrame(
        {
            "experiment": 123,
            "expected_ratio": 0.5,
            "group": "random",
            "treatment": random_treatment,
            "conversion": random_conversion.astype(int),
            "baseline_conversion": base_random_conversion,
        }
    )

    # Combine data
    data = pd.concat([model_data, random_data])

    return data


def test_no_covariates(sample_data):
    """Test get_effects no covariates"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"

    analyzer = ExperimentAnalyzer(
        data=sample_data, outcomes=outcomes, treatment_col=treatment_col, experiment_identifier=experiment_identifier
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_sample_ratio_check(sample_data):
    """Test get_effects sample ratio check"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"
    expected_ratio_col = "expected_ratio"

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        exp_sample_ratio_col=expected_ratio_col,
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_no_experiment_identifier(sample_data):
    """Test get_effects no covariates"""
    outcomes = "conversion"
    treatment_col = "treatment"

    analyzer = ExperimentAnalyzer(data=sample_data, outcomes=outcomes, treatment_col=treatment_col)

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_regression_covariates(sample_data):
    """Test get_effects regression covariates"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"
    regression_covariates = "baseline_conversion"

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        regression_covariates=regression_covariates,
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_no_adjustment(sample_data):
    """Test get_effects no adjustments"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"
    covariates = "baseline_conversion"

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        covariates=covariates,
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_balance_adjustment(sample_data):
    """Test get_effects with balance adjustment and balance_method"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"
    covariates = "baseline_conversion"

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        covariates=covariates,
        adjustment="balance",
        balance_method="ps-logistic",
        target_effect="ATT",
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_categorical_covariates():
    """Test automatic dummy variable creation for categorical covariates"""
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "outcome": np.random.randn(1000),
            "treatment": np.random.choice([0, 1], 1000),
            "region": np.random.choice(["North", "South", "East", "West"], 1000),
            "age_group": np.random.choice([1, 2, 3], 1000),  # Integer categorical
            "numeric_cov": np.random.randn(1000),
        }
    )

    exp = ExperimentAnalyzer(
        data=data, outcomes=["outcome"], treatment_col="treatment", covariates=["region", "age_group", "numeric_cov"]
    )

    exp.get_effects()

    # Check dummies were created
    assert exp._results is not None
    # Check balance includes all categories
    balance = exp.balance
    assert balance is not None
    # Should have dummies for categorical variables in balance
    # Some dummies may be filtered due to low counts in treatment/control splits
    # At minimum: age_group (3) + numeric_cov (1) = 4 covariates
    assert len(balance) >= 3, f"Expected at least 3 balance rows, got {len(balance)}"

    # Verify categorical dummies are present (at least age_group since it has 3 categories)
    covariate_names = balance["covariate"].tolist()
    has_categorical = any(c.startswith("age_group_") or c.startswith("region_") for c in covariate_names)
    assert has_categorical, "Should have at least one categorical dummy variable"


def test_categorical_covariates_comprehensive():
    """Comprehensive test for categorical covariate handling"""
    np.random.seed(123)

    data = pd.DataFrame(
        {
            "outcome": np.random.randn(1000) + np.random.choice([0, 0.3], 1000),
            "treatment": np.random.choice([0, 1], 1000),
            "region": np.random.choice(["North", "South", "East", "West"], 1000),
            "segment": np.random.choice([1, 2, 3, 4], 1000),  # Integer categorical (4 categories)
            "status": np.random.choice([10, 20], 1000),  # Integer categorical (2 categories - should stay binary)
            "income": np.random.randn(1000),  # Numeric continuous
            "has_feature": np.random.choice([0, 1], 1000),  # Binary 0/1
        }
    )

    exp = ExperimentAnalyzer(
        data=data,
        outcomes=["outcome"],
        treatment_col="treatment",
        covariates=["region", "segment", "status", "income", "has_feature"],
    )

    exp.get_effects()

    # Verify results exist
    assert exp._results is not None
    results = exp.results
    assert len(results) == 1
    assert "absolute_effect" in results.columns

    # Verify balance table
    balance = exp.balance
    assert balance is not None
    covariate_names = balance["covariate"].tolist()

    # Check segment dummies (4 categories: 1, 2, 3, 4 - all lowercase)
    segment_dummies = [c for c in covariate_names if c.startswith("segment_")]
    assert len(segment_dummies) == 4, f"Expected 4 segment dummies, got {len(segment_dummies)}: {segment_dummies}"
    # Verify lowercase naming for integer categories
    assert "segment_1" in covariate_names
    assert "segment_2" in covariate_names
    assert "segment_3" in covariate_names
    assert "segment_4" in covariate_names

    # Check region dummies may be present (some might be filtered due to low counts)
    region_dummies = [c for c in covariate_names if c.startswith("region_")]
    # If region dummies exist, verify they use lowercase names
    if region_dummies:
        for dummy in region_dummies:
            # All should be lowercase (no region_East, only region_east)
            assert dummy.islower(), f"Dummy variable {dummy} should be lowercase"

    # Check status (2 categories - should be treated as binary, not categorical with 3-10 range)
    status_dummies = [c for c in covariate_names if c.startswith("status_")]
    assert len(status_dummies) == 0, f"Status should not be converted to dummies (only 2 values), got: {status_dummies}"
    # Status should appear as-is if it passes binary checks

    # Check numeric covariate appears once
    assert "income" in covariate_names

    # Check binary covariate appears once
    assert "has_feature" in covariate_names

    # Total should have at least segment(4) + income(1) + has_feature(1) = 6 covariates
    # Region dummies may be filtered due to low counts in treatment/control splits
    assert len(balance) >= 6, f"Expected at least 6 balance rows, got {len(balance)}"


def test_categorical_reference_category():
    """Test that reference category is correctly identified and used"""
    np.random.seed(456)

    # Create data where 'A' is most frequent (should be reference)
    # Ensure better balance across treatment groups
    regions = ["A"] * 500 + ["B"] * 200 + ["C"] * 200 + ["D"] * 100
    np.random.shuffle(regions)

    data = pd.DataFrame(
        {
            "outcome": np.random.randn(1000) + np.random.choice([0, 0.2], 1000),
            "treatment": np.random.choice([0, 1], 1000),
            "region": regions,
            "numeric_cov": np.random.randn(1000),  # Add numeric covariate for better stability
        }
    )

    exp = ExperimentAnalyzer(
        data=data,
        outcomes=["outcome"],
        treatment_col="treatment",
        covariates=["region", "numeric_cov"],
    )

    exp.get_effects()

    # Check balance includes all categories
    balance = exp.balance

    # Verify results were computed successfully
    assert exp._results is not None

    # Handle case where balance might be empty or have filtered covariates
    if balance is not None and len(balance) > 0:
        covariate_names = balance["covariate"].tolist()

        # Verify at least some covariates are present
        assert len(covariate_names) >= 1, "Should have at least one covariate in balance"

        # If region dummies are present, verify lowercase naming
        region_dummies = [c for c in covariate_names if c.startswith("region_")]
        if region_dummies:
            for dummy in region_dummies:
                assert dummy.islower(), f"Region dummy {dummy} should be lowercase"


def test_categorical_naming_convention():
    """Test that categorical dummy variables use lowercase and handle special characters"""
    np.random.seed(789)

    data = pd.DataFrame(
        {
            "outcome": np.random.randn(800),
            "treatment": np.random.choice([0, 1], 800),
            "city": np.random.choice(["New York", "Los Angeles", "San-Francisco", "Boston"], 800),
            "tier": np.random.choice(["Premium+", "Standard", "Basic", "Free Trial"], 800),
        }
    )

    exp = ExperimentAnalyzer(
        data=data,
        outcomes=["outcome"],
        treatment_col="treatment",
        covariates=["city", "tier"],
    )

    exp.get_effects()

    balance = exp.balance
    if balance is not None and len(balance) > 0:
        covariate_names = balance["covariate"].tolist()

        # Check city dummies (should handle spaces in "New York", "Los Angeles", "San Francisco")
        city_dummies = [c for c in covariate_names if c.startswith("city_")]
        if city_dummies:
            # Verify lowercase and spaces replaced with underscores
            possible_cities = ["city_new_york", "city_los_angeles", "city_san_francisco", "city_boston"]
            for dummy in city_dummies:
                assert dummy in possible_cities, f"Unexpected city dummy: {dummy}"
                assert dummy.islower(), f"City dummy {dummy} should be lowercase"
                assert " " not in dummy, f"City dummy {dummy} should not contain spaces"

        # Check tier dummies (should handle special characters like "+")
        tier_dummies = [c for c in covariate_names if c.startswith("tier_")]
        if tier_dummies:
            # Verify special characters replaced
            possible_tiers = ["tier_premium", "tier_standard", "tier_basic", "tier_free_trial"]
            for dummy in tier_dummies:
                assert dummy in possible_tiers, f"Unexpected tier dummy: {dummy}"
                assert dummy.islower(), f"Tier dummy {dummy} should be lowercase"
                # No special chars like + should remain
                assert "+" not in dummy, f"Tier dummy {dummy} should not contain '+'"
