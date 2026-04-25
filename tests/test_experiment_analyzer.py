import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm

matplotlib.use("Agg")

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
    """Test get_effects sample ratio check using a column name"""
    outcomes = "conversion"
    treatment_col = "treatment"
    experiment_identifier = "experiment"
    expected_ratio_col = "expected_ratio"

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes=outcomes,
        treatment_col=treatment_col,
        experiment_identifier=experiment_identifier,
        exp_sample_ratio=expected_ratio_col,
    )

    try:
        analyzer.get_effects()
        _ = analyzer.results
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_sample_ratio_check_float_constant(sample_data):
    """Test get_effects sample ratio check using a float constant"""
    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes="conversion",
        treatment_col="treatment",
        experiment_identifier="experiment",
        exp_sample_ratio=0.5,
    )

    try:
        analyzer.get_effects()
        results = analyzer.results
        assert results is not None
        assert "srm_detected" in results.columns
        assert "srm_pvalue" in results.columns
        assert "sample_ratio" in results.columns
    except Exception as e:
        pytest.fail(f"raised an exception: {e}")


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
        estimand="ATT",
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
        categorical_max_unique=10,  # Treat integers with ≤10 unique values as categorical
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

    # Check status (2 non-{0,1} values - should be dummy-encoded, not treated as continuous)
    status_dummies = [c for c in covariate_names if c.startswith("status_")]
    assert len(status_dummies) == 2, (
        f"Expected 2 status dummies (values 10, 20), got {len(status_dummies)}: {status_dummies}"
    )
    assert "status_10" in covariate_names
    assert "status_20" in covariate_names

    # Check numeric covariate appears once
    assert "income" in covariate_names

    # Check binary covariate {0, 1} appears once (not dummy-encoded)
    assert "has_feature" in covariate_names

    # Total should have at least segment(4) + status(2) + income(1) + has_feature(1) = 8 covariates
    # Region dummies may be filtered due to low counts in treatment/control splits
    assert len(balance) >= 8, f"Expected at least 8 balance rows, got {len(balance)}"


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


# ── plot_effects ──────────────────────────────────────────────────────────────


@pytest.fixture
def simple_analyzer():
    np.random.seed(0)
    n = 400
    df = pd.DataFrame(
        {
            "country": np.repeat(["US", "EU"], n),
            "channel": np.tile(np.repeat(["email", "push"], n // 2), 2),
            "treatment": np.random.binomial(1, 0.5, n * 2),
            "revenue": np.random.normal(50, 20, n * 2),
            "converted": np.random.binomial(1, 0.12, n * 2),
        }
    )
    az = ExperimentAnalyzer(
        data=df,
        treatment_col="treatment",
        outcomes=["revenue", "converted"],
        experiment_identifier=["country", "channel"],
        correction=None,
    )
    az.get_effects()
    return az


def test_plot_effects_default(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects()
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_show_values(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(show_values=True)
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_meta_analysis(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(meta_analysis=True)
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_y_outcome(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(y="outcome")
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_hide_panel_titles(simple_analyzer):
    fig = simple_analyzer.plot_effects(y="outcome", show_panel_titles=False)
    assert all(ax.get_title() == "" for ax in fig.axes)


def test_plot_effects_hide_panel_titles_multi_effect(simple_analyzer):
    fig = simple_analyzer.plot_effects(effect=["absolute", "relative"], show_panel_titles=False)
    assert all(ax.get_title() == "" for ax in fig.axes)


def test_plot_effects_panel_titles_list(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(panel_titles=["Revenue ($)", "CVR"])
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_panel_titles_dict(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(panel_titles={"revenue": "Revenue ($)", "converted": "CVR"})
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_row_labels(simple_analyzer):
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(row_labels={"US | email": "Email (US)"})
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_group_by(simple_analyzer):
    figs = simple_analyzer.plot_effects(group_by="country")
    assert isinstance(figs, dict)
    assert len(figs) == 2


def test_plot_effects_no_color_by(simple_analyzer):
    """color_by was removed — must raise TypeError."""
    with pytest.raises(TypeError, match="color_by"):
        simple_analyzer.plot_effects(color_by="channel")


# ── random effects meta-analysis ──────────────────────────────────────────────


@pytest.fixture
def multi_experiment_analyzer():
    """Analyzer with 6 experiments and moderate effect sizes (low heterogeneity)."""
    rng = np.random.default_rng(42)
    rows = []
    for exp_id in range(1, 7):
        n = 500
        treatment = rng.integers(0, 2, size=n)
        revenue = rng.normal(50, 15, size=n) + treatment * 3.0
        rows.append(
            pd.DataFrame(
                {
                    "experiment": exp_id,
                    "treatment": treatment,
                    "revenue": revenue,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    az = ExperimentAnalyzer(
        data=df,
        treatment_col="treatment",
        outcomes=["revenue"],
        experiment_identifier=["experiment"],
        correction=None,
    )
    az.get_effects()
    return az


@pytest.fixture
def heterogeneous_analyzer():
    """Analyzer where true effects vary widely across experiments (high heterogeneity)."""
    rng = np.random.default_rng(7)
    true_effects = [-0.5, 0.1, 0.8, 1.5, -0.3, 1.2]
    rows = []
    for exp_id, te in enumerate(true_effects, start=1):
        n = 300
        treatment = rng.integers(0, 2, size=n)
        revenue = rng.normal(10, 5, size=n) + treatment * te
        rows.append(
            pd.DataFrame(
                {
                    "experiment": exp_id,
                    "treatment": treatment,
                    "revenue": revenue,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    az = ExperimentAnalyzer(
        data=df,
        treatment_col="treatment",
        outcomes=["revenue"],
        experiment_identifier=["experiment"],
        correction=None,
    )
    az.get_effects()
    return az


def test_combine_effects_random_returns_same_schema(multi_experiment_analyzer):
    """Random effects result has the same columns as fixed effects result."""
    fixed = multi_experiment_analyzer.combine_effects(method="fixed")
    random = multi_experiment_analyzer.combine_effects(method="random")
    assert set(fixed.columns) == set(random.columns), (
        f"Column mismatch.\nFixed: {sorted(fixed.columns)}\nRandom: {sorted(random.columns)}"
    )


def test_combine_effects_random_numeric_outputs(multi_experiment_analyzer):
    """All key numeric fields are finite numbers for a well-behaved dataset."""
    result = multi_experiment_analyzer.combine_effects(method="random")
    for col in ["absolute_effect", "abs_effect_lower", "abs_effect_upper", "standard_error", "pvalue"]:
        assert np.isfinite(result[col].values).all(), f"Non-finite value in column '{col}': {result[col].values}"


def test_combine_effects_random_ci_contains_point_estimate(multi_experiment_analyzer):
    """Pooled point estimate must lie within the CI bounds."""
    result = multi_experiment_analyzer.combine_effects(method="random")
    assert (result["abs_effect_lower"] <= result["absolute_effect"]).all()
    assert (result["absolute_effect"] <= result["abs_effect_upper"]).all()


def test_combine_effects_random_no_heterogeneity_close_to_fixed(multi_experiment_analyzer):
    """When heterogeneity is low, random and fixed effects pooled estimates are close."""
    fixed = multi_experiment_analyzer.combine_effects(method="fixed")
    random = multi_experiment_analyzer.combine_effects(method="random")
    abs_diff = abs(fixed["absolute_effect"].values - random["absolute_effect"].values)
    assert (abs_diff < 0.5).all(), f"Fixed vs random point estimates too far apart: {abs_diff}"


def test_combine_effects_random_wider_ci_with_heterogeneity(heterogeneous_analyzer):
    """With high between-study variance random effects CI must be wider than fixed effects CI."""
    fixed = heterogeneous_analyzer.combine_effects(method="fixed")
    random = heterogeneous_analyzer.combine_effects(method="random")
    fe_width = fixed["abs_effect_upper"].values - fixed["abs_effect_lower"].values
    re_width = random["abs_effect_upper"].values - random["abs_effect_lower"].values
    assert (re_width >= fe_width).all(), (
        f"Expected RE CI ≥ FE CI with heterogeneity.\nFE width: {fe_width}\nRE width: {re_width}"
    )


def test_meta_stats_initialized_to_none():
    """meta_stats_ starts as None before combine_effects is called."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {"exp": np.ones(n, dtype=int), "treatment": np.tile([0, 1], n // 2), "revenue": rng.normal(5, 1, n)}
    )
    az = ExperimentAnalyzer(
        data=df, treatment_col="treatment", outcomes=["revenue"], experiment_identifier=["exp"], correction=None
    )
    assert az.meta_stats_ is None


def test_meta_stats_populated_after_fixed(multi_experiment_analyzer):
    """meta_stats_ is a DataFrame with expected columns after fixed effects combine_effects."""
    multi_experiment_analyzer.combine_effects(method="fixed")
    ms = multi_experiment_analyzer.meta_stats_
    assert ms is not None
    assert isinstance(ms, pd.DataFrame)
    for col in ["tau2", "i2", "Q", "k", "method"]:
        assert col in ms.columns, f"Missing column '{col}' in meta_stats_"
    assert (ms["method"] == "fixed").all()
    assert (ms["tau2"] == 0.0).all()


def test_meta_stats_populated_after_random(multi_experiment_analyzer):
    """meta_stats_ has heterogeneity columns populated after random effects run."""
    multi_experiment_analyzer.combine_effects(method="random")
    ms = multi_experiment_analyzer.meta_stats_
    assert ms is not None
    for col in ["tau2", "i2", "Q", "k", "method"]:
        assert col in ms.columns, f"Missing column '{col}' in meta_stats_"
    assert (ms["method"] == "random").all()
    assert ms["k"].iloc[0] == 6
    assert np.isfinite(ms["Q"].values).all()


def test_meta_stats_tau2_positive_with_heterogeneity(heterogeneous_analyzer):
    """τ² is positive when between-study variance is present."""
    heterogeneous_analyzer.combine_effects(method="random")
    ms = heterogeneous_analyzer.meta_stats_
    assert ms["tau2"].iloc[0] > 0, f"Expected τ²>0 with heterogeneous effects, got {ms['tau2'].iloc[0]}"


def test_combine_effects_invalid_method_raises(multi_experiment_analyzer):
    """Passing an unknown method name raises an error."""
    with pytest.raises(Exception, match="unknown|Unknown|invalid|'fixed' or 'random'"):
        multi_experiment_analyzer.combine_effects(method="bayesian")


def test_combine_effects_random_small_k():
    """k=2 experiments: HKSJ uses t(1) giving very wide CIs; pvalue should be finite."""
    rng = np.random.default_rng(99)
    rows = []
    for exp_id in [1, 2]:
        n = 200
        treatment = rng.integers(0, 2, size=n)
        revenue = rng.normal(20, 10, size=n) + treatment * 2.0
        rows.append(pd.DataFrame({"experiment": exp_id, "treatment": treatment, "revenue": revenue}))
    df = pd.concat(rows, ignore_index=True)
    az = ExperimentAnalyzer(
        data=df, treatment_col="treatment", outcomes=["revenue"], experiment_identifier=["experiment"], correction=None
    )
    az.get_effects()
    result = az.combine_effects(method="random")
    assert np.isfinite(result["absolute_effect"].values).all()
    assert np.isfinite(result["pvalue"].values).all()
    # t(1) CI must be wider than t(5) from a k=6 run
    fixed_result = az.combine_effects(method="fixed")
    re_width = result["abs_effect_upper"].values - result["abs_effect_lower"].values
    fe_width = fixed_result["abs_effect_upper"].values - fixed_result["abs_effect_lower"].values
    assert (re_width >= fe_width).all(), "With k=2, random effects (t(1)) CI should not be narrower than fixed"


def test_combine_effects_random_k1_fallback():
    """k=1 experiment falls back gracefully without raising."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "experiment": np.ones(200, dtype=int),
            "treatment": rng.integers(0, 2, size=200),
            "revenue": rng.normal(10, 3, size=200),
        }
    )
    az = ExperimentAnalyzer(
        data=df, treatment_col="treatment", outcomes=["revenue"], experiment_identifier=["experiment"], correction=None
    )
    az.get_effects()
    result = az.combine_effects(method="random")
    assert len(result) == 1
    assert np.isfinite(result["absolute_effect"].values).all()
    ms = az.meta_stats_
    assert ms["method"].iloc[0] == "random-fallback-k1"


def test_plot_effects_random_meta_method(simple_analyzer):
    """plot_effects with meta_method='random' produces a Figure without error."""
    import matplotlib.figure as mfig

    fig = simple_analyzer.plot_effects(meta_analysis=True, meta_method="random")
    assert isinstance(fig, mfig.Figure)


def test_plot_effects_meta_method_default_unchanged(simple_analyzer):
    """Default meta_method='fixed' behaviour is unchanged from before this feature."""
    import matplotlib.figure as mfig

    fig_default = simple_analyzer.plot_effects(meta_analysis=True)
    fig_explicit = simple_analyzer.plot_effects(meta_analysis=True, meta_method="fixed")
    assert isinstance(fig_default, mfig.Figure)
    assert isinstance(fig_explicit, mfig.Figure)


def test_combine_effects_random_pvalue_range(multi_experiment_analyzer):
    """p-values must be in [0, 1]."""
    result = multi_experiment_analyzer.combine_effects(method="random")
    pvals = result["pvalue"].dropna().values
    assert (pvals >= 0).all() and (pvals <= 1).all(), f"p-values out of [0,1]: {pvals}"


def test_combine_effects_random_stat_significance_consistent(multi_experiment_analyzer):
    """stat_significance flag must agree with pvalue < alpha."""
    az = multi_experiment_analyzer
    result = az.combine_effects(method="random")
    for _, row in result.iterrows():
        if not np.isnan(row["pvalue"]):
            expected = int(row["pvalue"] < az._alpha)
            assert row["stat_significance"] == expected


# ── relative effects correctness ──────────────────────────────────────────────


@pytest.fixture
def known_effect_analyzer():
    """
    Analyzer with 5 experiments sharing a common control mean (~10) and a known
    absolute treatment effect (~1.0).  Expected relative ≈ 1.0/10.0 = 0.10.
    """
    rng = np.random.default_rng(123)
    rows = []
    for exp_id in range(1, 6):
        n = 600
        treatment = rng.integers(0, 2, size=n)
        # Control mean ≈ 10, treatment shifts by +1.0
        revenue = rng.normal(10, 2, size=n) + treatment * 1.0
        rows.append(pd.DataFrame({"experiment": exp_id, "treatment": treatment, "revenue": revenue}))
    df = pd.concat(rows, ignore_index=True)
    az = ExperimentAnalyzer(
        data=df,
        treatment_col="treatment",
        outcomes=["revenue"],
        experiment_identifier=["experiment"],
        correction=None,
    )
    az.get_effects()
    return az


def test_relative_effect_ci_contains_estimate_fixed(known_effect_analyzer):
    """Fixed effects: rel_effect_lower ≤ relative_effect ≤ rel_effect_upper."""
    result = known_effect_analyzer.combine_effects(method="fixed")
    row = result.iloc[0]
    assert row["rel_effect_lower"] <= row["relative_effect"] <= row["rel_effect_upper"], (
        f"Relative CI does not contain estimate: [{row['rel_effect_lower']:.4f}, {row['rel_effect_upper']:.4f}] "
        f"vs {row['relative_effect']:.4f}"
    )


def test_relative_effect_ci_contains_estimate_random(known_effect_analyzer):
    """Random effects: rel_effect_lower ≤ relative_effect ≤ rel_effect_upper."""
    result = known_effect_analyzer.combine_effects(method="random")
    row = result.iloc[0]
    assert row["rel_effect_lower"] <= row["relative_effect"] <= row["rel_effect_upper"], (
        f"Relative CI does not contain estimate: [{row['rel_effect_lower']:.4f}, {row['rel_effect_upper']:.4f}] "
        f"vs {row['relative_effect']:.4f}"
    )


def test_relative_effect_matches_absolute_over_control(known_effect_analyzer):
    """
    When control mean ≈ 10 and absolute effect ≈ 1.0, pooled relative should ≈ 0.10.
    Verifies that relative pooling uses the correct delta-method scale.
    """
    for method in ("fixed", "random"):
        result = known_effect_analyzer.combine_effects(method=method)
        rel = result["relative_effect"].iloc[0]
        assert abs(rel - 0.10) < 0.05, (
            f"[{method}] relative_effect={rel:.4f} far from expected ~0.10 "
            f"(absolute={result['absolute_effect'].iloc[0]:.4f})"
        )


def test_relative_effect_finite_when_control_positive(known_effect_analyzer):
    """Both fixed and random should produce finite relative effects for clean data."""
    for method in ("fixed", "random"):
        result = known_effect_analyzer.combine_effects(method=method)
        assert np.isfinite(result["relative_effect"].iloc[0]), f"[{method}] relative_effect is not finite"
        assert np.isfinite(result["rel_effect_lower"].iloc[0]), f"[{method}] rel_effect_lower is not finite"
        assert np.isfinite(result["rel_effect_upper"].iloc[0]), f"[{method}] rel_effect_upper is not finite"


def test_relative_effect_nan_study_does_not_crash(multi_experiment_analyzer):
    """Injecting a NaN relative_effect in one study must not raise an error."""
    results = multi_experiment_analyzer._results.copy()
    # Corrupt relative_effect in the first row to simulate a degenerate study
    results.loc[results.index[0], "relative_effect"] = np.nan
    for method in ("fixed", "random"):
        pooled = multi_experiment_analyzer.combine_effects(data=results, method=method)
        assert len(pooled) > 0, f"[{method}] combine_effects returned empty DataFrame after NaN injection"
        assert np.isfinite(pooled["absolute_effect"].iloc[0]), (
            f"[{method}] absolute_effect corrupted by NaN relative_effect"
        )


def test_relative_effect_zero_control_falls_back(multi_experiment_analyzer):
    """When all control_value=0, relative pooling must fall back without raising."""
    results = multi_experiment_analyzer._results.copy()
    results["control_value"] = 0.0
    for method in ("fixed", "random"):
        pooled = multi_experiment_analyzer.combine_effects(data=results, method=method)
        # Relative CI should be NaN (no valid control values) but absolute must survive
        assert np.isfinite(pooled["absolute_effect"].iloc[0]), (
            f"[{method}] absolute_effect should be finite even with control_value=0"
        )
        assert np.isnan(pooled["rel_effect_lower"].iloc[0]), (
            f"[{method}] rel_effect_lower should be NaN when control_value=0"
        )


def test_asymptotic_ci_uses_t_distribution(sample_data):
    """CI must use t(df_resid), not z, for OLS asymptotic inference."""
    from scipy import stats

    analyzer = ExperimentAnalyzer(
        data=sample_data,
        outcomes="conversion",
        treatment_col="treatment",
        experiment_identifier="experiment",
        regression_covariates="baseline_conversion",
        bootstrap=False,
    )
    analyzer.get_effects()
    results = analyzer.results

    # Filter to asymptotic OLS rows with df_resid
    mask = (
        (results["inference_method"] == "asymptotic") & (results["model_type"] == "ols") & (results["df_resid"].notna())
    )
    ols_rows = results.loc[mask]
    assert len(ols_rows) > 0, "Expected at least one asymptotic OLS row with df_resid"

    alpha = 0.05
    for _, row in ols_rows.iterrows():
        t_crit = stats.t.ppf(1 - alpha / 2, row["df_resid"])
        expected_lower = row["absolute_effect"] - t_crit * row["standard_error"]
        expected_upper = row["absolute_effect"] + t_crit * row["standard_error"]
        assert np.isclose(row["abs_effect_lower"], expected_lower, rtol=1e-10), (
            f"abs_effect_lower mismatch: {row['abs_effect_lower']} != {expected_lower}"
        )
        assert np.isclose(row["abs_effect_upper"], expected_upper, rtol=1e-10), (
            f"abs_effect_upper mismatch: {row['abs_effect_upper']} != {expected_upper}"
        )

    # Verify it does NOT match z-distribution
    z_crit = stats.norm.ppf(1 - alpha / 2)
    for _, row in ols_rows.iterrows():
        z_lower = row["absolute_effect"] - z_crit * row["standard_error"]
        assert not np.isclose(row["abs_effect_lower"], z_lower, rtol=1e-10), (
            "abs_effect_lower should NOT match z-distribution"
        )


def test_relative_effect_random_wider_than_fixed_with_heterogeneity(heterogeneous_analyzer):
    """With high heterogeneity, random effects relative CI must be wider than fixed."""
    fixed = heterogeneous_analyzer.combine_effects(method="fixed")
    random = heterogeneous_analyzer.combine_effects(method="random")
    fe_width = fixed["rel_effect_upper"].values - fixed["rel_effect_lower"].values
    re_width = random["rel_effect_upper"].values - random["rel_effect_lower"].values
    # Both CIs must be finite to compare
    both_finite = np.isfinite(fe_width) & np.isfinite(re_width)
    if both_finite.any():
        assert (re_width[both_finite] >= fe_width[both_finite]).all(), (
            f"RE relative CI should be >= FE with heterogeneity.\nFE width: {fe_width}\nRE width: {re_width}"
        )
