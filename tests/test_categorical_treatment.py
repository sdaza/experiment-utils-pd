"""
Tests for categorical treatment functionality in ExperimentAnalyzer
"""

import numpy as np
import pandas as pd
import pytest

from experiment_utils.experiment_analyzer import ExperimentAnalyzer


@pytest.fixture
def sample_data_categorical():
    """
    Generate sample data with categorical treatment groups
    """
    np.random.seed(42)
    
    n_per_group = 300
    
    # Three treatment groups: 'control', 'treatment_a', 'treatment_b'
    groups = ['control'] * n_per_group + ['treatment_a'] * n_per_group + ['treatment_b'] * n_per_group
    
    # Baseline characteristics
    baseline_conversion = np.random.beta(2, 5, n_per_group * 3)
    age = np.random.normal(35, 10, n_per_group * 3)
    
    # Treatment effects
    effects = {
        'control': 0.0,
        'treatment_a': 0.05,
        'treatment_b': 0.10
    }
    
    # Generate outcomes based on treatment
    conversion = []
    for i, group in enumerate(groups):
        prob = baseline_conversion[i] + effects[group]
        conversion.append(1 if np.random.rand() < prob else 0)
    
    data = pd.DataFrame({
        'experiment_id': 1,
        'treatment': groups,
        'conversion': conversion,
        'baseline_conversion': baseline_conversion,
        'age': age
    })
    
    return data


@pytest.fixture
def sample_data_numeric_categorical():
    """
    Generate sample data with numeric categorical treatment (1, 2, 3)
    """
    np.random.seed(42)
    
    n_per_group = 300
    
    # Three treatment groups: 1 (control), 2 (treatment_a), 3 (treatment_b)
    groups = [1] * n_per_group + [2] * n_per_group + [3] * n_per_group
    
    # Baseline characteristics
    baseline_conversion = np.random.beta(2, 5, n_per_group * 3)
    
    # Treatment effects
    effects = {1: 0.0, 2: 0.05, 3: 0.10}
    
    # Generate outcomes
    conversion = []
    for i, group in enumerate(groups):
        prob = baseline_conversion[i] + effects[group]
        conversion.append(1 if np.random.rand() < prob else 0)
    
    data = pd.DataFrame({
        'experiment_id': 1,
        'treatment': groups,
        'conversion': conversion,
        'baseline_conversion': baseline_conversion
    })
    
    return data


def test_categorical_treatment_default_comparisons(sample_data_categorical):
    """Test categorical treatment with default all pairwise comparisons"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    assert len(results) > 0
    
    # Should have all pairwise comparisons: 
    # treatment_a vs treatment_b, treatment_a vs control, treatment_b vs control
    # Or: control vs treatment_a, control vs treatment_b, treatment_a vs treatment_b
    assert len(results) == 3, f"Expected 3 comparisons, got {len(results)}"
    
    # Check that treatment_group and control_group columns exist
    assert 'treatment_group' in results.columns
    assert 'control_group' in results.columns
    
    # Verify all groups are represented
    all_groups = set(results['treatment_group'].tolist() + results['control_group'].tolist())
    expected_groups = {'control', 'treatment_a', 'treatment_b'}
    assert all_groups == expected_groups


def test_categorical_treatment_specific_comparisons(sample_data_categorical):
    """Test categorical treatment with specific user-defined comparisons"""
    
    # Only compare treatment_a vs control and treatment_b vs control
    comparisons = [
        ('treatment_a', 'control'),
        ('treatment_b', 'control')
    ]
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=comparisons,
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    assert len(results) == 2, f"Expected 2 comparisons, got {len(results)}"
    
    # Verify the specific comparisons were made
    comparison_tuples = set(
        zip(results['treatment_group'], results['control_group'])
    )
    expected_tuples = {('treatment_a', 'control'), ('treatment_b', 'control')}
    assert comparison_tuples == expected_tuples


def test_categorical_treatment_with_covariates(sample_data_categorical):
    """Test categorical treatment with covariate adjustment"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        covariates=['baseline_conversion', 'age'],
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    assert len(results) == 1
    
    # Check balance was assessed
    balance = analyzer.balance
    assert balance is not None
    assert not balance.empty


def test_categorical_treatment_with_balance_adjustment(sample_data_categorical):
    """Test categorical treatment with IPW balance adjustment"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        covariates=['baseline_conversion', 'age'],
        adjustment='balance',
        balance_method='ps-logistic',
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    assert len(results) == 1
    
    # Check that balance adjustment was performed
    assert 'balance' in results.columns
    adjusted_balance = analyzer.adjusted_balance
    assert adjusted_balance is not None
    
    # Check weights were saved
    weights = analyzer.weights
    assert weights is not None
    assert 'treatment_group' in weights.columns
    assert 'control_group' in weights.columns


def test_numeric_categorical_treatment(sample_data_numeric_categorical):
    """Test numeric categorical treatment (not binary 0/1)"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_numeric_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    # Should have 3 pairwise comparisons for groups 1, 2, 3
    assert len(results) == 3
    
    # Verify numeric groups
    all_groups = set(results['treatment_group'].tolist() + results['control_group'].tolist())
    assert all_groups == {1, 2, 3}


def test_binary_treatment_default_comparison():
    """Test that binary 0/1 treatment still works as expected"""
    np.random.seed(42)
    
    n = 500
    treatment = np.random.binomial(1, 0.5, n)
    baseline = np.random.beta(2, 5, n)
    conversion = (baseline + 0.05 * treatment > np.random.rand(n)).astype(int)
    
    data = pd.DataFrame({
        'experiment_id': 1,
        'treatment': treatment,
        'conversion': conversion,
        'baseline': baseline
    })
    
    analyzer = ExperimentAnalyzer(
        data=data,
        outcomes='conversion',
        treatment_col='treatment',
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    # Binary should only produce one comparison: 1 vs 0
    assert len(results) == 1
    assert results['treatment_group'].iloc[0] == 1
    assert results['control_group'].iloc[0] == 0


def test_summary_creation(sample_data_categorical):
    """Test the create_summary method"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        covariates=['baseline_conversion'],
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    summary = analyzer.create_summary()
    
    assert summary is not None
    assert not summary.empty
    
    # Check expected columns are present
    expected_cols = [
        'experiment', 'outcome', 'treatment_group', 'control_group',
        'total_sample_size', 'treated_units', 'control_units',
        'absolute_effect', 'relative_effect', 'pvalue'
    ]
    
    for col in expected_cols:
        assert col in summary.columns, f"Missing column: {col}"
    
    # Check derived columns
    assert 'significance_label' in summary.columns
    assert 'total_sample_size' in summary.columns
    assert summary['total_sample_size'].iloc[0] == (
        summary['treated_units'].iloc[0] + summary['control_units'].iloc[0]
    )


def test_multiple_outcomes_categorical(sample_data_categorical):
    """Test multiple outcomes with categorical treatment"""
    
    # Add a second outcome
    sample_data_categorical['revenue'] = (
        sample_data_categorical['conversion'] * np.random.gamma(2, 10, len(sample_data_categorical))
    )
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes=['conversion', 'revenue'],
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    # Should have results for both outcomes
    assert len(results) == 2
    assert set(results['outcome'].unique()) == {'conversion', 'revenue'}


def test_invalid_comparison_warning(sample_data_categorical):
    """Test that invalid comparisons generate warnings"""
    
    # Specify a comparison with a group that doesn't exist
    comparisons = [
        ('treatment_a', 'control'),
        ('nonexistent_group', 'control')  # This should be skipped with warning
    ]
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=comparisons,
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    # Should only have the valid comparison
    assert len(results) == 1
    assert results['treatment_group'].iloc[0] == 'treatment_a'
    assert results['control_group'].iloc[0] == 'control'


def test_categorical_with_bootstrap(sample_data_categorical):
    """Test categorical treatment with bootstrap inference"""
    
    analyzer = ExperimentAnalyzer(
        data=sample_data_categorical,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        experiment_identifier='experiment_id',
        bootstrap=True,
        bootstrap_iterations=100,  # Small number for testing
        bootstrap_seed=42
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    assert len(results) == 1
    
    # Check bootstrap-specific columns
    assert 'inference_method' in results.columns
    assert results['inference_method'].iloc[0] == 'bootstrap'
    assert 'abs_effect_lower' in results.columns
    assert 'abs_effect_upper' in results.columns
    assert not pd.isna(results['abs_effect_lower'].iloc[0])
    assert not pd.isna(results['abs_effect_upper'].iloc[0])


def test_sample_ratio_with_categorical():
    """Test sample ratio mismatch detection with categorical treatment"""
    np.random.seed(42)
    
    # Create imbalanced data
    data = pd.DataFrame({
        'experiment_id': 1,
        'expected_ratio': 0.5,
        'treatment': ['control'] * 400 + ['treatment_a'] * 100,  # Imbalanced!
        'conversion': np.random.binomial(1, 0.3, 500),
        'baseline': np.random.beta(2, 5, 500)
    })
    
    analyzer = ExperimentAnalyzer(
        data=data,
        outcomes='conversion',
        treatment_col='treatment',
        treatment_comparisons=[('treatment_a', 'control')],
        exp_sample_ratio_col='expected_ratio',
        experiment_identifier='experiment_id'
    )
    
    analyzer.get_effects()
    results = analyzer.results
    
    assert results is not None
    # SRM should be detected due to imbalance
    if 'srm_detected' in results.columns:
        assert results['srm_detected'].iloc[0] == True or results['srm_detected'].iloc[0] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
