import numpy as np
import pandas as pd
import pytest

from experiment_utils.power_sim import PowerSim


def test_power_estimation():
    """Test power estimation"""
    p = PowerSim(
        metric="proportion", relative_effect=False, variants=1, nsim=1000, alpha=0.05, alternative="two-tailed"
    )
    try:
        p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_plot_power():
    """Test plot power"""
    p = PowerSim(
        metric="proportion", relative_effect=False, variants=2, alternative="two-tailed", nsim=100, correction="holm"
    )
    try:
        p.grid_sim_power(
            baseline_rates=[[0.33]],
            effects=[[0.01, 0.03], [0.03, 0.05], [0.03, 0.07]],
            sample_sizes=[[1000], [5000], [9000]],
            threads=16,
            plot=False,
        )
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_power_from_data():
    """Test power estimation from data"""
    p = PowerSim(metric="proportion", relative_effect=False, variants=1, nsim=100, alpha=0.05, alternative="two-tailed")
    np.random.seed(42)
    n = 6000
    df = pd.DataFrame({"converted": np.random.binomial(1, 0.15, n)})
    try:
        p.get_power_from_data(df=df, metric_col="converted", effect=[0.03], sample_size=[300])
        assert True
    except Exception as e:
        pytest.fail(f" raised an exception: {e}")


def test_find_sample_size():
    """Test find_sample_size method"""
    # Test with proportion metric
    p = PowerSim(metric="proportion", relative_effect=False, variants=1, nsim=500, alpha=0.05, alternative="two-tailed")
    try:
        result = p.find_sample_size(
            target_power=0.80, baseline=[0.10], effect=[0.02], min_sample_size=100, max_sample_size=20000
        )

        # Check that result is a DataFrame with expected columns
        assert isinstance(result, pd.DataFrame)
        assert "total_sample_size" in result.columns
        assert "achieved_power_by_comparison" in result.columns
        assert "target_power_by_comparison" in result.columns

        # For single comparison, should have just 1 row
        assert len(result) == 1

        # Check that dictionaries contain the comparison
        achieved = result.iloc[0]["achieved_power_by_comparison"]
        target = result.iloc[0]["target_power_by_comparison"]
        assert isinstance(achieved, dict)
        assert isinstance(target, dict)
        assert len(achieved) > 0
        
        # Check that achieved power is close to target
        for comp_str in achieved:
            assert achieved[comp_str] >= target[comp_str] - 0.05

        # Check that sample_sizes_by_group is present as a dictionary
        assert "sample_sizes_by_group" in result.columns
        assert isinstance(result.iloc[0]["sample_sizes_by_group"], dict)
        assert "control" in result.iloc[0]["sample_sizes_by_group"]
        assert "variant_1" in result.iloc[0]["sample_sizes_by_group"]

    except Exception as e:
        pytest.fail(f"find_sample_size raised an exception: {e}")


def test_find_sample_size_custom_allocation():
    """Test find_sample_size with custom allocation ratio"""
    p = PowerSim(metric="proportion", relative_effect=False, variants=1, nsim=500, alpha=0.05, alternative="two-tailed")
    try:
        # Test with 30% control, 70% treatment allocation
        result = p.find_sample_size(
            target_power=0.80,
            baseline=[0.10],
            effect=[0.02],
            allocation_ratio=[0.3, 0.7],
            min_sample_size=100,
            max_sample_size=20000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Single row with all info
        assert "sample_sizes_by_group" in result.columns
        
        # Check allocation reflects the requested ratio
        sample_dict = result.iloc[0]["sample_sizes_by_group"]
        control_n = sample_dict["control"]
        variant_n = sample_dict["variant_1"]
        total_n = control_n + variant_n
        assert abs(control_n / total_n - 0.3) < 0.02  # Within 2% of requested

    except Exception as e:
        pytest.fail(f"find_sample_size with custom allocation raised an exception: {e}")


def test_find_sample_size_average_metric():
    """Test find_sample_size with average metric"""
    p = PowerSim(metric="average", relative_effect=False, variants=1, nsim=500, alpha=0.05, alternative="two-tailed")
    try:
        result = p.find_sample_size(
            target_power=0.80,
            baseline=[10.0],
            effect=[1.5],
            standard_deviation=[3.0],
            min_sample_size=50,
            max_sample_size=5000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # Single row with all info
        assert "achieved_power_by_comparison" in result.columns
        assert "sample_sizes_by_group" in result.columns

    except Exception as e:
        pytest.fail(f"find_sample_size with average metric raised an exception: {e}")


def test_find_sample_size_multiple_variants():
    """Test find_sample_size with multiple variants and different effects"""
    p = PowerSim(
        metric="proportion",
        variants=3,
        comparisons=[(0, 1), (0, 2), (0, 3)],
        correction="bonferroni",
        nsim=500,
        alpha=0.05,
        alternative="two-tailed",
    )
    try:
        # Test 1: Same power for all comparisons (default behavior)
        result = p.find_sample_size(
            target_power=0.80,
            baseline=[0.10],
            effect=[0.05, 0.03, 0.07],  # Different effects for each variant
            min_sample_size=300,
            max_sample_size=10000,
        )

        assert isinstance(result, pd.DataFrame)
        # Should have single row with dictionaries
        assert len(result) == 1
        
        # Check sample_sizes_by_group column exists
        assert "sample_sizes_by_group" in result.columns
        
        # Check it's a dictionary with all groups
        sample_dict = result.iloc[0]["sample_sizes_by_group"]
        assert isinstance(sample_dict, dict)
        assert "control" in sample_dict
        assert "variant_1" in sample_dict
        assert "variant_2" in sample_dict
        assert "variant_3" in sample_dict

        # Check that we have power dictionaries
        assert "target_power_by_comparison" in result.columns
        assert "achieved_power_by_comparison" in result.columns
        
        target_powers = result.iloc[0]["target_power_by_comparison"]
        achieved_powers = result.iloc[0]["achieved_power_by_comparison"]
        assert isinstance(target_powers, dict)
        assert isinstance(achieved_powers, dict)
        assert len(achieved_powers) == 3  # Three comparisons

        # The limiting comparison should be marked
        assert "limiting_comparison" in result.columns

        # All comparisons should achieve at least the target power
        for comp_str in achieved_powers:
            assert achieved_powers[comp_str] >= 0.75

        # Test 2: Different power targets per comparison
        result2 = p.find_sample_size(
            target_power={(0, 1): 0.90, (0, 2): 0.80, (0, 3): 0.70},  # Different targets
            baseline=[0.10],
            effect=[0.05, 0.03, 0.07],
            min_sample_size=300,
            max_sample_size=10000,
        )

        assert isinstance(result2, pd.DataFrame)
        assert len(result2) == 1  # Single row
        
        # Check that target powers are different
        target_powers = result2.iloc[0]["target_power_by_comparison"]
        assert len(set(target_powers.values())) > 1  # Should have different values
        assert 0.90 in target_powers.values()
        assert 0.80 in target_powers.values()
        assert 0.70 in target_powers.values()

        # Test 3: Only power specific comparisons
        result3 = p.find_sample_size(
            target_power=0.80,
            baseline=[0.10],
            effect=[0.05, 0.03, 0.07],
            target_comparisons=[(0, 1), (0, 2)],  # Only first two
            min_sample_size=300,
            max_sample_size=10000,
        )

        assert isinstance(result3, pd.DataFrame)
        assert len(result3) == 1  # Single row
        
        # Should only have 2 comparisons in dictionaries
        achieved_powers = result3.iloc[0]["achieved_power_by_comparison"]
        assert len(achieved_powers) == 2
        
        # Test 4: Power criteria "any"
        result4 = p.find_sample_size(
            target_power=0.80,
            baseline=[0.10],
            effect=[0.05, 0.03, 0.07],
            power_criteria="any",  # At least one comparison
            min_sample_size=300,
            max_sample_size=10000,
        )

        assert isinstance(result4, pd.DataFrame)
        assert len(result4) == 1
        assert result4.iloc[0]["power_criteria"] == "any"
        
        # With "any", sample size should be smaller than or equal to with "all"
        assert result4.iloc[0]["total_sample_size"] <= result.iloc[0]["total_sample_size"]

    except Exception as e:
        pytest.fail(f"find_sample_size with multiple variants raised an exception: {e}")
