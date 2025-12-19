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
            target_power=0.80, baseline=[0.10], effect=[0.02], min_sample_size=100, max_sample_size=10000
        )

        # Check that result is a DataFrame with expected columns
        assert isinstance(result, pd.DataFrame)
        assert "comparison" in result.columns
        assert "total_sample_size" in result.columns
        assert "achieved_power" in result.columns
        assert "target_power" in result.columns

        # Check that we have results for all comparisons
        assert len(result) == len(p.comparisons)

        # Check that achieved power is close to target
        assert all(result["achieved_power"] >= result["target_power"] - 0.05)

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
            max_sample_size=10000,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "allocation_ratio" in result.columns

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
        assert len(result) == len(p.comparisons)
        assert all(result["achieved_power"] >= 0.75)

    except Exception as e:
        pytest.fail(f"find_sample_size with average metric raised an exception: {e}")
