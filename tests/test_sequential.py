import numpy as np
import pytest

from experiment_utils.power_sim import PowerSim


def test_msprt_boundary_standard():
    # log(1 / 0.05) ≈ 2.996
    result = PowerSim.msprt_boundary(alpha=0.05)
    assert abs(result - 2.996) < 0.001


def test_msprt_boundary_alpha_01():
    # log(1 / 0.01) ≈ 4.605
    result = PowerSim.msprt_boundary(alpha=0.01)
    assert abs(result - 4.605) < 0.001


def test_msprt_boundary_invalid():
    with pytest.raises(ValueError):
        PowerSim.msprt_boundary(alpha=0.0)
    with pytest.raises(ValueError):
        PowerSim.msprt_boundary(alpha=1.0)


def test_msprt_llr_null_effect():
    # z=0 → LLR should be negative (evidence for null)
    result = PowerSim.msprt_llr(z=0.0, se=0.01, tau=0.02)
    assert result < 0


def test_msprt_llr_large_z():
    # Large z → LLR should be large positive
    result = PowerSim.msprt_llr(z=5.0, se=0.01, tau=0.02)
    assert result > 2.0


def test_msprt_llr_invalid_se():
    with pytest.raises(ValueError):
        PowerSim.msprt_llr(z=2.0, se=0.0, tau=0.02)


def test_msprt_llr_invalid_tau():
    with pytest.raises(ValueError):
        PowerSim.msprt_llr(z=2.0, se=0.01, tau=0.0)


def test_msprt_should_stop_false_for_null():
    # z=0 should never trigger stopping
    result = PowerSim.msprt_should_stop(z=0.0, se=0.01, tau=0.02, alpha=0.05)
    assert result is False


def test_msprt_should_stop_true_for_strong_signal():
    # Very large z should trigger stopping
    result = PowerSim.msprt_should_stop(z=10.0, se=0.01, tau=0.02, alpha=0.05)
    assert result is True


def test_msprt_should_stop_boundary_case():
    # LLR exactly at boundary → stop
    alpha = 0.05
    boundary = PowerSim.msprt_boundary(alpha)
    # Construct z, se, tau such that LLR is slightly above boundary
    # From LLR formula: with se=tau, LLR = 0.5 * [log(0.5) + z²*0.5]
    # LLR = boundary when z² = 2*(boundary - 0.5*log(0.5)) / 0.5
    # = 4*boundary + 2*log(2)
    se = 0.01
    tau = 0.01
    z_critical = np.sqrt(4 * boundary + 2 * np.log(2)) + 0.01  # just over boundary
    result = PowerSim.msprt_should_stop(z=z_critical, se=se, tau=tau, alpha=alpha)
    assert result is True
