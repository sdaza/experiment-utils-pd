import pytest

from experiment_utils import false_positive_risk as fpr_from_init
from experiment_utils.utils import estimate_true_success_rate, false_positive_risk


def test_fpr_known_values():
    # alpha=0.10, power=0.80, prior_success_rate=0.12
    # FPR = (0.10*0.88) / (0.10*0.88 + 0.80*0.12) = 0.088 / 0.184 = 0.4783
    result = false_positive_risk(alpha=0.10, power=0.80, prior_success_rate=0.12)
    assert abs(result - 0.478) < 0.01


def test_fpr_known_values_low_alpha():
    # alpha=0.05, power=0.80, prior_success_rate=0.15
    # FPR = (0.05*0.85) / (0.05*0.85 + 0.80*0.15) = 0.0425 / 0.1625 = 0.2615
    result = false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.15)
    assert abs(result - 0.262) < 0.01


def test_fpr_high_success_rate():
    # At 50% success rate, FPR should be lower
    result = false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.50)
    assert result < 0.10


def test_fpr_invalid_prior():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=0.0)
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.80, prior_success_rate=1.0)


def test_fpr_invalid_alpha():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.0, power=0.80, prior_success_rate=0.15)


def test_fpr_invalid_power():
    with pytest.raises(ValueError):
        false_positive_risk(alpha=0.05, power=0.0, prior_success_rate=0.15)


def test_estimate_true_success_rate_known():
    # win_rate=0.12, alpha=0.10, power=0.80
    # pi = (0.80-0.12)/(0.80-0.10) = 0.68/0.70 = 0.9714 -> true = 1 - pi = 0.0286
    result = estimate_true_success_rate(win_rate=0.12, alpha=0.10, power=0.80)
    assert abs(result - 0.029) < 0.01


def test_estimate_true_success_rate_standard():
    # win_rate=0.15, alpha=0.05, power=0.80
    # pi = (0.80-0.15)/(0.80-0.05) = 0.65/0.75 = 0.8667 -> true = 1 - pi = 0.1333
    result = estimate_true_success_rate(win_rate=0.15, alpha=0.05, power=0.80)
    assert abs(result - 0.14) < 0.02


def test_estimate_true_success_rate_clamps_to_zero():
    # win_rate < alpha -> pi > 1 -> clamp to 0
    result = estimate_true_success_rate(win_rate=0.03, alpha=0.05, power=0.80)
    assert result == 0.0


def test_estimate_true_success_rate_power_equals_alpha():
    with pytest.raises(ValueError):
        estimate_true_success_rate(win_rate=0.15, alpha=0.80, power=0.80)


def test_exported_from_init():
    # Verify the function is importable from the top-level package
    result = fpr_from_init(alpha=0.05, power=0.80, prior_success_rate=0.15)
    assert 0 < result < 1
