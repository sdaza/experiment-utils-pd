import numpy as np
import pytest
from scipy.stats import norm

from experiment_utils.utils import winners_curse_estimate


def test_returns_documented_keys():
    out = winners_curse_estimate(effect=5.0, standard_error=2.0, alpha=0.05)
    assert set(out) == {"corrected", "ci_lower", "ci_upper", "observed_z", "shrinkage"}


def test_no_shrink_far_above_threshold():
    # z = 10 >> 1.96: correction should be negligible
    out = winners_curse_estimate(effect=10.0, standard_error=1.0, alpha=0.05)
    assert abs(out["corrected"] - 10.0) < 0.05
    assert out["shrinkage"] > 0.99


def test_strong_shrink_near_threshold():
    # just-significant: z just above 1.96 -> heavy shrink toward 0
    s = 1.0
    b = norm.ppf(0.975) * s + 1e-3
    out = winners_curse_estimate(effect=b, standard_error=s, alpha=0.05)
    assert abs(out["corrected"]) < abs(b)
    assert out["shrinkage"] < 0.6


def test_sign_symmetry():
    pos = winners_curse_estimate(effect=4.0, standard_error=1.5, alpha=0.05)
    neg = winners_curse_estimate(effect=-4.0, standard_error=1.5, alpha=0.05)
    assert np.isclose(pos["corrected"], -neg["corrected"], atol=1e-6)
    assert np.isclose(pos["ci_lower"], -neg["ci_upper"], atol=1e-6)


def test_monotone_in_effect():
    s = 1.0
    vals = [winners_curse_estimate(b, s, alpha=0.05)["corrected"] for b in [2.1, 3.0, 4.0, 6.0]]
    assert all(x < y for x, y in zip(vals, vals[1:], strict=False))


def test_ci_ordering():
    out = winners_curse_estimate(effect=4.0, standard_error=1.5, alpha=0.05, ci=0.95)
    assert out["ci_lower"] < out["corrected"] < out["ci_upper"]


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_raises_on_bad_se(bad):
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=bad)


def test_raises_on_bad_alpha_ci():
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, alpha=1.5)
    with pytest.raises(ValueError):
        winners_curse_estimate(effect=4.0, standard_error=1.0, ci=0.0)


def test_warns_below_threshold():
    with pytest.warns(RuntimeWarning):
        winners_curse_estimate(effect=1.0, standard_error=1.0, alpha=0.05)  # z=1.0 < 1.96


def _selected_draws(beta_true, s, alpha, n, seed):
    rng = np.random.default_rng(seed)
    c = norm.ppf(1 - alpha / 2) * s
    draws = rng.normal(beta_true, s, size=n)
    return draws[np.abs(draws) >= c]


def test_sim_median_unbiased():
    # MUE solves F_beta(b)=0.5; F_beta(B)~U(0,1) => median of corrected ~ beta_true.
    beta_true, s, alpha = 1.0, 1.0, 0.05
    sel = _selected_draws(beta_true, s, alpha, n=400_000, seed=7)[:4000]
    corrected = np.array([winners_curse_estimate(b, s, alpha)["corrected"] for b in sel])
    assert abs(np.median(corrected) - beta_true) < 0.12


def test_sim_conditional_ci_coverage():
    # P(gamma/2 <= F_beta_true(B) <= 1-gamma/2) = 0.95 exactly in theory.
    beta_true, s, alpha = 1.0, 1.0, 0.05
    sel = _selected_draws(beta_true, s, alpha, n=400_000, seed=11)[:3000]
    cov = np.mean(
        [
            (lambda o: o["ci_lower"] <= beta_true <= o["ci_upper"])(winners_curse_estimate(b, s, alpha, ci=0.95))
            for b in sel
        ]
    )
    assert 0.93 <= cov <= 0.97
