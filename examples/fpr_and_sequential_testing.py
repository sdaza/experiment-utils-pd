"""False Positive Risk and sequential testing — worked example.

Demonstrates the Kohavi & Chen (2024) False Positive Risk framework and the
always-valid sequential-testing utilities (mSPRT, Goldilocks alpha-spending).

The company in this example runs experiments at alpha = 0.10.

Run with:

    python examples/fpr_and_sequential_testing.py
"""

import os
import tempfile

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
os.environ["XDG_CACHE_HOME"] = tempfile.gettempdir()

from experiment_utils import (
    ExperimentAnalyzer,
    PowerSim,
    estimate_true_success_rate,
    false_positive_risk,
)

ALPHA = 0.10  # the company's significance level


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# 1. False Positive Risk: "is my significant result actually real?"
# A p-value < alpha is NOT the probability the result is a fluke — that depends
# on how often your ideas truly work (the base rate / prior success rate).
# =============================================================================
section("1. False Positive Risk vs. the prior success rate (alpha = 0.10, power = 0.80)")

for prior in [0.05, 0.12, 0.30, 0.50]:
    fpr = false_positive_risk(alpha=ALPHA, power=0.80, prior_success_rate=prior)
    print(f"  prior success rate {prior:>4.0%}  ->  FPR = {fpr:6.1%}  of significant wins are noise")

print("\n  A stricter alpha lowers FPR (prior fixed at 12%):")
for a in [0.10, 0.05, 0.01]:
    fpr = false_positive_risk(alpha=a, power=0.80, prior_success_rate=0.12)
    print(f"    alpha {a:<5}  ->  FPR = {fpr:6.1%}")

# Invert it: recover the TRUE success rate hidden behind an observed win rate.
section("   Recovering the true success rate from an observed win rate")
for win_rate in [0.08, 0.12, 0.20]:
    true_sr = estimate_true_success_rate(win_rate=win_rate, alpha=ALPHA, power=0.80)
    print(f"  observed win rate {win_rate:>4.0%}  ->  true success rate ~ {true_sr:5.1%}")

# =============================================================================
# 2. fpr_summary(): FPR for your own analyzer results
# =============================================================================
section("2. ExperimentAnalyzer.fpr_summary() — FPR on real results")

rng = np.random.default_rng(42)
n = 4000
treatment = rng.binomial(1, 0.5, n)
outcome = rng.binomial(1, 0.10 + 0.02 * treatment, n)  # true +2pp lift

ea = ExperimentAnalyzer(
    data=pd.DataFrame({"treatment": treatment, "outcome": outcome}),
    outcomes=["outcome"],
    treatment_col="treatment",
    alpha=ALPHA,
)
ea.get_effects()
print(ea.results[["outcome", "absolute_effect", "standard_error", "pvalue", "stat_significance"]].round(4))
print("\n  FPR summary (prior = historical 12% win rate):")
print(ea.fpr_summary(prior_success_rate=0.12).to_string(index=False))

# =============================================================================
# 3. aggregate_effects(prior_success_rate=...): portfolio FPR column
# =============================================================================
section("3. aggregate_effects() — portfolio FPR across experiments")

frames = []
for exp_id in [1, 2, 3]:
    nn = 1500
    tt = rng.binomial(1, 0.5, nn)
    yy = rng.binomial(1, 0.10 + 0.02 * tt, nn)
    frames.append(pd.DataFrame({"experiment": exp_id, "treatment": tt, "outcome": yy}))

ea_multi = ExperimentAnalyzer(
    data=pd.concat(frames, ignore_index=True),
    outcomes=["outcome"],
    treatment_col="treatment",
    experiment_identifier=["experiment"],
    alpha=ALPHA,
)
ea_multi.get_effects()
print(ea_multi.aggregate_effects(prior_success_rate=0.12).to_string(index=False))

# =============================================================================
# 4. mSPRT: always-valid sequential testing — peek as often as you like
# Stop when the log-likelihood ratio (LLR) exceeds the boundary log(1/alpha).
# tau = prior SD on the effect ~ the magnitude you expect / are powered for.
# =============================================================================
section("4. mSPRT — always-valid peeking (boundary = log(1/alpha))")

for a in [0.10, 0.05, 0.01]:
    print(f"  alpha {a:<5}  ->  boundary = {PowerSim.msprt_boundary(a):.3f}")


def two_proportion_effect_se(control, treat):
    """Difference in proportions and its standard error."""
    p_c, p_t = control.mean(), treat.mean()
    n_c, n_t = len(control), len(treat)
    se = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return p_t - p_c, se


WEEKLY_PER_ARM = 800
TAU = 0.02  # expect ~2pp lift

print("\n  Weekly peeking under a TRUE +2pp lift (alpha = 0.10):")
rng_real = np.random.default_rng(1)
ctrl_batches, treat_batches = [], []
for week in range(1, 13):
    ctrl_batches.append(rng_real.binomial(1, 0.10, WEEKLY_PER_ARM))
    treat_batches.append(rng_real.binomial(1, 0.12, WEEKLY_PER_ARM))
    control = np.concatenate(ctrl_batches)
    treat = np.concatenate(treat_batches)
    effect, se = two_proportion_effect_se(control, treat)
    z = effect / se
    llr = PowerSim.msprt_llr(z=z, se=se, tau=TAU)
    stop = PowerSim.msprt_should_stop(z=z, se=se, tau=TAU, alpha=ALPHA)
    print(f"    week {week:>2}  n/arm={len(control):>5}  z={z:5.2f}  LLR={llr:6.2f}  stop={stop}")
    if stop:
        print(f"    --> conclusive at week {week}; safe to stop early.")
        break

print("\n  Weekly peeking under the NULL (no effect): mSPRT protects you from noise:")
rng_null = np.random.default_rng(2)
ctrl_batches, treat_batches = [], []
stopped = False
for week in range(1, 13):
    ctrl_batches.append(rng_null.binomial(1, 0.10, WEEKLY_PER_ARM))
    treat_batches.append(rng_null.binomial(1, 0.10, WEEKLY_PER_ARM))  # no lift
    control = np.concatenate(ctrl_batches)
    treat = np.concatenate(treat_batches)
    effect, se = two_proportion_effect_se(control, treat)
    if PowerSim.msprt_should_stop(z=effect / se, se=se, tau=TAU, alpha=ALPHA):
        print(f"    stopped (false positive) at week {week}")
        stopped = True
        break
if not stopped:
    print("    12 weeks of peeking, never crossed the boundary — type-I error controlled.")

# =============================================================================
# 5. Goldilocks: simpler two-stage plan (one early look, one final look)
# =============================================================================
section("5. goldilocks_alpha_spending() — two-stage design at alpha_total = 0.10")

for method in ["goldilocks", "obrien_fleming", "pocock"]:
    a1, a2 = PowerSim.goldilocks_alpha_spending(alpha_total=ALPHA, method=method)
    print(f"  {method:>14}:  stop early if p < {a1:.3f}   |   final look if p < {a2:.3f}")

# =============================================================================
# 6. msprt_summary(): per-comparison mSPRT verdict straight from get_effects()
# Provide tau (absolute effect scale) OR tau_rel (fraction of control mean).
# =============================================================================
section("6. ExperimentAnalyzer.msprt_summary() — verdict per comparison")

print("  Absolute prior SD (tau = 0.02):")
print(ea.msprt_summary(tau=0.02).to_string(index=False))
print("\n  Relative prior SD (tau_rel = 0.10  ->  ~10% lift over the control mean):")
print(ea.msprt_summary(tau_rel=0.10).to_string(index=False))

print("\nDone.")
