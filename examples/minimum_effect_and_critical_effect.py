# %% [markdown]
# # Beyond Type S/M: critical effects, minimum-effect tests, and honest retrodesign
#
# Lakens et al. (2026, "Rethinking Type S and Type M Errors") argue that Type S/M
# numbers are diagnostics, not inference, and recommend three alternatives that
# this package implements:
#
# 1. **Critical effect size** (Perugini et al. 2025): the smallest effect that
#    could have reached significance — `critical_effect` in `get_effects()`,
#    `critical_effect_mcp` after multiple-comparison correction.
# 2. **Minimum-effect test** (superiority by margin): reject all effects too
#    small to matter instead of an effect of exactly zero —
#    `test_equivalence(test_type="minimum_effect")`.
# 3. **Bias-adjusted estimates** from a truncated-distribution model (the same
#    family as Hedges 1984 / Taylor & Muller 1996 / BUCSS):
#    `winners_curse_summary()`, which `calculate_retrodesign()` now also uses
#    as its default assumed true effect.

# %%
import numpy as np
import pandas as pd

from experiment_utils import ExperimentAnalyzer

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# Three binary outcomes at a 10% baseline. True lifts: +1.2pp, +0.8pp, and 0.
rng = np.random.default_rng(42)
n = 8_000
true_lifts = {"activated": 0.012, "retained": 0.008, "upsold": 0.0}
treatment = np.array([0] * n + [1] * n)
data = pd.DataFrame({"treatment": treatment, "experiment": "exp1"})
for outcome, lift in true_lifts.items():
    p = np.where(treatment == 1, 0.10 + lift, 0.10)
    data[outcome] = rng.binomial(1, p)

analyzer = ExperimentAnalyzer(
    data=data,
    outcomes=list(true_lifts),
    treatment_col="treatment",
    experiment_identifier=["experiment"],
)
analyzer.get_effects()

# %% [markdown]
# ## 1. The critical effect size
#
# `critical_effect` is the smallest absolute effect this design could have
# declared significant. Everything below it is censored when you select on
# significance — so if your SESOI is smaller than the critical effect, the
# experiment cannot even see the effects you care about. After a
# multiple-comparison correction the bar rises (`critical_effect_mcp`).

# %%
analyzer.adjust_pvalues(method="bonferroni")
print(
    analyzer.results[
        [
            "outcome",
            "absolute_effect",
            "critical_effect",
            "critical_effect_mcp",
            "alpha_mcp",
            "stat_significance_mcp",
        ]
    ]
)

# %% [markdown]
# ## 2. Minimum-effect test against a SESOI
#
# Suppose lifts below 0.5pp are not worth shipping. Instead of testing against
# zero, reject H0: effect <= 0.005. `eq_conclusion` separates "meaningful"
# (exceeds the SESOI) from "significant_below_margin" (nonzero, but not shown
# to matter) and "inconclusive".

# %%
analyzer.test_equivalence(
    test_type="minimum_effect",
    absolute_bound=0.005,
    direction="higher_is_better",
)
print(analyzer.results[["outcome", "absolute_effect", "pvalue", "eq_pvalue", "eq_conclusion"]])

# %% [markdown]
# ## 3. Retrodesign with an honest default
#
# With no `true_effect`, `calculate_retrodesign()` de-biases each significant
# winner via the truncated-normal winner's-curse correction and uses that as
# the assumed truth — the raw observed effect is inflated by selection and
# would overstate power. `retrodesign_alpha` shows each row was simulated at
# the threshold that actually selected it (here the Bonferroni-adjusted alpha).

# %%
retro = analyzer.calculate_retrodesign(nsim=2000, seed=7)
if retro.empty:
    print("No significant effects to analyze.")
else:
    print(
        retro[
            [
                "outcome",
                "absolute_effect",
                "true_effect",
                "power",
                "type_s_error",
                "type_m_error",
                "retrodesign_alpha",
            ]
        ]
    )

# %% [markdown]
# Reading the output: `true_effect < absolute_effect` (the correction undoes the
# winner's curse), and `type_m_error` tells you how inflated significant effects
# from this *design* are on average — it is not a divisor for any single
# estimate; the corrected point estimates live in `winners_curse_summary()`.

# %%
print(analyzer.winners_curse_summary(method="conditional"))

# %%
