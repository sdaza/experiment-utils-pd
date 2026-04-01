# TOST Equivalence Testing Design

**Date:** 2026-04-01
**Status:** Approved
**Scope:** Add TOST equivalence testing to `ExperimentAnalyzer`, following Daniel Lakens' practices

---

## Overview

Replace the existing `test_non_inferiority()` method with a unified `test_equivalence()` method that handles equivalence (two-sided TOST), non-inferiority (one-sided lower), and non-superiority (one-sided upper) testing. Add a `plot_equivalence()` visualization and `power_tost()` for sample size planning.

This follows Daniel Lakens' framework where equivalence and non-inferiority are part of the same inferential family, distinguished only by whether one or both bounds are tested.

## Breaking Change

`test_non_inferiority()` is **removed entirely**. Users migrate to:

```python
# Old
analyzer.test_non_inferiority(absolute_margin=0.5, direction="higher_is_better")

# New
analyzer.test_equivalence(test_type="non_inferiority", absolute_bound=0.5, direction="higher_is_better")
```

---

## 1. Unified `test_equivalence()` Method

### Signature

```python
def test_equivalence(
    self,
    test_type: str = "equivalence",        # "equivalence" | "non_inferiority" | "non_superiority"
    absolute_bound: float | None = None,    # raw units (e.g., 0.5)
    relative_bound: float | None = None,    # fraction (e.g., 0.10 = 10%)
    cohens_d_bound: float | None = None,    # standardized effect size (e.g., 0.3)
    alpha: float = 0.05,
    direction: str = "higher_is_better",    # only used for non_inferiority / non_superiority
) -> None:
```

### Location

`experiment_utils/experiment_analyzer.py` — replaces `test_non_inferiority()` at its current location (line ~1955).

### Precondition

`get_effects()` must have been called first. Raises error otherwise.

### Bound Specification

Exactly one of `absolute_bound`, `relative_bound`, or `cohens_d_bound` must be provided. All must be > 0.

**Bound resolution:**

- `absolute_bound` -> used directly as Δ
- `relative_bound` -> `Δ = relative_bound * |control_value|` (per row)
- `cohens_d_bound` -> `Δ = cohens_d_bound * pooled_sd` (per row, OLS only)

For `cohens_d_bound` with non-OLS model types (logistic, Poisson, NB, Cox), issue a warning via the logger and skip those rows (set `eq_conclusion = NaN`), since Cohen's d is not well-defined for those models. Recommend using `absolute_bound` or `relative_bound` instead.

### Pooled SD Column

A `pooled_sd` column is added to the results DataFrame during `get_effects()`. Computed as:

```
pooled_sd = sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
```

Where `s1`, `s2` are the sample standard deviations of the outcome in treatment and control groups. This is computed for all model types (useful for descriptive purposes) but only used by `cohens_d_bound` for OLS.

### Test Logic

**Equivalence (TOST):** Two one-sided tests against symmetric bounds [-Δ, +Δ]:

- Lower test: H0: effect <= -Δ. z_lower = (effect + Δ) / SE. p_lower = 1 - Φ(z_lower)
- Upper test: H0: effect >= +Δ. z_upper = (Δ - effect) / SE. p_upper = 1 - Φ(z_upper)
- Conclude equivalence when max(p_lower, p_upper) < α
- 90% CI: effect ± z_{1-α} * SE (note: 1-2α confidence level, so z_{0.95} for α=0.05)

**Non-inferiority (one-sided):**

- `direction="higher_is_better"`: test lower bound only. H0: effect <= -Δ. Same z_lower formula.
- `direction="lower_is_better"`: test upper bound only. H0: effect >= +Δ. Same z_upper formula.

**Non-superiority (one-sided):**

- `direction="higher_is_better"`: test upper bound only. H0: effect >= +Δ. Same z_upper formula. Confirms treatment isn't too much better (e.g., cost didn't increase too much).
- `direction="lower_is_better"`: test lower bound only. H0: effect <= -Δ. Same z_lower formula. Confirms treatment isn't too much worse.

### Validation

- `test_type` must be one of `{"equivalence", "non_inferiority", "non_superiority"}`
- Exactly one bound type specified
- `alpha` in (0, 1)
- `direction` must be `{"higher_is_better", "lower_is_better"}`
- `direction` is only used (and required to be meaningful) for `non_inferiority` and `non_superiority` test types

### Result Columns Added

All columns are prefixed with `eq_` to avoid collision with existing columns.

| Column | Type | Description |
|--------|------|-------------|
| `eq_test_type` | str | "equivalence", "non_inferiority", or "non_superiority" |
| `eq_bound_lower` | float | Lower equivalence bound in raw units (-Δ) |
| `eq_bound_upper` | float | Upper equivalence bound in raw units (+Δ) |
| `eq_bound_type` | str | "absolute", "relative", or "cohens_d" |
| `eq_pvalue_lower` | float | p-value for lower bound test (H0: effect <= -Δ) |
| `eq_pvalue_upper` | float | p-value for upper bound test (H0: effect >= +Δ) |
| `eq_pvalue` | float | TOST: max(lower, upper). NI/NS: the relevant one-sided p-value |
| `eq_ci_lower` | float | Lower bound of (1-2α) CI |
| `eq_ci_upper` | float | Upper bound of (1-2α) CI |
| `eq_cohens_d` | float | Observed effect expressed as Cohen's d (absolute_effect / pooled_sd) |
| `eq_conclusion` | str | See conclusion logic below |

### Conclusion Logic (Lakens' Four-Cell Matrix)

Combines the TOST result with the existing NHST result (`stat_significance`) from `get_effects()`.

**For `test_type="equivalence"`:**

| NHST significant | TOST significant | `eq_conclusion` |
|---|---|---|
| No | Yes | `"equivalent"` |
| No | No | `"inconclusive"` |
| Yes | Yes | `"equivalent_with_difference"` |
| Yes | No | `"not_equivalent"` |

**For `test_type="non_inferiority"`:**

| NHST significant | NI test significant | `eq_conclusion` |
|---|---|---|
| No | Yes | `"non_inferior"` |
| No | No | `"inconclusive"` |
| Yes | Yes | `"non_inferior_with_difference"` |
| Yes | No | `"not_non_inferior"` |

**For `test_type="non_superiority"`:**

| NHST significant | NS test significant | `eq_conclusion` |
|---|---|---|
| No | Yes | `"non_superior"` |
| No | No | `"inconclusive"` |
| Yes | Yes | `"non_superior_with_difference"` |
| Yes | No | `"not_non_superior"` |

---

## 2. `pooled_sd` in `get_effects()`

### Change Location

`experiment_utils/experiment_analyzer.py` — inside `get_effects()`, at the point where per-comparison results are assembled.

### Computation

For each treatment-vs-control comparison, compute the pooled standard deviation of the outcome variable from the raw data:

```python
n1, n2 = len(treatment_data), len(control_data)
s1 = treatment_data[outcome].std(ddof=1)
s2 = control_data[outcome].std(ddof=1)
pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
```

Stored as a new column `pooled_sd` in the results DataFrame.

For ratio outcomes (delta method), compute `pooled_sd` on the linearized metric.

For Cox outcomes (time-to-event), compute `pooled_sd` on the time variable.

---

## 3. `plot_equivalence()` Visualization

### Standalone Function

```python
def plot_equivalence(
    data: pd.DataFrame,
    outcomes: list[str] | str | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    show_values: bool = True,
    value_decimals: int = 2,
    sort_by_magnitude: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
```

### Location

`experiment_utils/plotting.py` — new standalone function, following existing patterns.

### Required Columns

The input DataFrame must contain: `eq_ci_lower`, `eq_ci_upper`, `eq_bound_lower`, `eq_bound_upper`, `eq_conclusion`, `absolute_effect`, `outcome`, and a row identifier (experiment or treatment_group).

### Visual Design

Follows the existing `plot_effects()` style (Cleveland dot plot, same color palette file, same spine/grid styling):

- **Layout:** Horizontal dot plot, paneled by outcome
- **Equivalence region:** Shaded vertical band from `eq_bound_lower` to `eq_bound_upper` in a light color (similar to `_CLR_META_BG` but a light green/teal tint)
- **Bound lines:** Dashed vertical lines at -Δ and +Δ
- **Zero line:** Solid vertical line at 0 (same as `plot_effects()`)
- **Point estimate:** Dot at `absolute_effect`
- **CI bars:** Horizontal error bars from `eq_ci_lower` to `eq_ci_upper` (the 90% CI)
- **Color coding by `eq_conclusion`:**
  - `"equivalent"` / `"non_inferior"` / `"non_superior"` -> deep green (`#166534`)
  - `"not_equivalent"` / `"not_non_inferior"` / `"not_non_superior"` -> deep red (`#991b1b`)
  - `"inconclusive"` -> muted slate (`#64748b`, same as `_CLR_NSIG`)
  - `"equivalent_with_difference"` / `"*_with_difference"` -> amber (`#b45309`)
- **Value labels:** Show effect estimate and conclusion text next to each row (when `show_values=True`)
- **Axes, spines, fonts:** Match `plot_effects()` styling exactly

### Class Method Wrapper

```python
def plot_equivalence(self, outcomes=None, figsize=None, title=None,
                     show_values=True, value_decimals=2,
                     sort_by_magnitude=True, save_path=None) -> plt.Figure:
```

Extracts `self._results` and delegates to the standalone `plot_equivalence()`. Raises error if `test_equivalence()` hasn't been called (checks for `eq_conclusion` column).

---

## 4. `power_tost()` in PowerSim

### Method Signature

```python
def power_tost(
    self,
    sample_sizes: list[int],
    equivalence_bound: float,
    true_effect: float = 0.0,
    pooled_sd: float = 1.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
```

### Location

`experiment_utils/power_sim.py` — new method on `PowerSim`.

### Logic

For each sample size in `sample_sizes`:
1. Simulate `nsim` datasets (using existing simulation infrastructure in PowerSim)
2. For each simulated dataset, run the TOST procedure:
   - Compute effect estimate and SE from the simulated data
   - Run two one-sided tests against [-equivalence_bound, +equivalence_bound]
   - Record whether max(p_lower, p_upper) < alpha
3. Power = proportion of simulations where equivalence was concluded

### Parameters

- `sample_sizes`: per-group sample sizes to evaluate
- `equivalence_bound`: absolute Δ (raw units for "average", probability difference for "proportion")
- `true_effect`: the assumed true difference between groups (default 0 = truly equivalent)
- `pooled_sd`: population SD (for "average" metric; ignored for "proportion")
- `alpha`: significance level for TOST

Inherits from the `PowerSim` instance: `metric`, `nsim`, `early_stopping`, `early_stopping_precision`, `parallel_strategy`.

### Output

Returns a DataFrame with the same structure as existing power methods, so it integrates with `plot_power()`:

| Column | Description |
|--------|-------------|
| `sample_size` | Per-group N |
| `power` | Proportion of simulations achieving equivalence |
| `se` | Standard error of power estimate |
| `nsim` | Number of simulations run |

---

## 5. Exports and Tests

### `__init__.py` Updates

- Add `plot_equivalence` to the public exports (standalone function)
- No new classes needed

### Test Coverage

New test file: `tests/test_equivalence.py`

**Test cases for `test_equivalence()`:**

1. **TOST with OLS, absolute bound** — effect near zero, confirm "equivalent"
2. **TOST with OLS, large effect** — confirm "not_equivalent"
3. **TOST inconclusive** — effect near bound edge, insufficient power
4. **TOST equivalent_with_difference** — small but significant effect within bounds
5. **Non-inferiority, higher_is_better** — replicates old `test_non_inferiority` behavior
6. **Non-inferiority, lower_is_better** — mirror case
7. **Non-superiority** — tests upper bound only
8. **Relative bound** — bound computed from control_value
9. **Cohen's d bound** — bound computed from pooled_sd, OLS only
10. **Cohen's d with non-OLS model** — confirm warning and NaN conclusion
11. **Multiple outcomes** — different conclusions per outcome
12. **Multiple experiments** — correct per-experiment bounds
13. **Validation errors** — missing bounds, invalid test_type, etc.
14. **90% CI containment** — verify eq_ci within bounds iff equivalent
15. **Bootstrap inference** — TOST works with bootstrap SE

**Test cases for `plot_equivalence()`:**

16. **Basic rendering** — returns Figure, no errors
17. **Color coding** — correct colors for each conclusion type
18. **Paneled by outcome** — multiple panels

**Test cases for `power_tost()`:**

19. **High power scenario** — true_effect=0, large N, confirm power near 1
20. **Low power scenario** — small N, confirm low power
21. **Non-zero true effect** — power decreases as true_effect approaches bound
22. **Output format** — compatible with plot_power()

---

## 6. Files Modified

| File | Change |
|------|--------|
| `experiment_utils/experiment_analyzer.py` | Remove `test_non_inferiority()`, add `test_equivalence()`, add `pooled_sd` computation in `get_effects()`, add `plot_equivalence()` class wrapper |
| `experiment_utils/plotting.py` | Add standalone `plot_equivalence()` function |
| `experiment_utils/power_sim.py` | Add `power_tost()` method |
| `experiment_utils/__init__.py` | Export `plot_equivalence` |
| `tests/test_equivalence.py` | New test file |
| `tests/test_experiment_analyzer.py` | Update any tests that reference `test_non_inferiority()` |

---

## References

- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355-362.
- Lakens, D., Scheel, A. M., & Isager, P. M. (2018). Equivalence testing for psychological research: A tutorial. *Advances in Methods and Practices in Psychological Science*, 1(2), 259-269.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657-680.
