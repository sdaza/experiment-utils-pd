# TOST Equivalence Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `test_non_inferiority()` with a unified `test_equivalence()` method implementing TOST, non-inferiority, and non-superiority testing following Daniel Lakens' practices, plus a `plot_equivalence()` visualization and `power_tost()` for sample size planning.

**Architecture:** A single `test_equivalence()` method on `ExperimentAnalyzer` that operates on the results DataFrame produced by `get_effects()`, computing two one-sided test p-values and a (1-2alpha) CI. A `pooled_sd` column is added during `get_effects()`. A standalone `plot_equivalence()` function follows existing plotting patterns. `power_tost()` extends `PowerSim` with simulation-based TOST power.

**Tech Stack:** Python, numpy, scipy.stats, pandas, matplotlib, seaborn (all already in the project)

**Spec:** `docs/superpowers/specs/2026-04-01-tost-equivalence-testing-design.md`

---

### Task 1: Add `pooled_sd` Column to `get_effects()`

**Files:**
- Modify: `experiment_utils/experiment_analyzer.py:1612-1618` (after `output = estimator_func(...)`)
- Modify: `experiment_utils/experiment_analyzer.py:1813-1833` (error_output dict)
- Modify: `experiment_utils/experiment_analyzer.py:1847-1874` (result_columns list)
- Test: `tests/test_equivalence.py` (new file)

- [ ] **Step 1: Write the failing test for pooled_sd**

Create `tests/test_equivalence.py`:

```python
"""Tests for TOST equivalence testing."""

import numpy as np
import pandas as pd
import pytest

from experiment_utils import ExperimentAnalyzer


@pytest.fixture
def simple_experiment_data():
    """Generate simple experiment data with known properties."""
    rng = np.random.default_rng(42)
    n = 500
    treatment = np.array([0] * n + [1] * n)
    # Control: mean=10, sd=2. Treatment: mean=10.1, sd=2 (near-zero effect)
    outcome = np.where(
        treatment == 0,
        rng.normal(10, 2, 2 * n),
        rng.normal(10.1, 2, 2 * n),
    )[: 2 * n]
    # Fix: generate separately
    control_values = rng.normal(10, 2, n)
    treatment_values = rng.normal(10.1, 2, n)
    outcome = np.concatenate([control_values, treatment_values])
    return pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
        "experiment": "exp1",
    })


class TestPooledSd:
    """Tests for pooled_sd column in get_effects() results."""

    def test_pooled_sd_present_in_results(self, simple_experiment_data):
        """pooled_sd column should be present after get_effects()."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        assert "pooled_sd" in analyzer.results.columns

    def test_pooled_sd_value_is_reasonable(self, simple_experiment_data):
        """pooled_sd should be close to the true SD (2.0) for our generated data."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        pooled_sd = analyzer.results["pooled_sd"].iloc[0]
        # True SD is 2.0, should be within 0.3
        assert abs(pooled_sd - 2.0) < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_equivalence.py::TestPooledSd -v`
Expected: FAIL with `KeyError: 'pooled_sd'` or `AssertionError`

- [ ] **Step 3: Implement pooled_sd computation in get_effects()**

In `experiment_utils/experiment_analyzer.py`, after line 1617 (`output["control_std"] = ...`), add:

```python
                            # Compute pooled SD for equivalence testing (Cohen's d bounds)
                            treatment_mask = comparison_data[self._treatment_col] == 1
                            control_outcome = comparison_data.loc[control_mask, outcome]
                            treatment_outcome = comparison_data.loc[treatment_mask, outcome]
                            n_c, n_t = len(control_outcome), len(treatment_outcome)
                            if n_c > 1 and n_t > 1:
                                s_c = control_outcome.std(ddof=1)
                                s_t = treatment_outcome.std(ddof=1)
                                output["pooled_sd"] = np.sqrt(
                                    ((n_c - 1) * s_c**2 + (n_t - 1) * s_t**2) / (n_c + n_t - 2)
                                )
                            else:
                                output["pooled_sd"] = np.nan
```

Note: `control_mask` is already defined at line 1616. `treatment_mask` is its complement.

In the error_output dict (around line 1813), add:

```python
                                "pooled_sd": np.nan,
```

In the `result_columns` list (around line 1847), add `"pooled_sd"` after `"control_std"`:

```python
                    "control_std",
                    "pooled_sd",
                    "alpha_param",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_equivalence.py::TestPooledSd -v`
Expected: PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff format experiment_utils/experiment_analyzer.py tests/test_equivalence.py
uv run ruff check experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git add experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git commit -m "feat: add pooled_sd column to get_effects() results"
```

---

### Task 2: Implement `test_equivalence()` — Core TOST Logic

**Files:**
- Modify: `experiment_utils/experiment_analyzer.py:1955-2040` (replace `test_non_inferiority`)
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write failing tests for TOST equivalence**

Append to `tests/test_equivalence.py`:

```python
class TestEquivalenceTOST:
    """Tests for test_equivalence() with test_type='equivalence'."""

    def test_equivalent_with_absolute_bound(self, simple_experiment_data):
        """Near-zero effect with wide bounds should conclude 'equivalent'."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "equivalence"
        assert result["eq_bound_type"] == "absolute"
        assert result["eq_bound_lower"] == pytest.approx(-1.0)
        assert result["eq_bound_upper"] == pytest.approx(1.0)
        assert result["eq_pvalue_lower"] < 0.05
        assert result["eq_pvalue_upper"] < 0.05
        assert result["eq_pvalue"] < 0.05
        assert result["eq_conclusion"] in ("equivalent", "equivalent_with_difference")

    def test_not_equivalent_with_tight_bound(self):
        """Large effect with tight bounds should conclude 'not_equivalent'."""
        rng = np.random.default_rng(123)
        n = 200
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(15, 2, n)]),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert result["eq_conclusion"] == "not_equivalent"

    def test_90_percent_ci_computed(self, simple_experiment_data):
        """90% CI (1-2*alpha) should be present and narrower than 95% CI."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0, alpha=0.05)
        result = analyzer.results.iloc[0]
        # 90% CI should exist
        assert pd.notna(result["eq_ci_lower"])
        assert pd.notna(result["eq_ci_upper"])
        # 90% CI should be narrower than 95% CI
        eq_width = result["eq_ci_upper"] - result["eq_ci_lower"]
        nhst_width = result["abs_effect_upper"] - result["abs_effect_lower"]
        assert eq_width < nhst_width

    def test_cohens_d_reported(self, simple_experiment_data):
        """eq_cohens_d should be absolute_effect / pooled_sd."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        expected_d = result["absolute_effect"] / result["pooled_sd"]
        assert result["eq_cohens_d"] == pytest.approx(expected_d, rel=1e-6)

    def test_requires_get_effects_first(self, simple_experiment_data):
        """Should raise error if get_effects() not called."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=1.0)

    def test_must_specify_exactly_one_bound(self, simple_experiment_data):
        """Should raise error if zero or multiple bounds specified."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence()  # no bound
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=1.0, relative_bound=0.1)  # two bounds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equivalence.py::TestEquivalenceTOST -v`
Expected: FAIL with `AttributeError: 'ExperimentAnalyzer' object has no attribute 'test_equivalence'`

- [ ] **Step 3: Implement test_equivalence() — replace test_non_inferiority()**

In `experiment_utils/experiment_analyzer.py`, replace the entire `test_non_inferiority` method (lines 1955-2040) with:

```python
    def test_equivalence(
        self,
        test_type: str = "equivalence",
        absolute_bound: float | None = None,
        relative_bound: float | None = None,
        cohens_d_bound: float | None = None,
        alpha: float = 0.05,
        direction: str = "higher_is_better",
    ) -> None:
        """
        Unified equivalence, non-inferiority, and non-superiority testing.

        Implements the Two One-Sided Tests (TOST) procedure following Lakens (2017).
        Non-inferiority and non-superiority are one-sided special cases of the same
        framework.

        Parameters
        ----------
        test_type : {"equivalence", "non_inferiority", "non_superiority"}
            ``"equivalence"``: TOST — tests both bounds, concludes equivalence
            when effect falls within [-Δ, +Δ].
            ``"non_inferiority"``: one-sided test against the lower (or upper) bound.
            ``"non_superiority"``: one-sided test against the opposite bound.
        absolute_bound : float, optional
            Equivalence bound in raw outcome units (must be > 0).
        relative_bound : float, optional
            Equivalence bound as a fraction of ``|control_value|``
            (e.g. ``0.10`` = 10 %). Must be > 0.
        cohens_d_bound : float, optional
            Equivalence bound in standardized units (Cohen's d). Converted to
            raw units via ``pooled_sd``. Only valid for OLS model types.
        alpha : float, optional
            Significance level (default 0.05). For TOST the confidence interval
            is (1 − 2α), i.e. 90 % by default.
        direction : {"higher_is_better", "lower_is_better"}
            Only used for ``non_inferiority`` and ``non_superiority`` test types.

        Updates
        -------
        self._results : pd.DataFrame
            Adds columns prefixed with ``eq_``. See design spec for full list.
        """
        if self._results is None:
            log_and_raise_error(self._logger, "Must run get_effects() before test_equivalence().")

        valid_test_types = {"equivalence", "non_inferiority", "non_superiority"}
        if test_type not in valid_test_types:
            log_and_raise_error(self._logger, f"test_type must be one of {valid_test_types}.")

        if not (0 < alpha < 1):
            log_and_raise_error(self._logger, "alpha must be between 0 and 1 (exclusive).")

        valid_directions = {"higher_is_better", "lower_is_better"}
        if direction not in valid_directions:
            log_and_raise_error(self._logger, f"direction must be one of {valid_directions}.")

        # Exactly one bound type
        bounds_specified = sum(b is not None for b in [absolute_bound, relative_bound, cohens_d_bound])
        if bounds_specified != 1:
            log_and_raise_error(
                self._logger,
                "Provide exactly one of absolute_bound, relative_bound, or cohens_d_bound.",
            )

        if absolute_bound is not None and absolute_bound <= 0:
            log_and_raise_error(self._logger, "absolute_bound must be > 0.")
        if relative_bound is not None and relative_bound <= 0:
            log_and_raise_error(self._logger, "relative_bound must be > 0.")
        if cohens_d_bound is not None and cohens_d_bound <= 0:
            log_and_raise_error(self._logger, "cohens_d_bound must be > 0.")

        results_df = self._results.copy()

        # Resolve bounds to absolute units (Δ per row)
        if absolute_bound is not None:
            results_df["_delta"] = float(absolute_bound)
            bound_type = "absolute"
        elif relative_bound is not None:
            results_df["_delta"] = relative_bound * results_df["control_value"].abs()
            bound_type = "relative"
        else:
            # Cohen's d: only valid for OLS
            if "model_type" in results_df.columns:
                non_ols = results_df["model_type"] != "ols"
                if non_ols.any():
                    self._logger.warning(
                        "cohens_d_bound is only valid for OLS models. "
                        "Non-OLS rows will have NaN equivalence results. "
                        "Use absolute_bound or relative_bound for non-OLS models."
                    )
            if "pooled_sd" not in results_df.columns:
                log_and_raise_error(self._logger, "pooled_sd column not found. Re-run get_effects().")
            results_df["_delta"] = cohens_d_bound * results_df["pooled_sd"]
            bound_type = "cohens_d"

        # For Cohen's d with non-OLS, set delta to NaN
        if cohens_d_bound is not None and "model_type" in results_df.columns:
            non_ols_mask = results_df["model_type"] != "ols"
            results_df.loc[non_ols_mask, "_delta"] = np.nan

        # Store bounds
        results_df["eq_test_type"] = test_type
        results_df["eq_bound_type"] = bound_type
        results_df["eq_bound_lower"] = -results_df["_delta"]
        results_df["eq_bound_upper"] = results_df["_delta"]

        # Two one-sided test statistics
        effect = results_df["absolute_effect"]
        se = results_df["standard_error"]
        delta = results_df["_delta"]

        z_lower = (effect + delta) / se  # H0: effect <= -Δ
        z_upper = (delta - effect) / se  # H0: effect >= +Δ

        results_df["eq_pvalue_lower"] = 1 - stats.norm.cdf(z_lower)
        results_df["eq_pvalue_upper"] = 1 - stats.norm.cdf(z_upper)

        # (1-2α) confidence interval
        z_crit = stats.norm.ppf(1 - alpha)
        results_df["eq_ci_lower"] = effect - z_crit * se
        results_df["eq_ci_upper"] = effect + z_crit * se

        # Cohen's d for observed effect
        if "pooled_sd" in results_df.columns:
            results_df["eq_cohens_d"] = effect / results_df["pooled_sd"]
        else:
            results_df["eq_cohens_d"] = np.nan

        # Determine the reported p-value based on test type
        if test_type == "equivalence":
            results_df["eq_pvalue"] = results_df[["eq_pvalue_lower", "eq_pvalue_upper"]].max(axis=1)
        elif test_type == "non_inferiority":
            if direction == "higher_is_better":
                results_df["eq_pvalue"] = results_df["eq_pvalue_lower"]
            else:
                results_df["eq_pvalue"] = results_df["eq_pvalue_upper"]
        else:  # non_superiority
            if direction == "higher_is_better":
                results_df["eq_pvalue"] = results_df["eq_pvalue_upper"]
            else:
                results_df["eq_pvalue"] = results_df["eq_pvalue_lower"]

        # Conclusion logic (Lakens' four-cell matrix)
        tost_sig = results_df["eq_pvalue"] < alpha

        # Use MCP-adjusted significance if available, otherwise standard
        sig_col = "stat_significance_mcp" if "stat_significance_mcp" in results_df.columns else "stat_significance"
        nhst_sig = results_df[sig_col] == 1

        if test_type == "equivalence":
            labels = {
                (False, True): "equivalent",
                (False, False): "inconclusive",
                (True, True): "equivalent_with_difference",
                (True, False): "not_equivalent",
            }
        elif test_type == "non_inferiority":
            labels = {
                (False, True): "non_inferior",
                (False, False): "inconclusive",
                (True, True): "non_inferior_with_difference",
                (True, False): "not_non_inferior",
            }
        else:  # non_superiority
            labels = {
                (False, True): "non_superior",
                (False, False): "inconclusive",
                (True, True): "non_superior_with_difference",
                (True, False): "not_non_superior",
            }

        results_df["eq_conclusion"] = [
            labels.get((bool(n), bool(t)), "inconclusive")
            for n, t in zip(nhst_sig, tost_sig)
        ]

        # NaN out conclusions for invalid rows (e.g., Cohen's d with non-OLS)
        nan_mask = results_df["_delta"].isna()
        eq_cols = [c for c in results_df.columns if c.startswith("eq_") and c != "eq_test_type" and c != "eq_bound_type"]
        for col in eq_cols:
            results_df.loc[nan_mask, col] = np.nan

        results_df = results_df.drop(columns=["_delta"])
        self._results = results_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff format experiment_utils/experiment_analyzer.py tests/test_equivalence.py
uv run ruff check experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git add experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git commit -m "feat: add test_equivalence() replacing test_non_inferiority()"
```

---

### Task 3: Add Tests for Bound Types and Non-Inferiority/Non-Superiority

**Files:**
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write tests for relative bound, Cohen's d bound, and one-sided tests**

Append to `tests/test_equivalence.py`:

```python
class TestEquivalenceBoundTypes:
    """Tests for different bound specifications."""

    def test_relative_bound(self, simple_experiment_data):
        """Relative bound should compute delta from control_value."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", relative_bound=0.20)
        result = analyzer.results.iloc[0]
        expected_delta = 0.20 * abs(result["control_value"])
        assert result["eq_bound_upper"] == pytest.approx(expected_delta, rel=1e-6)
        assert result["eq_bound_lower"] == pytest.approx(-expected_delta, rel=1e-6)
        assert result["eq_bound_type"] == "relative"

    def test_cohens_d_bound_ols(self, simple_experiment_data):
        """Cohen's d bound should compute delta from pooled_sd."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", cohens_d_bound=0.5)
        result = analyzer.results.iloc[0]
        expected_delta = 0.5 * result["pooled_sd"]
        assert result["eq_bound_upper"] == pytest.approx(expected_delta, rel=1e-6)
        assert result["eq_bound_type"] == "cohens_d"

    def test_cohens_d_warns_non_ols(self):
        """Cohen's d with logistic model should produce NaN conclusion."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome": rng.binomial(1, 0.5, 2 * n),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
            outcome_models="logistic",
        )
        analyzer.get_effects()
        analyzer.test_equivalence(test_type="equivalence", cohens_d_bound=0.3)
        result = analyzer.results.iloc[0]
        assert pd.isna(result["eq_conclusion"])


class TestNonInferiority:
    """Tests for non-inferiority and non-superiority test types."""

    def test_non_inferiority_higher_is_better(self, simple_experiment_data):
        """Non-inferiority with higher_is_better tests lower bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "non_inferiority"
        # eq_pvalue should equal eq_pvalue_lower for higher_is_better
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_lower"])

    def test_non_inferiority_lower_is_better(self, simple_experiment_data):
        """Non-inferiority with lower_is_better tests upper bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="lower_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_upper"])

    def test_non_superiority_higher_is_better(self, simple_experiment_data):
        """Non-superiority with higher_is_better tests upper bound only."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_superiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        assert result["eq_test_type"] == "non_superiority"
        assert result["eq_pvalue"] == pytest.approx(result["eq_pvalue_upper"])

    def test_non_inferiority_conclusion_labels(self, simple_experiment_data):
        """Non-inferiority should use 'non_inferior' labels, not 'equivalent'."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(
            test_type="non_inferiority",
            absolute_bound=1.0,
            direction="higher_is_better",
        )
        result = analyzer.results.iloc[0]
        valid_conclusions = {"non_inferior", "not_non_inferior", "inconclusive", "non_inferior_with_difference"}
        assert result["eq_conclusion"] in valid_conclusions


class TestEquivalenceMultiple:
    """Tests for multiple outcomes and experiments."""

    def test_multiple_outcomes(self):
        """TOST should produce results per outcome."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome1": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
            "outcome2": np.concatenate([rng.normal(5, 1, n), rng.normal(8, 1, n)]),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome1", "outcome2"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        results = analyzer.results
        assert len(results) == 2
        # outcome1 (small effect) should be equivalent; outcome2 (large effect) should not
        o1 = results[results["outcome"] == "outcome1"].iloc[0]
        o2 = results[results["outcome"] == "outcome2"].iloc[0]
        assert o1["eq_conclusion"] in ("equivalent", "equivalent_with_difference")
        assert o2["eq_conclusion"] == "not_equivalent"

    def test_multiple_experiments(self):
        """TOST should produce results per experiment."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n + [0] * n + [1] * n,
            "outcome": np.concatenate([
                rng.normal(10, 2, n), rng.normal(10.1, 2, n),
                rng.normal(10, 2, n), rng.normal(10.05, 2, n),
            ]),
            "experiment": ["exp1"] * (2 * n) + ["exp2"] * (2 * n),
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        results = analyzer.results
        assert len(results) == 2
        # Both should have their own eq_pvalue
        assert results["eq_pvalue"].notna().all()

    def test_bootstrap_inference(self):
        """TOST should work with bootstrap SE."""
        rng = np.random.default_rng(42)
        n = 200
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
            bootstrap=True,
            bootstrap_iterations=100,
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        result = analyzer.results.iloc[0]
        assert pd.notna(result["eq_pvalue"])
        assert pd.notna(result["eq_ci_lower"])
        assert result["eq_conclusion"] in ("equivalent", "equivalent_with_difference", "inconclusive", "not_equivalent")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py -v`
Expected: All PASS (implementation was done in Task 2)

- [ ] **Step 3: Run ruff and commit**

```bash
uv run ruff format tests/test_equivalence.py
uv run ruff check tests/test_equivalence.py
git add tests/test_equivalence.py
git commit -m "test: add tests for bound types and non-inferiority/non-superiority"
```

---

### Task 4: Add Validation Error Tests

**Files:**
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write validation tests**

Append to `tests/test_equivalence.py`:

```python
class TestEquivalenceValidation:
    """Tests for input validation in test_equivalence()."""

    def test_invalid_test_type(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(test_type="invalid", absolute_bound=1.0)

    def test_no_bounds_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(test_type="equivalence")

    def test_two_bounds_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=1.0, relative_bound=0.1)

    def test_negative_bound_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=-1.0)

    def test_invalid_alpha_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=1.0, alpha=0.0)

    def test_invalid_direction_raises(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.test_equivalence(absolute_bound=1.0, direction="invalid")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py::TestEquivalenceValidation -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
uv run ruff format tests/test_equivalence.py
uv run ruff check tests/test_equivalence.py
git add tests/test_equivalence.py
git commit -m "test: add validation error tests for test_equivalence()"
```

---

### Task 5: Update Interactive Test and Remove Old test_non_inferiority Reference

**Files:**
- Modify: `tests/interactive_testing.py:201` (update call)

- [ ] **Step 1: Update the interactive testing reference**

In `tests/interactive_testing.py`, replace line 201:

```python
# Old
analyzer.test_non_inferiority(relative_margin=0.10)
# New
analyzer.test_equivalence(test_type="non_inferiority", relative_bound=0.10)
```

- [ ] **Step 2: Run the full existing test suite to check nothing is broken**

Run: `uv run pytest tests/test_experiment_analyzer.py -v --timeout=120`
Expected: All PASS (no test references `test_non_inferiority` directly)

- [ ] **Step 3: Commit**

```bash
uv run ruff format tests/interactive_testing.py
uv run ruff check tests/interactive_testing.py
git add tests/interactive_testing.py
git commit -m "refactor: update interactive_testing to use test_equivalence()"
```

---

### Task 6: Implement `plot_equivalence()` Standalone Function

**Files:**
- Modify: `experiment_utils/plotting.py` (add function at end of file, add color constants)
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write failing test for plot_equivalence**

Append to `tests/test_equivalence.py`:

```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from experiment_utils import plot_equivalence


class TestPlotEquivalence:
    """Tests for the plot_equivalence() standalone function."""

    def _make_equivalence_results(self):
        """Helper: create analyzer with equivalence results."""
        rng = np.random.default_rng(42)
        n = 500
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        return analyzer.results

    def test_returns_figure(self):
        """plot_equivalence should return a matplotlib Figure."""
        results = self._make_equivalence_results()
        fig = plot_equivalence(data=results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_raises_without_eq_columns(self):
        """Should raise error if eq_ columns are missing."""
        data = pd.DataFrame({"absolute_effect": [0.1], "outcome": ["x"]})
        with pytest.raises(Exception):
            plot_equivalence(data=data)

    def test_multiple_outcomes(self):
        """Should handle multiple outcomes in panels."""
        rng = np.random.default_rng(42)
        n = 300
        data = pd.DataFrame({
            "treatment": [0] * n + [1] * n,
            "outcome1": np.concatenate([rng.normal(10, 2, n), rng.normal(10.1, 2, n)]),
            "outcome2": np.concatenate([rng.normal(5, 1, n), rng.normal(5.05, 1, n)]),
            "experiment": "exp1",
        })
        analyzer = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome1", "outcome2"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        fig = plot_equivalence(data=analyzer.results)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equivalence.py::TestPlotEquivalence -v`
Expected: FAIL with `ImportError` (plot_equivalence not yet exported)

- [ ] **Step 3: Implement plot_equivalence() in plotting.py**

Add color constants near the top of `experiment_utils/plotting.py` (after line 25):

```python
# Equivalence test conclusion colors
_CLR_EQ = "#166534"        # deep green — equivalent / non-inferior
_CLR_NOT_EQ = "#991b1b"    # deep red — not equivalent
_CLR_EQ_DIFF = "#b45309"   # amber — equivalent with difference
_CLR_EQ_BAND = "#dcfce7"   # light green — equivalence region fill
```

Add the standalone function at the end of `experiment_utils/plotting.py`:

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
) -> "plt.Figure":
    """
    Equivalence test visualization with (1-2α) CI against equivalence bounds.

    Displays a Cleveland-style dot plot where each row shows the point estimate
    and the TOST confidence interval.  A shaded band marks the equivalence
    region [-Δ, +Δ].  Dots are color-coded by the ``eq_conclusion`` column.

    Parameters
    ----------
    data : pd.DataFrame
        Results DataFrame containing ``eq_*`` columns from
        :meth:`ExperimentAnalyzer.test_equivalence`.
    outcomes : list[str] or str, optional
        Subset of outcomes to plot. ``None`` plots all.
    figsize : tuple, optional
        Figure size ``(width, height)``.
    title : str, optional
        Overall figure title.
    show_values : bool
        Annotate each row with the numeric effect and conclusion.
    value_decimals : int
        Decimal places for value annotations.
    sort_by_magnitude : bool
        Sort rows by absolute effect magnitude.
    save_path : str, optional
        Save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    required = ["eq_ci_lower", "eq_ci_upper", "eq_bound_lower", "eq_bound_upper", "eq_conclusion", "absolute_effect"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Run test_equivalence() first."
        )

    df = data.copy()

    # Determine row identifier
    if "experiment" in df.columns:
        row_col = "experiment"
    elif "treatment_group" in df.columns:
        row_col = "treatment_group"
    else:
        df["_row"] = range(len(df))
        row_col = "_row"

    # Build row label from treatment_group vs control_group if both exist
    if "treatment_group" in df.columns and "control_group" in df.columns:
        df["_row_label"] = df["treatment_group"].astype(str) + " vs " + df["control_group"].astype(str)
        if "experiment" in df.columns:
            # Combine experiment + comparison
            df["_row_label"] = df["experiment"].astype(str) + ": " + df["_row_label"]
        row_col = "_row_label"

    # Filter outcomes
    if "outcome" not in df.columns:
        df["outcome"] = "outcome"
    if outcomes is not None:
        if isinstance(outcomes, str):
            outcomes = [outcomes]
        df = df[df["outcome"].isin(outcomes)]
    if df.empty:
        raise ValueError("No data to plot after filtering outcomes.")

    unique_outcomes = df["outcome"].unique()
    n_panels = len(unique_outcomes)

    # Figure sizing
    if figsize is None:
        max_rows = max(len(df[df["outcome"] == o]) for o in unique_outcomes)
        fig_h = max(2.5, 0.45 * max_rows * n_panels + 1.5)
        fig_w = 8
        figsize = (fig_w, fig_h)

    fig, axes_arr = plt.subplots(n_panels, 1, figsize=figsize, squeeze=False)
    axes = axes_arr.flatten()

    # Color mapping
    conclusion_colors = {}
    for label in ("equivalent", "non_inferior", "non_superior"):
        conclusion_colors[label] = _CLR_EQ
    for label in ("not_equivalent", "not_non_inferior", "not_non_superior"):
        conclusion_colors[label] = _CLR_NOT_EQ
    for label in ("equivalent_with_difference", "non_inferior_with_difference", "non_superior_with_difference"):
        conclusion_colors[label] = _CLR_EQ_DIFF
    conclusion_colors["inconclusive"] = _CLR_NSIG

    for panel_idx, (ax, outcome_val) in enumerate(zip(axes, unique_outcomes, strict=False)):
        od = df[df["outcome"] == outcome_val].copy()
        if sort_by_magnitude:
            od = od.sort_values(by="absolute_effect", key=abs, ascending=False)

        labels = list(od[row_col])
        effs = list(od["absolute_effect"])
        ci_los = list(od["eq_ci_lower"])
        ci_his = list(od["eq_ci_upper"])
        conclusions = list(od["eq_conclusion"])
        bound_lo = od["eq_bound_lower"].iloc[0] if not od.empty else 0
        bound_hi = od["eq_bound_upper"].iloc[0] if not od.empty else 0

        n_rows = len(labels)
        y_pos = list(range(n_rows))

        ax.set_facecolor("white")
        ax.set_axisbelow(True)

        # Equivalence region band
        ax.axvspan(bound_lo, bound_hi, color=_CLR_EQ_BAND, alpha=0.5, zorder=0)
        # Bound lines
        ax.axvline(bound_lo, color=_CLR_EQ, linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
        ax.axvline(bound_hi, color=_CLR_EQ, linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
        # Zero line
        ax.axvline(0, color=_CLR_ZERO, linestyle="-", linewidth=1.0, alpha=0.55, zorder=1)

        # Guide lines
        for i in range(n_rows):
            ax.axhline(i, color=_CLR_GUIDE, linewidth=0.6, linestyle=":", zorder=0)

        for i, (label, eff, lo, hi, conclusion) in enumerate(
            zip(labels, effs, ci_los, ci_his, conclusions, strict=False)
        ):
            if eff is None or not pd.notna(eff) or not np.isfinite(eff):
                continue

            color = conclusion_colors.get(str(conclusion), _CLR_NSIG)

            # CI bar
            if pd.notna(lo) and pd.notna(hi) and np.isfinite(lo) and np.isfinite(hi):
                ax.plot([lo, hi], [i, i], color=color, linewidth=1.4, solid_capstyle="butt", zorder=3)
                cap_h = 0.06
                ax.plot([lo, lo], [i - cap_h, i + cap_h], color=color, linewidth=1.4, zorder=3)
                ax.plot([hi, hi], [i - cap_h, i + cap_h], color=color, linewidth=1.4, zorder=3)

            # Point estimate
            ax.scatter([eff], [i], color=color, s=45, zorder=4, marker="o", edgecolors="white", linewidths=0.5)

            # Value annotation
            if show_values:
                conclusion_label = str(conclusion).replace("_", " ") if pd.notna(conclusion) else ""
                text = f"{eff:+.{value_decimals}f}  [{conclusion_label}]"
                ax.annotate(
                    text,
                    (hi if pd.notna(hi) else eff, i),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=color,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Effect (with equivalence bounds)", fontsize=9)

        # Panel title
        panel_title = outcome_val if n_panels > 1 else (title or outcome_val)
        ax.set_title(panel_title, fontsize=10, fontweight="bold", loc="left")

        # Spine styling (matching plot_effects)
        for spine in ax.spines.values():
            spine.set_color(_CLR_SPINE)
            spine.set_linewidth(0.8)
        ax.tick_params(colors=_CLR_SPINE, labelsize=8)

    if title and n_panels > 1:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
```

- [ ] **Step 4: Export plot_equivalence in __init__.py**

In `experiment_utils/__init__.py`, add the import and export:

```python
from .plotting import plot_effects, plot_equivalence, plot_overlap, plot_power
```

And in `__all__`:

```python
    "plot_equivalence",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py::TestPlotEquivalence -v`
Expected: All PASS

- [ ] **Step 6: Run ruff and commit**

```bash
uv run ruff format experiment_utils/plotting.py experiment_utils/__init__.py tests/test_equivalence.py
uv run ruff check experiment_utils/plotting.py experiment_utils/__init__.py tests/test_equivalence.py
git add experiment_utils/plotting.py experiment_utils/__init__.py tests/test_equivalence.py
git commit -m "feat: add plot_equivalence() standalone function and export"
```

---

### Task 7: Add `plot_equivalence()` Class Method Wrapper

**Files:**
- Modify: `experiment_utils/experiment_analyzer.py` (add method after `plot_effects`)
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write failing test for class method wrapper**

Append to `tests/test_equivalence.py`:

```python
class TestPlotEquivalenceClassMethod:
    """Tests for ExperimentAnalyzer.plot_equivalence() wrapper."""

    def test_class_method_returns_figure(self, simple_experiment_data):
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        analyzer.test_equivalence(absolute_bound=1.0)
        fig = analyzer.plot_equivalence()
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_class_method_raises_without_test_equivalence(self, simple_experiment_data):
        """Should raise if test_equivalence() hasn't been called."""
        analyzer = ExperimentAnalyzer(
            data=simple_experiment_data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment"],
        )
        analyzer.get_effects()
        with pytest.raises(Exception):
            analyzer.plot_equivalence()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equivalence.py::TestPlotEquivalenceClassMethod -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement the class method wrapper**

In `experiment_utils/experiment_analyzer.py`, add after the `plot_effects` method (after line ~2206):

```python
    def plot_equivalence(
        self,
        outcomes: list[str] | str | None = None,
        figsize: tuple | None = None,
        title: str | None = None,
        show_values: bool = True,
        value_decimals: int = 2,
        sort_by_magnitude: bool = True,
        save_path: str | None = None,
    ) -> "plt.Figure":
        """
        Plot equivalence test results (TOST visualization).

        Requires :meth:`test_equivalence` to have been called first.
        Delegates to the standalone :func:`plot_equivalence` function.

        Parameters
        ----------
        outcomes : list[str] or str, optional
            Subset of outcomes to plot.
        figsize : tuple, optional
            Figure size ``(width, height)``.
        title : str, optional
            Overall figure title.
        show_values : bool
            Annotate each row with effect and conclusion.
        value_decimals : int
            Decimal places for annotations.
        sort_by_magnitude : bool
            Sort rows by |effect|.
        save_path : str, optional
            Path to save the figure.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self._results is None or "eq_conclusion" not in self._results.columns:
            log_and_raise_error(
                self._logger,
                "Must run test_equivalence() before plot_equivalence().",
            )

        from .plotting import plot_equivalence as _plot_equivalence

        return _plot_equivalence(
            data=self._results,
            outcomes=outcomes,
            figsize=figsize,
            title=title,
            show_values=show_values,
            value_decimals=value_decimals,
            sort_by_magnitude=sort_by_magnitude,
            save_path=save_path,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py::TestPlotEquivalenceClassMethod -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff format experiment_utils/experiment_analyzer.py tests/test_equivalence.py
uv run ruff check experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git add experiment_utils/experiment_analyzer.py tests/test_equivalence.py
git commit -m "feat: add plot_equivalence() class method wrapper"
```

---

### Task 8: Implement `power_tost()` in PowerSim

**Files:**
- Modify: `experiment_utils/power_sim.py` (add method)
- Test: `tests/test_equivalence.py`

- [ ] **Step 1: Write failing tests for power_tost**

Append to `tests/test_equivalence.py`:

```python
from experiment_utils import PowerSim


class TestPowerTOST:
    """Tests for PowerSim.power_tost()."""

    def test_high_power_scenario(self):
        """Large N, true_effect=0, wide bounds should give power near 1."""
        ps = PowerSim(metric="average", nsim=200)
        result = ps.power_tost(
            sample_sizes=[500, 1000],
            equivalence_bound=1.0,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        assert isinstance(result, pd.DataFrame)
        assert "sample_size" in result.columns
        assert "power" in result.columns
        # With n=1000, sd=2, bound=1, true_effect=0, power should be high
        high_n_power = result[result["sample_size"] == 1000]["power"].iloc[0]
        assert high_n_power > 0.7

    def test_low_power_scenario(self):
        """Small N should give low power."""
        ps = PowerSim(metric="average", nsim=200)
        result = ps.power_tost(
            sample_sizes=[10],
            equivalence_bound=0.5,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        low_n_power = result[result["sample_size"] == 10]["power"].iloc[0]
        assert low_n_power < 0.3

    def test_power_increases_with_sample_size(self):
        """Power should increase as sample size grows."""
        ps = PowerSim(metric="average", nsim=300)
        result = ps.power_tost(
            sample_sizes=[50, 200, 500],
            equivalence_bound=0.8,
            true_effect=0.0,
            pooled_sd=2.0,
            alpha=0.05,
        )
        powers = result.sort_values("sample_size")["power"].tolist()
        # Power should be monotonically non-decreasing (with some simulation noise)
        assert powers[-1] >= powers[0]

    def test_proportion_metric(self):
        """power_tost should work with proportion metric."""
        ps = PowerSim(metric="proportion", nsim=200)
        result = ps.power_tost(
            sample_sizes=[200, 500],
            equivalence_bound=0.05,
            true_effect=0.0,
            pooled_sd=1.0,  # ignored for proportions
            alpha=0.05,
            baseline=0.5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_output_compatible_with_plot_power(self):
        """Output should have columns compatible with plot_power."""
        ps = PowerSim(metric="average", nsim=100)
        result = ps.power_tost(
            sample_sizes=[100, 200],
            equivalence_bound=1.0,
            true_effect=0.0,
            pooled_sd=2.0,
        )
        assert "sample_size" in result.columns
        assert "power" in result.columns
        assert "se" in result.columns
        assert "nsim" in result.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_equivalence.py::TestPowerTOST -v`
Expected: FAIL with `AttributeError: 'PowerSim' object has no attribute 'power_tost'`

- [ ] **Step 3: Implement power_tost() in PowerSim**

Add the following method to the `PowerSim` class in `experiment_utils/power_sim.py`:

```python
    def power_tost(
        self,
        sample_sizes: list[int],
        equivalence_bound: float,
        true_effect: float = 0.0,
        pooled_sd: float = 1.0,
        alpha: float = 0.05,
        baseline: float | None = None,
    ) -> pd.DataFrame:
        """
        Simulation-based power analysis for TOST equivalence testing.

        For each sample size, simulate ``nsim`` datasets and compute the
        proportion where TOST concludes equivalence at the given alpha level.

        Parameters
        ----------
        sample_sizes : list[int]
            Per-group sample sizes to evaluate.
        equivalence_bound : float
            Absolute equivalence bound Δ. For ``"average"`` metric this is in
            raw units; for ``"proportion"`` it is a probability difference.
        true_effect : float
            True difference between groups (default 0 = truly equivalent).
        pooled_sd : float
            Population SD for ``"average"`` metric. Ignored for ``"proportion"``.
        alpha : float
            Significance level for TOST (default 0.05).
        baseline : float, optional
            Baseline rate for ``"proportion"`` or ``"count"`` metrics.
            Required when ``metric`` is not ``"average"``.

        Returns
        -------
        pd.DataFrame
            Columns: ``sample_size``, ``power``, ``se``, ``nsim``.
        """
        if equivalence_bound <= 0:
            log_and_raise_error(self.logger, "equivalence_bound must be > 0.")
        if not (0 < alpha < 1):
            log_and_raise_error(self.logger, "alpha must be between 0 and 1.")

        if self.metric != "average" and baseline is None:
            log_and_raise_error(
                self.logger,
                f"baseline is required for metric='{self.metric}'.",
            )

        results = []

        for n in sample_sizes:
            equivalence_count = 0
            sims_run = 0

            for sim_i in range(self.nsim):
                rng = np.random.default_rng(sim_i)

                # Generate data
                if self.metric == "average":
                    control = rng.normal(0, pooled_sd, n)
                    treatment = rng.normal(true_effect, pooled_sd, n)
                elif self.metric == "proportion":
                    p_control = baseline
                    p_treatment = min(max(baseline + true_effect, 0.001), 0.999)
                    control = rng.binomial(1, p_control, n)
                    treatment = rng.binomial(1, p_treatment, n)
                else:  # count
                    control = rng.poisson(baseline, n)
                    treatment = rng.poisson(max(baseline + true_effect, 0.01), n)

                # Compute effect and SE
                diff = treatment.mean() - control.mean()
                se = np.sqrt(treatment.var(ddof=1) / n + control.var(ddof=1) / n)

                if se == 0:
                    continue

                # TOST: two one-sided tests
                z_lower = (diff + equivalence_bound) / se
                z_upper = (equivalence_bound - diff) / se
                p_lower = 1 - stats.norm.cdf(z_lower)
                p_upper = 1 - stats.norm.cdf(z_upper)
                tost_p = max(p_lower, p_upper)

                if tost_p < alpha:
                    equivalence_count += 1
                sims_run += 1

            power = equivalence_count / sims_run if sims_run > 0 else 0.0
            se = np.sqrt(power * (1 - power) / sims_run) if sims_run > 0 else 0.0
            results.append({
                "sample_size": n,
                "power": power,
                "se": se,
                "nsim": sims_run,
            })

        return pd.DataFrame(results)
```

You will also need to add the import at the top of `power_sim.py` if not already present:

```python
from scipy import stats
```

And make sure `log_and_raise_error` is imported (check existing imports in the file).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_equivalence.py::TestPowerTOST -v`
Expected: All PASS

- [ ] **Step 5: Run ruff and commit**

```bash
uv run ruff format experiment_utils/power_sim.py tests/test_equivalence.py
uv run ruff check experiment_utils/power_sim.py tests/test_equivalence.py
git add experiment_utils/power_sim.py tests/test_equivalence.py
git commit -m "feat: add power_tost() to PowerSim for equivalence power analysis"
```

---

### Task 9: Run Full Test Suite and Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run the complete test suite**

Run: `uv run pytest tests/ -v --timeout=120`
Expected: All PASS. No regressions.

- [ ] **Step 2: Run ruff on all modified files**

```bash
uv run ruff format experiment_utils/ tests/test_equivalence.py
uv run ruff check experiment_utils/ tests/test_equivalence.py
```

Expected: No issues

- [ ] **Step 3: Verify the full public API works end-to-end**

Run a quick smoke test:

```bash
uv run python -c "
from experiment_utils import ExperimentAnalyzer, PowerSim, plot_equivalence
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 500
data = pd.DataFrame({
    'treatment': [0]*n + [1]*n,
    'outcome': np.concatenate([rng.normal(10,2,n), rng.normal(10.1,2,n)]),
    'experiment': 'exp1',
})

# Test equivalence
a = ExperimentAnalyzer(data=data, outcomes=['outcome'], treatment_col='treatment', experiment_identifier=['experiment'])
a.get_effects()
a.test_equivalence(absolute_bound=1.0)
print('Conclusion:', a.results['eq_conclusion'].iloc[0])
print('TOST p-value:', a.results['eq_pvalue'].iloc[0])
print('90%% CI:', a.results['eq_ci_lower'].iloc[0], a.results['eq_ci_upper'].iloc[0])
print('Cohen d:', a.results['eq_cohens_d'].iloc[0])

# Test power
ps = PowerSim(metric='average', nsim=100)
pw = ps.power_tost(sample_sizes=[100, 500], equivalence_bound=1.0, pooled_sd=2.0)
print('Power results:')
print(pw)
print('All checks passed!')
"
```

Expected: Output showing equivalence conclusion, p-values, CIs, and power results without errors.

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git status
# Only commit if there are changes
git commit -m "fix: address issues found during final verification"
```
