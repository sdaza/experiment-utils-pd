[![ci](https://github.com/sdaza/experiment-utils-pd/actions/workflows/ci.yaml/badge.svg)](https://github.com/sdaza/experiment-utils-pd/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/experiment-utils-pd.svg)](https://pypi.org/project/experiment-utils-pd/)

# Experiment Utils

A comprehensive Python package for designing, analyzing, and validating experiments with advanced causal inference capabilities.

## Features

- **Experiment Analysis**: Estimate treatment effects with multiple adjustment methods (covariate balancing, regression, IV, AIPW)
- **Multiple Outcome Models**: OLS, logistic, Poisson, negative binomial, and Cox proportional hazards
- **Doubly Robust Estimation**: Augmented IPW (AIPW) for OLS, logistic, Poisson, and negative binomial models
- **Survival Analysis**: Cox proportional hazards with IPW and regression adjustment
- **Covariate Balance**: Check and visualize balance between treatment groups
- **Marginal Effects**: Average marginal effects for GLMs (probability change, count change)
- **Overlap Weighting & Trimming**: Overlap weights (ATO) and propensity score trimming for robust handling of limited common support
- **Bootstrap Inference**: Robust confidence intervals and p-values via bootstrap resampling
- **Multiple Comparison Correction**: Family-wise error rate control (Bonferroni, Holm, Sidak, FDR)
- **Effect Visualization**: Cleveland dot plots of treatment effects across experiments, with percentage-point scaling, combined absolute/relative annotations, optional meta-analysis pooling, magnitude sorting, and grouping by any experiment column
- **Power Analysis**: Calculate statistical power and find optimal sample sizes
- **Retrodesign Analysis**: Assess reliability of study designs (Type S/M errors)
- **Random Assignment**: Generate balanced treatment assignments with stratification

## Table of Contents

- [Experiment Utils](#experiment-utils)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [From PyPI (Recommended)](#from-pypi-recommended)
    - [From GitHub (Latest Development Version)](#from-github-latest-development-version)
  - [Quick Start](#quick-start)
  - [User Guide](#user-guide)
    - [Basic Experiment Analysis](#basic-experiment-analysis)
    - [Covariate Parameters](#covariate-parameters)
    - [Checking Covariate Balance](#checking-covariate-balance)
    - [Covariate Adjustment Methods](#covariate-adjustment-methods)
    - [Outcome Models](#outcome-models)
    - [Ratio Metrics (Delta Method)](#ratio-metrics-delta-method)
    - [Survival Analysis (Cox Models)](#survival-analysis-cox-models)
    - [Bootstrap Inference](#bootstrap-inference)
    - [Multiple Experiments](#multiple-experiments)
    - [Categorical Treatment Variables](#categorical-treatment-variables)
    - [Instrumental Variables (IV)](#instrumental-variables-iv)
    - [Multiple Comparison Adjustments](#multiple-comparison-adjustments)
    - [Non-Inferiority Testing](#non-inferiority-testing)
    - [Combining Effects (Meta-Analysis)](#combining-effects-meta-analysis)
    - [Visualizing Effects](#visualizing-effects)
    - [Retrodesign Analysis](#retrodesign-analysis)
  - [Power Analysis](#power-analysis)
    - [Calculate Power](#calculate-power)
    - [Power from Real Data](#power-from-real-data)
    - [Grid Power Simulation](#grid-power-simulation)
    - [Find Sample Size](#find-sample-size)
    - [Simulate Retrodesign](#simulate-retrodesign)
  - [Utilities](#utilities)
    - [Balanced Random Assignment](#balanced-random-assignment)
    - [Standalone Balance Checker](#standalone-balance-checker)
    - [Advanced Topics](#advanced-topics)
        - [Covariate Adjustment Methods](#covariate-adjustment-methods)  
        - [Outcome Models](#outcome-models)  
        - [Survival Analysis (Cox Models)](#survival-analysis-cox-models)  
        - [When to Use Different Adjustment Methods](#when-to-use-different-adjustment-methods)  
        - [Non-Collapsibility of Hazard and Odds Ratios](#non-collapsibility-of-hazard-and-odds-ratios)  
        - [Handling Missing Data](#handling-missing-data)
        - [Best Practices](#best-practices)
        - [Common Workflows](#common-workflows)
  - [Contributing](#contributing)
  - [License](#license)
  - [Citation](#citation)

## Installation

### From PyPI (Recommended)

```bash
pip install experiment-utils-pd
```

### From GitHub (Latest Development Version)

```bash
pip install git+https://github.com/sdaza/experiment-utils-pd.git
```

## Quick Start

All main classes and standalone functions are available directly from the package:

```python
from experiment_utils import ExperimentAnalyzer, PowerSim
from experiment_utils import balanced_random_assignment, check_covariate_balance
from experiment_utils import plot_effects, plot_power
```

Here's a complete example analyzing an A/B test with covariate adjustment:

```python
import pandas as pd
import numpy as np
from experiment_utils import ExperimentAnalyzer

# Create sample experiment data
np.random.seed(42)
df = pd.DataFrame({
    "user_id": range(1000),
    "treatment": np.random.choice([0, 1], 1000),
    "conversion": np.random.binomial(1, 0.15, 1000),
    "revenue": np.random.normal(50, 20, 1000),
    "age": np.random.normal(35, 10, 1000),
    "is_member": np.random.choice([0, 1], 1000),
})

# Initialize analyzer
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion", "revenue"],
    balance_covariates=["age", "is_member"],  # balance checking
    adjustment="balance",
    balance_method="ps-logistic",
)

# Estimate treatment effects
analyzer.get_effects()

# View results
results = analyzer.results
print(results[["outcome", "absolute_effect", "relative_effect", 
               "pvalue", "stat_significance"]])

# Balance is automatically calculated when covariates are provided
balance = analyzer.balance
print(f"\nBalance: {balance['balance_flag'].mean():.1%} of covariates balanced")
```

Output:
```
       outcome  absolute_effect  relative_effect   pvalue stat_significance
0   conversion           0.0234           0.1623   0.0456                 1
1      revenue           2.1450           0.0429   0.1234                 0

Balance: 100.0% of covariates balanced
```

## User Guide

### Basic Experiment Analysis

Analyze a simple A/B test without covariate adjustment:

```python
from experiment_utils import ExperimentAnalyzer

# Simple analysis (no covariates)
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
)

analyzer.get_effects()
print(analyzer.results)
```

**Key columns in results:**
- `outcome`: Outcome variable name
- `absolute_effect`: Treatment effect (treatment - control mean)
- `relative_effect`: Lift (absolute_effect / control_mean)
- `standard_error`: Standard error of the effect
- `pvalue`: P-value for hypothesis test
- `stat_significance`: 1 if significant at alpha level, 0 otherwise
- `abs_effect_lower/upper`: Confidence interval bounds (absolute)
- `rel_effect_lower/upper`: Confidence interval bounds (relative)

### Covariate Parameters

Three covariate parameters control balance checking and regression adjustment. Each can be specified independently and they can overlap freely — any covariate appearing in any list is automatically included in the balance table.

| Parameter | Role | Balance checked? | In regression formula? |
|---|---|---|---|
| `balance_covariates` | Balance checking only | Yes | No |
| `regression_covariates` | Regression main effects | Yes | Yes (main effects) |
| `interaction_covariates` | CUPED / Lin interactions | Yes | Yes (`z_col + treatment:z_col`) |

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue"],
    balance_covariates=["region"],           # balance table only
    regression_covariates=["age", "tenure"], # OLS main effects + balance
    interaction_covariates=["pre_revenue"],  # CUPED variance reduction + balance
)

analyzer.get_effects()

# Balance table covers all three lists
print(analyzer.balance[["covariate", "smd", "balance_flag"]])
```

> `covariates` is still accepted as a deprecated alias for `balance_covariates`.

### Checking Covariate Balance

**Balance is automatically calculated** when you provide any covariates and run `get_effects()`:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
    balance_covariates=["age", "income", "region"],  # Can include categorical
)

analyzer.get_effects()

# Balance is automatically available
balance = analyzer.balance
print(balance[["covariate", "smd", "balance_flag"]])
print(f"\nBalanced: {balance['balance_flag'].mean():.1%}")

# Identify imbalanced covariates
imbalanced = balance[balance["balance_flag"] == 0]
if not imbalanced.empty:
    print(f"Imbalanced: {imbalanced['covariate'].tolist()}")
```

**Check balance independently** (optional, before running `get_effects()` or with custom parameters):

```python
# Check balance with different threshold
balance_strict = analyzer.check_balance(threshold=0.05)
```

**Balance metrics explained:**
- `smd`: Standardized Mean Difference (|SMD| < 0.1 indicates good balance)
- `balance_flag`: 1 if balanced, 0 if imbalanced
- `mean_treated/control`: Group means for the covariate

### Covariate Adjustment Methods

When treatment and control groups differ on covariates, adjust for bias:

**Option 1: Propensity Score Weighting (Recommended)**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion", "revenue"],
    balance_covariates=["age", "income", "is_member"],
    adjustment="balance",
    balance_method="ps-logistic",  # Logistic regression for propensity scores
    estimand="ATT",  # Average Treatment Effect on Treated
)

analyzer.get_effects()

# Check post-adjustment balance
print(analyzer.adjusted_balance)

# Retrieve weights for transparency
weights_df = analyzer.weights
print(weights_df.head())
```

**Available methods:**
- `ps-logistic`: Propensity score via logistic regression (fast, interpretable)
- `ps-xgboost`: Propensity score via XGBoost (flexible, non-linear)
- `entropy`: Entropy balancing (exact moment matching)

Target effects:

    ATT: Average Treatment Effect on Treated (most common)
    ATE: Average Treatment Effect (entire population)
    ATC: Average Treatment Effect on Control
    ATO: Average Treatment Effect for the Overlap population (overlap weights — see below)

**Option 2: Regression Adjustment**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
    regression_covariates=["age", "income"],
    adjustment=None,  # No weighting, just regression
)

analyzer.get_effects()
```

**Option 3: CUPED / Interaction Adjustment**

Add pre-experiment metrics as treatment interactions (Lin 2013 estimator). Each covariate is standardized and entered as `z_col + treatment:z_col`. This reduces variance without changing the point estimate interpretation:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue"],
    interaction_covariates=["pre_revenue", "pre_orders"],
)

analyzer.get_effects()
# adjustment column in results will show "regression+interactions"
```

**Option 4: IPW + Regression (Combined)**

Use both propensity score weighting and regression covariates for extra robustness:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion", "revenue"],
    balance_covariates=["age", "income", "is_member"],
    adjustment="balance",
    regression_covariates=["age", "income"],
    estimand="ATE",
)

analyzer.get_effects()
```

**Option 5: Doubly Robust / AIPW**

Augmented Inverse Probability Weighting is consistent if either the propensity score model or the outcome model is correctly specified. Available for OLS, logistic, Poisson, and negative binomial models:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue"],
    balance_covariates=["age", "income", "is_member"],
    adjustment="aipw",
    estimand="ATE",
)

analyzer.get_effects()

# AIPW results include influence-function based standard errors
print(analyzer.results[["outcome", "absolute_effect", "standard_error", "pvalue"]])
```

AIPW works by fitting separate outcome models for treated and control groups, predicting potential outcomes for all units, and combining them with IPW via the augmented influence function. Standard errors are derived from the influence function, making them robust without requiring bootstrap.

> **Note**: AIPW is not supported for Cox survival models due to the complexity of survival-specific doubly robust methods. For Cox models, use IPW + Regression instead.

**Option 6: Overlap Weighting (ATO)**

Overlap weights (Li, Morgan & Zaslavsky 2018) naturally downweight units with extreme propensity scores — treated units receive weight `(1 - ps)` and control units receive weight `ps`. Units near `ps = 0.5` (the region of maximum overlap) receive the highest weight. No trimming threshold is required.

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue"],
    balance_covariates=["age", "income"],
    adjustment="balance",
    balance_method="ps-logistic",  # or "ps-xgboost"
    estimand="ATO",                # overlap weights
)

analyzer.get_effects()
```

> **Note**: ATO is only supported with `balance_method="ps-logistic"` or `"ps-xgboost"`. It is not compatible with `"entropy"`.

**Option 7: Propensity Score Trimming**

Trimming drops units with propensity scores outside `[trim_ps_lower, trim_ps_upper]` and recomputes weights on the remaining sample. This is useful as a robustness check when overlap is already reasonable but you want to restrict to the region where PS estimation is reliable.

```python
# Always trim to [0.1, 0.9]
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue"],
    balance_covariates=["age", "income"],
    adjustment="balance",
    trim_ps=True,
    trim_ps_lower=0.1,  # default
    trim_ps_upper=0.9,  # default
)

# Trim only when overlap is good (overlap coefficient >= threshold)
analyzer = ExperimentAnalyzer(
    ...
    trim_ps=True,
    trim_overlap_threshold=0.8,  # skip trimming if overlap < 0.8
    assess_overlap=True,
)

analyzer.get_effects()

# trimmed_units column shows how many units were dropped
print(analyzer.results[["outcome", "absolute_effect", "trimmed_units"]])
```

**Choosing between overlap weights and trimming:**

| | Overlap weights (`ATO`) | Trimming |
|---|---|---|
| Mechanism | Continuously downweights extreme-PS units | Drops units outside threshold |
| Threshold required | No | Yes (`trim_ps_lower`, `trim_ps_upper`) |
| Changes `n` | No | Yes |
| Estimand | ATO (overlap population) | ATT/ATE/ATC on trimmed sample |
| When overlap is poor | Handles gracefully | May drop many units |
| Use as robustness check | Yes | Yes |

### Outcome Models

By default, all outcomes are analyzed with OLS. Use `outcome_models` to specify different model types:

**Logistic regression (binary outcomes)**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["converted", "churned"],
    outcome_models="logistic",  # Apply to all outcomes
    balance_covariates=["age", "tenure"],
)

analyzer.get_effects()

# By default, results report marginal effects (probability change in percentage points)
# Use compute_marginal_effects=False for odds ratios instead
```

**Poisson / Negative binomial (count outcomes)**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["orders", "page_views"],
    outcome_models="poisson",  # or "negative_binomial" for overdispersed counts
    balance_covariates=["age", "tenure"],
)

analyzer.get_effects()

# Results report change in expected count (marginal effects) by default
# Use compute_marginal_effects=False for rate ratios
```

**Mixed models per outcome**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue", "converted", "orders"],
    outcome_models={
        "revenue": "ols",
        "converted": "logistic",
        "orders": ["poisson", "negative_binomial"],  # Compare both
    },
    balance_covariates=["age"],
)

analyzer.get_effects()

# Results include model_type column to distinguish
print(analyzer.results[["outcome", "model_type", "absolute_effect", "pvalue"]])
```

**Marginal effects options**

```python
# Average Marginal Effect (default) - recommended
analyzer = ExperimentAnalyzer(..., compute_marginal_effects="overall")

# Marginal Effect at the Mean
analyzer = ExperimentAnalyzer(..., compute_marginal_effects="mean")

# Odds ratios / rate ratios instead of marginal effects
analyzer = ExperimentAnalyzer(..., compute_marginal_effects=False)
```

| `compute_marginal_effects` | Logistic output | Poisson/NB output |
|---|---|---|
| `"overall"` (default) | Probability change (pp) | Change in expected count |
| `"mean"` | Probability change at mean | Count change at mean |
| `False` | Odds ratio | Rate ratio |

### Ratio Metrics (Delta Method)

Use `ratio_outcomes` for metrics where both the numerator and denominator include randomness — for example, *leads per converter* or *revenue per session*. Conditioning on the denominator (e.g., analysing only converters) introduces selection bias, so the correct approach is the **delta method linearization** (Deng et al. 2018):

```
linearized_i = numerator_i  −  R_control × denominator_i
where  R_control = mean(numerator_control) / mean(denominator_control)
```

OLS on `linearized_i` estimates the difference in population-average ratios with correct standard errors. `R_control` is computed separately for each `(treatment, control)` comparison pair, so multi-arm experiments work out of the box.

**Basic usage**

```python
import numpy as np
import pandas as pd
from experiment_utils import ExperimentAnalyzer

np.random.seed(42)
n = 20_000
treatment = np.random.choice(["control", "variant_1", "variant_2"], n)

# ~30% of users convert; converters generate ~2 leads on average
converters = np.where(
    treatment == "variant_2", np.random.binomial(1, 0.32, n),
    np.where(treatment == "variant_1", np.random.binomial(1, 0.31, n),
                                       np.random.binomial(1, 0.30, n)),
)
leads = np.where(converters == 1, np.random.poisson(2 + 0.1 * (treatment == "variant_2"), n), 0)

df = pd.DataFrame({"treatment": treatment, "converters": converters, "leads": leads})

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["converters", "leads"],           # regular outcomes
    ratio_outcomes={"leads_per_converter": ("leads", "converters")},
)

analyzer.get_effects()

cols = ["outcome", "treatment_group", "control_group",
        "control_value", "absolute_effect", "standard_error",
        "stat_significance", "effect_type"]
print(analyzer.results[cols].to_string())
```

Output:
```
               outcome treatment_group control_group  control_value  absolute_effect  standard_error  stat_significance      effect_type
0           converters       variant_1       control       0.301              0.010        0.006                  1   mean_difference
1                leads       variant_1       control       0.602              0.046        0.017                  1   mean_difference
2  leads_per_converter       variant_1       control       1.977              0.022        0.011                  1  ratio_difference
3           converters       variant_2       control       0.301              0.019        0.006                  1   mean_difference
4                leads       variant_2       control       0.602              0.076        0.017                  1   mean_difference
5  leads_per_converter       variant_2       control       1.977              0.037        0.011                  1  ratio_difference
6           converters       variant_2     variant_1       0.311              0.009        0.007                  0   mean_difference
7                leads       variant_2     variant_1       0.647              0.030        0.017                  0   mean_difference
8  leads_per_converter       variant_2     variant_1       2.049              0.014        0.012                  0  ratio_difference
```

The `control_value` column shows `R_control` (the control arm's ratio), and `absolute_effect` is the estimated difference in ratios. Results integrate normally with `plot_effects`, `calculate_retrodesign`, and MCP correction.

**With bootstrap**

Bootstrap correctly re-estimates `R_control` on each resample, so standard errors fully capture the uncertainty in the ratio baseline:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["leads"],
    ratio_outcomes={"leads_per_converter": ("leads", "converters")},
    bootstrap=True,
    bootstrap_iterations=1000,
    bootstrap_seed=42,
)

analyzer.get_effects()
print(analyzer.results[["outcome", "absolute_effect", "standard_error",
                         "abs_effect_lower", "abs_effect_upper"]])
```

> **Why not just subset to converters?** Analysing only users who converted conditions on a post-randomisation variable, creating selection bias. The delta method preserves the full randomised sample and gives an unbiased estimate of the causal effect on the population-average ratio.

**Key result columns for ratio outcomes**

| Column | Meaning |
|---|---|
| `control_value` | `R_control = mean(num_control) / mean(den_control)` for this comparison |
| `absolute_effect` | Estimated difference in population-average ratios |
| `relative_effect` | `absolute_effect / control_value` |
| `effect_type` | `"ratio_difference"` |

### Survival Analysis (Cox Models)

Analyze time-to-event outcomes using Cox proportional hazards:

```python
from experiment_utils import ExperimentAnalyzer

# Specify Cox outcomes as tuples: (time_col, event_col)
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=[("time_to_event", "event_occurred")],
    outcome_models="cox",
    balance_covariates=["age", "income"],
)

analyzer.get_effects()

# Results report log(HR) as absolute_effect and HR as relative_effect
print(analyzer.results[["outcome", "absolute_effect", "relative_effect", "pvalue"]])
```

**Cox with regression adjustment**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=[("survival_time", "died")],
    outcome_models="cox",
    regression_covariates=["age", "comorbidity_score"],
)

analyzer.get_effects()
```

**Cox with IPW + Regression (recommended for confounded data)**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=[("survival_time", "died")],
    outcome_models="cox",
    balance_covariates=["age", "comorbidity_score"],
    adjustment="balance",
    regression_covariates=["age", "comorbidity_score"],
    estimand="ATE",
)

analyzer.get_effects()
```

> **Note**: IPW alone for Cox models estimates the marginal hazard ratio, which differs from the conditional HR due to non-collapsibility. The package will warn you if you use IPW without regression covariates. See [Non-Collapsibility](#non-collapsibility-of-hazard-and-odds-ratios) for details.

**Alternative: separate event_col parameter**

```python
# Equivalent to tuple notation
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["survival_time"],
    outcome_models="cox",
    event_col="died",  # Applies to all outcomes
)
```

**Bootstrap for survival models**

Bootstrap can be slow for Cox models with low event rates. Use `skip_bootstrap_for_survival` to fall back to robust standard errors:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=[("survival_time", "died")],
    outcome_models="cox",
    bootstrap=True,
    skip_bootstrap_for_survival=True,  # Use Cox robust SEs instead
)
```

### Bootstrap Inference

Get robust confidence intervals and p-values via bootstrapping:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
    balance_covariates=["age", "income"],
    adjustment="balance",
    bootstrap=True,
    bootstrap_iterations=2000,
    bootstrap_ci_method="percentile",
    bootstrap_seed=42,  # For reproducibility
)

analyzer.get_effects()

# Bootstrap results include robust CIs
results = analyzer.results
print(results[["outcome", "absolute_effect", "abs_effect_lower", 
               "abs_effect_upper", "inference_method"]])
```

**When to use bootstrap:**
- Small sample sizes
- Non-normal distributions
- Skepticism about asymptotic assumptions
- Want robust, distribution-free inference

### Multiple Experiments

Analyze multiple experiments simultaneously:

```python
# Data with multiple experiments
df = pd.DataFrame({
    "experiment": ["exp_A", "exp_A", "exp_B", "exp_B"] * 100,
    "treatment": [0, 1, 0, 1] * 100,
    "outcome": np.random.randn(400),
    "age": np.random.normal(35, 10, 400),
})

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["outcome"],
    experiment_identifier="experiment",  # Group by experiment
    balance_covariates=["age"],
)

analyzer.get_effects()

# Results include experiment column
results = analyzer.results
print(results.groupby("experiment")[["absolute_effect", "pvalue"]].first())

# Balance per experiment (automatically calculated)
balance = analyzer.balance
print(balance.groupby("experiment")["balance_flag"].mean())
```

### Categorical Treatment Variables

Compare multiple treatment variants:

```python
df = pd.DataFrame({
    "treatment": np.random.choice(["control", "variant_A", "variant_B"], 1000),
    "outcome": np.random.randn(1000),
})

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["outcome"],
)

analyzer.get_effects()

# Results show all pairwise comparisons
results = analyzer.results
print(results[["treatment_group", "control_group", "absolute_effect", "pvalue"]])
```

### Instrumental Variables (IV)

When treatment assignment is confounded (e.g., non-compliance in an experiment), use an instrument -- a variable that affects treatment receipt but only affects the outcome through treatment:

```python
import numpy as np
import pandas as pd
from experiment_utils import ExperimentAnalyzer

# Simulate encouragement design with non-compliance
np.random.seed(42)
n = 5000
Z = np.random.binomial(1, 0.5, n)            # Random encouragement (instrument)
U = np.random.normal(0, 1, n)                 # Unobserved confounder
D = np.random.binomial(1, 1 / (1 + np.exp(-(-1 + 0.5 * U + 2.5 * Z))))  # Actual treatment (confounded)
Y = 2.0 * D + 1.0 * U + np.random.normal(0, 1, n)  # Outcome (true LATE = 2.0)

df = pd.DataFrame({"encouragement": Z, "treatment": D, "outcome": Y})

# IV estimation using encouragement as instrument for treatment
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["outcome"],
    instrument_col="encouragement",
    adjustment="IV",
)

analyzer.get_effects()
print(analyzer.results[["outcome", "absolute_effect", "standard_error", "pvalue"]])
```

**IV with covariates:**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["outcome"],
    instrument_col="encouragement",
    adjustment="IV",
    balance_covariates=["age", "region"],  # Balance checked on instrument
)

analyzer.get_effects()
```

**Key assumptions for valid IV estimation:**
- **Relevance**: The instrument must be correlated with treatment (check first-stage F-statistic)
- **Exclusion restriction**: The instrument affects the outcome *only* through treatment
- **Independence**: The instrument is independent of unobserved confounders (holds by design in randomized encouragement)

> **Note**: IV estimation is only supported for OLS outcome models. For other model types (logistic, Cox, etc.), the analyzer will fall back to unadjusted estimation with a warning.

### Multiple Comparison Adjustments

Control family-wise error rate when testing multiple hypotheses:

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion", "revenue", "retention", "engagement"],
)

analyzer.get_effects()

# Apply Bonferroni correction
analyzer.adjust_pvalues(method="bonferroni")

results = analyzer.results
print(results[["outcome", "pvalue", "pvalue_mcp", "stat_significance_mcp"]])
```

**Available methods:**
- `bonferroni`: Most conservative, controls FWER
- `holm`: Less conservative than Bonferroni, still controls FWER
- `sidak`: Similar to Bonferroni, assumes independence
- `fdr_bh`: Benjamini-Hochberg FDR control (less conservative)

### Non-Inferiority Testing

Test whether the treatment stays within an acceptable margin of control
(`test_non_inferiority`).  Given margin M, the question is: *"Is the treatment
close enough to control to be acceptable?"*

**Intuition** — given `control_value = 0.03` and `margin = 0.01`:

| Direction | Passes when … | In other words … |
|---|---|---|
| `higher_is_better` (default) | one-sided lower CI of effect > −M | treatment value is confidently above `control − margin` = 0.02 |
| `lower_is_better` | one-sided upper CI of effect < +M | treatment value is confidently below `control + margin` = 0.04 |

The test uses a one-sided normal z-test at the specified `alpha` (default 0.05).

```python
analyzer.get_effects()

# Higher-is-better (e.g. conversion rate): treatment must stay within 1 pp of control
analyzer.test_non_inferiority(absolute_margin=0.01)

# Margin as a fraction of the control mean — 10% of control rate
analyzer.test_non_inferiority(relative_margin=0.10)

# Lower-is-better (e.g. error rate, churn): treatment must not increase by more than 1 pp
analyzer.test_non_inferiority(absolute_margin=0.01, direction="lower_is_better")

results = analyzer.results
print(results[["outcome", "absolute_effect", "margin", "margin_pvalue", "within_margin"]])
```

Added columns:
- `margin` — absolute margin used for each row
- `margin_pvalue` — one-sided p-value; smaller = stronger evidence the treatment is within the margin
- `within_margin` — `True` when `margin_pvalue < alpha`

### Combining Effects (Meta-Analysis)

When you have multiple experiments or segments, pool results using fixed-effects meta-analysis or weighted averaging.

**Fixed-effects meta-analysis (`combine_effects`)**

Combines effect estimates using inverse-variance weighting, producing a pooled effect with proper standard errors:

```python
from experiment_utils import ExperimentAnalyzer

# Analyze multiple experiments
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
    experiment_identifier="experiment",
    balance_covariates=["age"],
)

analyzer.get_effects()

# Pool results across experiments using fixed-effects meta-analysis
pooled = analyzer.combine_effects(grouping_cols=["outcome"])
print(pooled[["outcome", "experiments", "absolute_effect", "standard_error", "pvalue"]])
```

**Custom grouping:**

```python
# Pool by outcome and region (e.g., combine experiments within each region)
pooled_by_region = analyzer.combine_effects(grouping_cols=["region", "outcome"])
print(pooled_by_region)
```

**Weighted average aggregation (`aggregate_effects`)**

A simpler alternative that weights by treatment group size (useful for quick summaries, but `combine_effects` provides better standard error estimates):

```python
aggregated = analyzer.aggregate_effects(grouping_cols=["outcome"])
print(aggregated[["outcome", "experiments", "absolute_effect", "pvalue"]])
```

### Visualizing Effects

`plot_effects` produces a Cleveland dot plot with confidence intervals and optional meta-analysis pooling. It is available both as a **standalone function** and as a method on `ExperimentAnalyzer`.

The two axis roles are controlled by `y`:

| `y` | Rows (y-axis) | Panels (subplots) |
|---|---|---|
| `"experiment"` *(default)* | Experiment labels | Outcomes |
| `"outcome"` | Outcomes | Experiment labels |

**Basic usage — multiple experiments, outcomes as panels (default)**

```python
analyzer.get_effects()

# show_values=True is the default — each dot is annotated with its effect value
fig = analyzer.plot_effects(title="Treatment Effects")
plt.show()
```

**Percentage points (`pct_points=True`)**

For rate/proportion outcomes, display absolute effects as percentage points instead of raw decimals (e.g. `+3.0pp` instead of `+0.030`):

```python
fig = analyzer.plot_effects(
    outcomes="converted",
    pct_points=True,
    title="Conversion Rate (pp)",
)
plt.show()
```

**Combined label — absolute (pp) + relative in one annotation**

Show both metrics on a single panel with `combine_values=True`. The x-axis label updates automatically:

```python
# "+3.0pp (+15.4%)" on the absolute panel
fig = analyzer.plot_effects(
    outcomes="converted",
    effect="absolute",
    pct_points=True,
    combine_values=True,
    title="Conversion Rate",
)
plt.show()

# "+15.4% (+3.0pp)" on the relative panel
fig = analyzer.plot_effects(
    outcomes="converted",
    effect="relative",
    pct_points=True,
    combine_values=True,
    title="Conversion Rate",
)
plt.show()
```

X-axis labels when `combine_values=True`:

| `effect` | `pct_points` | x-axis label |
|---|---|---|
| `"absolute"` | `False` | `Absolute (Relative) Effect` |
| `"absolute"` | `True` | `Absolute (Relative) Effect (pp)` |
| `"relative"` | — | `Relative (Absolute) Effect` |

**Side-by-side absolute (pp) and relative panels**

```python
fig = analyzer.plot_effects(
    effect=["absolute", "relative"],
    pct_points=True,
    title="Effects — Absolute & Relative",
)
plt.show()
```

**Single experiment, multiple outcomes on the y-axis**

When you have one experiment and several outcomes, flip the axes with `y="outcome"` and customise the panel subtitle with `panel_titles`:

```python
fig = analyzer.plot_effects(
    y="outcome",
    title="My Experiment",
    panel_titles="Treatment vs Control",   # single string → same for all panels
)
plt.show()
```

**Multiple experiments, outcomes on the y-axis**

```python
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["revenue", "converted", "orders"],
    experiment_identifier=["country", "type"],
)
analyzer.get_effects()

# One panel per experiment group; rows = outcomes
fig = analyzer.plot_effects(
    y="outcome",
    panel_titles={"US | email": "US — Email", "EU | push": "EU — Push"},
)
plt.show()
```

**Standalone usage**

```python
from experiment_utils import plot_effects

fig = plot_effects(
    results=analyzer.results,
    experiment_identifier="experiment",
    alpha=0.05,
    title="Treatment Effects",
    save_path="effects.png",   # optional; supports png, pdf, svg, ...
)
plt.show()
```

**Add a pooled meta-analysis row**

```python
# Auto-compute pooled estimate (IVW of visible rows)
fig = analyzer.plot_effects(
    outcomes="revenue",
    meta_analysis=True,
    title="Revenue — with Pooled Estimate",
)
plt.show()

# Pass a pre-computed combine_effects() DataFrame
pooled = analyzer.combine_effects(grouping_cols=["outcome"])
fig = analyzer.plot_effects(meta_analysis=pooled)
plt.show()
```

**Split into one figure per group**

When `experiment_identifier` contains multiple columns (e.g. `["country", "type"]`), `group_by` produces one figure per unique value. Row labels are built from the remaining identifier columns automatically.

```python
# One figure per country; rows = type
figs = analyzer.plot_effects(group_by="country", meta_analysis=True)
for fig in figs.values():
    plt.figure(fig.number)
    plt.show()

# save_path inserts the group key before the extension:
#   "effects.png" → "effects_US.png", "effects_EU.png", ...
figs = analyzer.plot_effects(group_by="country", save_path="effects.png")
```

`group_by` returns `dict[str, Figure]`; without it a single `Figure` is returned.

**Multiple comparison adjustments**

If `adjust_pvalues()` has been called before plotting, the plot automatically uses the adjusted significance column (`stat_significance_mcp`) and updates the legend label accordingly:

```python
analyzer.get_effects()
analyzer.adjust_pvalues(method="holm")

# Legend shows "Significant (holm, α=0.05)" and coloring uses adjusted p-values
fig = analyzer.plot_effects()
plt.show()
```

**Key parameters**

| Parameter | Default | Description |
|---|---|---|
| `y` | `"experiment"` | `"experiment"` — rows = experiments, panels = outcomes; `"outcome"` — rows = outcomes, panels = experiments |
| `panel_titles` | `None` | Override subplot titles: `str` (all panels) or `dict` (per-panel) |
| `outcomes` | `None` | Outcome(s) to include; `None` = all |
| `effect` | `"absolute"` | `"absolute"`, `"relative"`, or `["absolute", "relative"]` for side-by-side |
| `meta_analysis` | `None` | `True` (auto-compute IVW from visible rows), `DataFrame` (pre-computed), or `None` |
| `sort_by_magnitude` | `True` | Sort rows by `\|effect\|` descending |
| `group_by` | `None` | Column(s) to split into separate figures |
| `comparison` | `None` | `(treatment, control)` tuple or list of tuples to filter to specific comparisons |
| `title` | `None` | Figure suptitle (group value used automatically when `group_by` is set) |
| `show_zero_line` | `True` | Vertical reference line at zero |
| `show_values` | `True` | Annotate each dot with its effect value (`*` when significant) |
| `value_decimals` | auto | Decimal places for value labels. Defaults to `1` when `pct_points=True` or relative effect shown; `2` otherwise |
| `pct_points` | `False` | Multiply absolute effects by 100 for display as percentage points (pp). Updates axis label and annotations |
| `combine_values` | `False` | Append the secondary effect in parentheses to each annotation: `+3.0pp (+15.4%)` or `+15.4% (+3.0pp)`. Also updates the x-axis label |
| `panel_spacing` | `None` | Horizontal whitespace between panels (`wspace`). Try `0.4`–`0.8` when panels overlap |
| `repeat_ylabels` | `False` | Show y-axis tick labels on every panel, not only the leftmost |
| `row_labels` | `None` | Rename individual y-axis row labels. `dict` mapping auto-generated labels to display strings, e.g. `{"US \| email": "Email (US)"}` |
| `save_path` | `None` | File path to save the figure. With `group_by`, the group key is inserted before the extension: `"effects.png"` → `"effects_US.png"`, etc. |
| `figsize` | auto | `(width, height)` in inches |

### Retrodesign Analysis

Assess reliability of significant results (post-hoc power analysis):

```python
from experiment_utils import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["conversion"],
)

analyzer.get_effects()

# Calculate Type S and Type M errors assuming true effect is 0.02
retro = analyzer.calculate_retrodesign(true_effect=0.02)

print(retro[["outcome", "power", "type_s_error", "type_m_error",
             "relative_bias", "trimmed_abs_effect"]])
```

**Metrics explained:**
- `power`: Probability of detecting the assumed true effect
- `type_s_error`: Probability of wrong sign when significant (if underpowered)
- `type_m_error`: Expected exaggeration ratio (mean |observed|/|true|)
- `relative_bias`: Expected bias ratio preserving signs (mean observed/true); typically lower than `type_m_error` because wrong-sign estimates partially cancel overestimates
- `trimmed_abs_effect`: Bias-corrected effect estimate (`absolute_effect / relative_bias`); deflates the observed effect by the sign-preserving exaggeration factor to approximate the true effect



## Power Analysis

Design well-powered experiments using simulation-based power analysis.

### Calculate Power

Estimate statistical power for a given sample size:

```python
from experiment_utils import PowerSim

# Initialize power simulator for proportion metric
power_sim = PowerSim(
    metric="proportion",      # or "average" for continuous outcomes
    relative_effect=False,    # False = absolute effect, True = relative
    variants=1,               # Number of treatment variants
    nsim=1000,               # Number of simulations
    alpha=0.05,              # Significance level
    alternative="two-tailed" # or "one-tailed"
)

# Calculate power
power_result = power_sim.get_power(
    baseline=[0.10],          # Control conversion rate
    effect=[0.02],           # Absolute effect size (2pp lift)
    sample_size=[5000]       # Total sample size
)

print(f"Power: {power_result['power'].iloc[0]:.2%}")
```

**Example: Multiple variants**

```python
# Compare 2 treatments vs control
power_sim = PowerSim(metric="proportion", variants=2, nsim=1000)

power_result = power_sim.get_power(
    baseline=0.10,
    effect=[0.02, 0.03],  # Different effects for each variant
    sample_size=6000
)

print(power_result[["comparison", "power"]])
```

### Power from Real Data

When your data doesn't follow standard parametric assumptions, estimate power by bootstrapping directly from observed data using `get_power_from_data()`. Instead of generating synthetic data from a distribution, it repeatedly samples from your actual dataset and injects the specified effect:

```python
from experiment_utils import PowerSim
import pandas as pd

# Use real data for power estimation
power_sim = PowerSim(metric="average", variants=1, nsim=1000)

power_result = power_sim.get_power_from_data(
    df=historical_data,          # Your actual dataset
    metric_col="revenue",        # Column to test
    sample_size=5000,            # Sample size per group
    effect=3.0,                  # Effect to inject (absolute)
)

print(f"Power: {power_result['power'].iloc[0]:.2%}")
```

**When to use `get_power_from_data` vs `get_power`:**
- Use `get_power_from_data` when your metric has a non-standard distribution (heavy tails, skewed, zero-inflated)
- Use `get_power` for standard parametric scenarios (proportions, means, counts)

**With compliance:**

```python
# Account for 80% compliance
power_result = power_sim.get_power_from_data(
    df=historical_data,
    metric_col="revenue",
    sample_size=5000,
    effect=3.0,
    compliance=0.80,
)
```

### Grid Power Simulation

Explore power across a grid of parameter combinations using `grid_sim_power()`. This is useful for understanding how power varies with sample size, effect size, and baseline rates:

```python
from experiment_utils import PowerSim

power_sim = PowerSim(metric="proportion", variants=1, nsim=1000)

# Simulate power across a grid of scenarios
grid_results = power_sim.grid_sim_power(
    baseline_rates=[0.05, 0.10, 0.15],
    effects=[0.02, 0.03, 0.05],
    sample_sizes=[1000, 2000, 5000, 10000],
    plot=True,  # Generate power curves
)

print(grid_results.head())
```

**With multiple variants and custom compliance:**

```python
power_sim = PowerSim(metric="average", variants=2, nsim=1000)

grid_results = power_sim.grid_sim_power(
    baseline_rates=[50.0],
    effects=[2.0, 5.0],
    sample_sizes=[500, 1000, 2000, 5000],
    standard_deviations=[[20.0]],
    compliances=[[0.8]],
    threads=4,        # Parallelize across scenarios
    plot=True,
)
```

The output DataFrame includes all input parameters alongside the estimated power for each comparison, making it easy to filter and compare scenarios.

### Find Sample Size

Find the minimum sample size needed to achieve target power:

```python
from experiment_utils import PowerSim

power_sim = PowerSim(metric="proportion", variants=1, nsim=1000)

# Find sample size for 80% power
sample_result = power_sim.find_sample_size(
    power=0.80,
    baseline=0.10,
    effect=0.02
)

print(f"Required sample size: {sample_result['total_sample_size'].iloc[0]:,.0f}")
print(f"Achieved power: {sample_result['achieved_power_by_comparison'].iloc[0]:.2%}")
```

**Different power targets per comparison:**

```python
# Primary outcome needs 90%, secondary needs 80%
power_sim = PowerSim(metric="proportion", variants=2, nsim=1000)

sample_result = power_sim.find_sample_size(
    power={(0,1): 0.90, (0,2): 0.80},
    baseline=0.10,
    effect=[0.05, 0.03]
)

print(sample_result[["comparison", "sample_size_by_group", "achieved_power"]])
```

**Optimize allocation ratio:**

```python
# Find optimal allocation to minimize total sample size
sample_result = power_sim.find_sample_size(
    power=0.80,
    baseline=0.10,
    effect=0.05,
    optimize_allocation=True
)

print(f"Optimal allocation: {sample_result['allocation_ratio'].iloc[0]}")
print(f"Total sample size: {sample_result['total_sample_size'].iloc[0]:,.0f}")
```

**Custom allocation:**

```python
# 30% control, 70% treatment
sample_result = power_sim.find_sample_size(
    power=0.80,
    baseline=0.10,
    effect=0.02,
    allocation_ratio=[0.3, 0.7]
)
```

### Simulate Retrodesign

Prospective analysis of Type S (sign) and Type M (magnitude) errors:

```python
from experiment_utils import PowerSim

power_sim = PowerSim(metric="proportion", variants=1, nsim=5000)

# Simulate underpowered study
retro = power_sim.simulate_retrodesign(
    true_effect=0.02,
    sample_size=500,
    baseline=0.10
)

print(f"Power: {retro['power'].iloc[0]:.2%}")
print(f"Type S Error: {retro['type_s_error'].iloc[0]:.2%}")
print(f"Exaggeration Ratio: {retro['exaggeration_ratio'].iloc[0]:.2f}x")
print(f"Relative Bias: {retro['relative_bias'].iloc[0]:.2f}x")
```

**Understanding retrodesign metrics:**

| Metric | Description |
|--------|-------------|
| `power` | Probability of detecting the true effect |
| `type_s_error` | Probability of getting wrong sign when significant |
| `exaggeration_ratio` | Expected overestimation (mean &#124;observed&#124;/&#124;true&#124;) |
| `relative_bias` | Expected bias preserving signs (mean observed/true) <br> Lower than exaggeration_ratio because Type S errors partially cancel out overestimates |
| `median_significant_effect` | Median effect among significant results |
| `prop_overestimate` | % of significant results that overestimate |

**Compare power scenarios:**

```python
# Low power scenario
retro_low = power_sim.simulate_retrodesign(
    true_effect=0.02, sample_size=500, baseline=0.10
)

# High power scenario
retro_high = power_sim.simulate_retrodesign(
    true_effect=0.02, sample_size=5000, baseline=0.10
)

print(f"Low power - Exaggeration: {retro_low['exaggeration_ratio'].iloc[0]:.2f}x, "
      f"Relative bias: {retro_low['relative_bias'].iloc[0]:.2f}x")
print(f"High power - Exaggeration: {retro_high['exaggeration_ratio'].iloc[0]:.2f}x, "
      f"Relative bias: {retro_high['relative_bias'].iloc[0]:.2f}x")
```

**Multiple variants:**

```python
power_sim = PowerSim(metric="proportion", variants=3, nsim=5000)

retro = power_sim.simulate_retrodesign(
    true_effect=[0.02, 0.03, 0.04],  # Different effects per variant
    sample_size=1000,
    baseline=0.10,
    comparisons=[(0, 1), (0, 2)]
)

print(retro[["comparison", "power", "type_s_error", "exaggeration_ratio", "relative_bias"]])
```

## Utilities

### Balanced Random Assignment

Generate balanced treatment assignments with optional block randomization.
Variant distribution and, when covariates are provided, a covariate balance
summary are always printed.

```python
from experiment_utils import balanced_random_assignment
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
users = pd.DataFrame({
    "user_id": range(1000),
    "age_group": np.random.choice(["18-25", "26-35", "36-45", "46+"], 1000),
    "region": np.random.choice(["North", "South", "East", "West"], 1000),
    "age": np.random.normal(35, 10, 1000),
})

# Simple 50/50 split — prints variant distribution automatically
users["treatment"] = balanced_random_assignment(
    users,
    allocation_ratio=0.5,
    seed=42
)
```

**Block randomization (stratify within subgroups):**

```python
# Stratify by age_group and region; check balance on the same variables
users["treatment_stratified"] = balanced_random_assignment(
    users,
    allocation_ratio=0.5,
    stratification_covariates=["age_group", "region"],
    seed=42
)
```

Warns automatically if any stratification category has low prevalence (< 5 % by
default) and suggests not blocking on that variable.

**Check balance on additional covariates:**

```python
# Stratify by region; check balance on a broader set
users["treatment_stratified"] = balanced_random_assignment(
    users,
    allocation_ratio=0.5,
    stratification_covariates=["region"],
    balance_covariates=["age_group", "region", "age"],
    seed=42
)
```

**Multiple variants:**

```python
# Three variants with equal allocation
users["assignment"] = balanced_random_assignment(
    users,
    variants=["control", "variant_A", "variant_B"]
)

# Custom allocation ratios with stratification
users["assignment_custom"] = balanced_random_assignment(
    users,
    variants=["control", "variant_A", "variant_B"],
    allocation_ratio={"control": 0.5, "variant_A": 0.3, "variant_B": 0.2},
    stratification_covariates=["age_group"]
)
```

**Key parameters:**
- `allocation_ratio`: Float (binary) or dict (multiple variants)
- `stratification_covariates`: Columns to block-randomize on (continuous vars are auto-binned)
- `balance_covariates`: Columns to check balance for after assignment (defaults to `stratification_covariates`)
- `smd_threshold`: SMD threshold for balance flag (default `0.1`)
- `min_stratum_pct`: Minimum category prevalence before a stratification warning is raised (default `0.05`)
- `min_stratum_n`: Minimum absolute category count before a stratification warning is raised (default `10`)
- `seed`: Random seed for reproducibility

### Standalone Balance Checker

Check covariate balance on any dataset without using ExperimentAnalyzer:

```python
from experiment_utils import check_covariate_balance
import pandas as pd
import numpy as np

# Create sample data with imbalance
np.random.seed(42)
n_treatment = 300
n_control = 200

df = pd.concat([
    pd.DataFrame({
        "treatment": [1] * n_treatment,
        "age": np.random.normal(40, 10, n_treatment),      # Older in treatment
        "income": np.random.normal(60000, 15000, n_treatment),  # Higher income
    }),
    pd.DataFrame({
        "treatment": [0] * n_control,
        "age": np.random.normal(30, 10, n_control),         # Younger in control
        "income": np.random.normal(45000, 15000, n_control),    # Lower income
    })
])

# Check balance
balance = check_covariate_balance(
    data=df,
    treatment_col="treatment",
    covariates=["age", "income"],
    threshold=0.1  # SMD threshold
)

print(balance)
```

Output:
```
  covariate  mean_treated  mean_control       smd  balance_flag
0       age         40.23         30.15  1.012345             0
1    income      59823.45      45234.12  0.923456             0
```

**With categorical variables:**

```python
df["region"] = np.random.choice(["North", "South", "East", "West"], len(df))

balance = check_covariate_balance(
    data=df,
    treatment_col="treatment",
    covariates=["age", "income", "region"],  # Automatic categorical detection
    threshold=0.1
)

# Region will be expanded to dummy variables
print(balance[balance["covariate"].str.contains("region")])
```

**Use cases:**
- Pre-experiment: Check if randomization worked
- Post-assignment: Validate treatment assignment quality
- Observational data: Assess comparability before adjustment
- Research: Standalone balance analysis for publications

## Advanced Topics

### When to Use Different Adjustment Methods

| Method | `adjustment` | Covariate params | Best for |
|---|---|---|---|
| No adjustment | `None` | none | Well-randomized experiments |
| Regression | `None` | `regression_covariates=["x1","x2"]` | Variance reduction |
| CUPED | `None` | `interaction_covariates=["pre_x"]` | Variance reduction with pre-experiment data |
| IPW | `"balance"` | `balance_covariates=["x1","x2"]` | Many covariates, non-linear confounding |
| IPW + Regression | `"balance"` | both `balance_covariates` and `regression_covariates` | Extra robustness, survival models |
| Overlap weights (ATO) | `"balance"` + `estimand="ATO"` | `balance_covariates=["x1","x2"]` | Poor or moderate overlap, no threshold needed |
| Trimming | `"balance"` + `trim_ps=True` | `balance_covariates=["x1","x2"]` | Robustness check, restrict to overlap region |
| AIPW (doubly robust) | `"aipw"` | `balance_covariates=["x1","x2"]` | Best protection against misspecification |
| IV | `"IV"` | `balance_covariates` optional | Non-compliance, endogenous treatment (requires `instrument_col`) |

**Choosing a balance method:**
- `ps-logistic`: Default, fast, interpretable
- `ps-xgboost`: Non-linear relationships, complex interactions
- `entropy`: Exact moment matching, but can be unstable with many covariates

**Choosing an outcome model:**

| Outcome type | Parameter |
|---|---|
| Continuous (revenue, time) | `outcome_models="ols"` (default) |
| Binary (converted, churned) | `outcome_models="logistic"` |
| Count (orders, clicks) | `outcome_models="poisson"` |
| Overdispersed count | `outcome_models="negative_binomial"` |
| Time-to-event | `outcome_models="cox"` |
| Ratio (leads/converter, revenue/session) | `ratio_outcomes={"name": ("num_col", "den_col")}` |

### Non-Collapsibility of Hazard and Odds Ratios

When using IPW without regression covariates for Cox or logistic models, the estimated effect may differ from the conditional effect even with perfect covariate balancing. This is not a bug -- it reflects a fundamental property called **non-collapsibility**.

**What happens**: IPW creates a pseudo-population where treatment is independent of covariates, then fits a model without covariates. This estimates the **marginal** effect (population-average). For non-collapsible measures like hazard ratios and odds ratios, the marginal effect differs from the conditional effect.

**When it matters**: The gap increases with stronger covariate effects on the outcome. For Cox models the effect is typically larger than for logistic models.

**Recommendations**:
- For Cox models: use **regression adjustment** or **IPW + Regression** to recover the conditional HR
- For logistic models: the default marginal effects output (probability change) is collapsible, so this mainly affects odds ratios (`compute_marginal_effects=False`)
- For OLS: no issue (mean differences are collapsible)
- AIPW estimates are on the marginal scale but are doubly robust

The package warns when IPW is used without regression covariates for Cox models.

### Handling Missing Data

The package handles missing data automatically:

- **Treatment variable**: Rows with missing treatment are dropped (logged as warning)
- **Categorical covariates**: Missing values become explicit "Missing" category
- **Numeric covariates**: Mean imputation
- **Binary covariates**: Mode imputation

```python
analyzer = ExperimentAnalyzer(
    data=df,  # Can contain missing values
    treatment_col="treatment",
    outcomes=["conversion"],
    balance_covariates=["age", "region"],
)
# Missing data is handled automatically
analyzer.get_effects()
```

### Best Practices

**1. Always check balance:**

```python
analyzer = ExperimentAnalyzer(data=df, treatment_col="treatment",
                              outcomes=["conversion"],
                              balance_covariates=["age", "income"])

analyzer.get_effects()

# Check balance from results
balance = analyzer.balance
if balance["balance_flag"].mean() < 0.8:  # <80% balanced
    print("Consider rerunning with covariate adjustment")
```

**2. Use bootstrap for small samples:**

```python
if len(df) < 500:
    analyzer = ExperimentAnalyzer(..., bootstrap=True, bootstrap_iterations=2000)
```

**3. Apply multiple comparison correction:**

```python
# Always correct when testing multiple outcomes/experiments
analyzer.get_effects()
analyzer.adjust_pvalues(method="holm")  # Less conservative than Bonferroni
```

**4. Report both absolute and relative effects:**

```python
results = analyzer.results
print(results[["outcome", "absolute_effect", "relative_effect", 
               "abs_effect_lower", "abs_effect_upper"]])
```

**5. Check sensitivity with retrodesign:**

```python
# After finding significant result, check reliability
retro = analyzer.calculate_retrodesign(true_effect=0.01)
if retro["type_m_error"].iloc[0] > 2:
    print("Warning: Results may be exaggerated")
```

### Common Workflows

**Pre-experiment: Sample size calculation**

```python
from experiment_utils import PowerSim

# Determine required sample size
power_sim = PowerSim(metric="proportion", variants=1, nsim=1000)
result = power_sim.find_sample_size(
    power=0.80,
    baseline=0.10,
    effect=0.02
)
print(f"Need {result['total_sample_size'].iloc[0]:,.0f} users")
```

**During experiment: Balance check**

```python
from experiment_utils import check_covariate_balance

# Check if randomization worked
balance = check_covariate_balance(
    data=experiment_df,
    treatment_col="treatment",
    covariates=["age", "region", "tenure"]
)
print(f"Balance: {balance['balance_flag'].mean():.1%}")
```

**Post-experiment: Analysis**

```python
from experiment_utils import ExperimentAnalyzer

# Full analysis pipeline
analyzer = ExperimentAnalyzer(
    data=df,
    treatment_col="treatment",
    outcomes=["primary_metric", "secondary_metric"],
    balance_covariates=["age", "region"],
    adjustment="balance",
    bootstrap=True,
)

analyzer.get_effects()
analyzer.adjust_pvalues(method="holm")

# Report
results = analyzer.results
print(results[["outcome", "absolute_effect", "relative_effect", 
               "pvalue_mcp", "stat_significance_mcp"]])
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{experiment_utils_pd,
  title = {Experiment Utils PD: A Python Package for Experiment Analysis},
  author = {Sebastian Daza},
  year = {2026},
  url = {https://github.com/sdaza/experiment-utils-pd}
}
```

