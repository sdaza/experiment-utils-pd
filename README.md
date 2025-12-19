[![ci](https://github.com/sdaza/experiment-utils-pd/actions/workflows/ci.yaml/badge.svg)](https://github.com/sdaza/experiment-utils-pd/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/experiment-utils-pd.svg)](https://pypi.org/project/experiment-utils-pd/)


# Experiment utils

Generic functions for experiment analysis and design:

- [Experiment utils](#experiment-utils)
- [Installation](#installation)
  - [PyPI](#pypi)
  - [From GitHub](#from-github)
- [How to use it](#how-to-use-it)
  - [Experiment Analyzer](#experiment-analyzer)
    - [Retrieve IPW Weights](#retrieve-ipw-weights)
    - [Non-inferiority test](#non-inferiority-test)
    - [Multiple comparison adjustment](#multiple-comparison-adjustment)
  - [Power Analysis](#power-analysis)
  - [Utilities](#utilities)
    - [Balanced Random Assignment](#balanced-random-assignment)

# Installation

## PyPI

```
pip install experiment-utils-pd
```


## From GitHub

```
pip install git+https://github.com/sdaza/experiment-utils-pd.git
```

# How to use it

## Experiment Analyzer

Suppose you have a DataFrame `df` with columns for experiment group, treatment assignment, outcomes, and covariates.

```python
import pandas as pd
from experiment_utils import ExperimentAnalyzer

# Example data
df = pd.DataFrame({
    "experiment_id": [1, 1, 1, 2, 2, 2],
    "user_id": [101, 102, 103, 201, 202, 203],
    "treatment": [0, 1, 0, 1, 0, 1],
    "age": [25, 34, 29, 40, 22, 31],
    "gender": [1, 0, 1, 0, 1, 0],
    "outcome1": [0, 1, 0, 1, 0, 1],
    "outcome2": [5.2, 6.1, 5.8, 7.0, 5.5, 6.8],
})

covariates = ["age", "gender"]

# Initialize analyzer with balance adjustment
analyzer = ExperimentAnalyzer(
    df,
    treatment_col="treatment",
    outcomes=["outcome1", "outcome2"],
    covariates=covariates,
    experiment_identifier=["experiment_id"],
    unit_identifier=["user_id"], # Optional: To retrieve balance weights
    adjustment="balance",  # Options: 'balance', 'IV', or None
    balance_method="ps-logistic",  # Options: 'ps-logistic', 'ps-xgboost', 'entropy'
    target_effect="ATE",  # Options: 'ATT', 'ATE', 'ATC'
    bootstrap=True,  # Enable bootstrap inference (default: False)
    bootstrap_iterations=1000,  # Number of bootstrap iterations (default: 1000)
    bootstrap_seed=42  # Seed for reproducibility (optional)
)

# Estimate effects
analyzer.get_effects()
print(analyzer.results)
```

**Parameters:**
- `adjustment`: Choose 'balance' for covariate balancing (using balance_method), 'IV' for instrumental variable adjustment, or None for unadjusted analysis.
- `balance_method`: Selects the method for balancing: 'ps-logistic' (logistic regression), 'ps-xgboost' (XGBoost), or 'entropy' (entropy balancing).
- `target_effect`: Specifies the estimand: 'ATT', 'ATE', or 'ATC'.
- `bootstrap`: Enable bootstrap inference for p-values and confidence intervals (default: False for asymptotic inference).
- `bootstrap_iterations`: Number of bootstrap resampling iterations (default: 1000).
- `bootstrap_ci_method`: Method for computing bootstrap CIs: 'percentile' or 'basic' (default: 'percentile').
- `bootstrap_stratify`: Whether to use stratified resampling by treatment group (default: True).
- `bootstrap_seed`: Random seed for reproducible bootstrap results (optional).


### Retrieve IPW Weights

To inspect the weights and selected sample after balancing:
```python
# Get the DataFrame with weights and experiment identifiers
weights_df = analyzer.weights
print(weights_df.head())
```

### Non-inferiority test

Test for non-inferiority after estimating effects:
```python
# Test non-inferiority with a 10% margin
analyzer.test_non_inferiority(relative_margin=0.10)
print(analyzer.results[["outcome1", "non_inferiority_margin", "ci_lower_bound", "is_non_inferior"]])
```

### Multiple comparison adjustment

Adjust p-values for multiple outcomes per experiment:
```python
# Bonferroni adjustment
analyzer.adjust_pvalues(method="bonferroni")
print(analyzer.results[["outcome1", "pvalue", "pvalue_adj", "stat_significance_adj"]])

# Or use FDR (Benjamini-Hochberg)
analyzer.adjust_pvalues(method="fdr_bh")
print(analyzer.results[["outcome1", "pvalue", "pvalue_adj", "stat_significance_adj"]])
```

## Power Analysis


```python
from experiment_utils import PowerSim
p = PowerSim(metric='proportion', relative_effect=False,
  variants=1, nsim=1000, alpha=0.05, alternative='two-tailed')

p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
```

## Utilities

### Balanced Random Assignment

You can use the `balanced_random_assignment` utility to assign units to experimental groups with forced balance. Optionally stratify by covariates to ensure balance within strata.

```python
from experiment_utils.utils import balanced_random_assignment
import pandas as pd

# Example DataFrame
users = pd.DataFrame({
    "user_id": range(100),
    "age_group": ["young", "old"] * 50,
    "gender": ["M", "F"] * 50
})

# Binary assignment (test/control, 50/50) without stratification
users["assignment"] = balanced_random_assignment(users, allocation_ratio=0.5)
print(users)

# Binary assignment with stratification by age_group and gender
users["assignment_stratified"] = balanced_random_assignment(
    users, 
    allocation_ratio=0.5, 
    balance_covariates=["age_group", "gender"]
)
print(users)

# Multiple variants with equal allocation
users["assignment_multi"] = balanced_random_assignment(
    users, 
    variants=["control", "A", "B"]
)
print(users)

# Multiple variants with custom allocation and stratification
users["assignment_custom"] = balanced_random_assignment(
    users,
    variants=["control", "A", "B"],
    allocation_ratio={"control": 0.5, "A": 0.3, "B": 0.2},
    balance_covariates=["age_group"]
)
print(users)
```

