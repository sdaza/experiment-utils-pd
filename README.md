
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
  - [Power Analysis](#power-analysis)

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

 `df` is a Pandas DataFrame:


```python
from experiment_utils import ExperimentAnalyzer

# Example with balance adjustment and balance_method
analyzer = ExperimentAnalyzer(
    df,
    treatment_col="treatment",
    outcomes=['registrations', 'visits'],
    covariates=covariates,
    experiment_identifier=["campaign_key"],
    adjustment="balance",  # Options: 'balance', 'IV', or None
    balance_method="ps-logistic",  # Options: 'ps-logistic', 'ps-xgboost', 'entropy'
    target_effect="ATT"  # Options: 'ATT', 'ATE', 'ATC'
)

analyzer.get_effects()
print(analyzer.results)
```

**Parameters:**
- `adjustment`: Choose 'balance' for covariate balancing (using balance_method), 'IV' for instrumental variable adjustment, or None for unadjusted analysis.
- `balance_method`: Selects the method for balancing: 'ps-logistic' (logistic regression), 'ps-xgboost' (XGBoost), or 'entropy' (entropy balancing).
- `target_effect`: Specifies the estimand: 'ATT', 'ATE', or 'ATC'.


## Power Analysis


```python
from experiment_utils import PowerSim
p = PowerSim(metric='proportion', relative_effect=False,
  variants=1, nsim=1000, alpha=0.05, alternative='two-tailed')

p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
```

