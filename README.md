![ci](https://github.com/sdaza/experiment-utils/actions/workflows/ci.yaml/badge.svg)


# Experiment utils

Generic functions for PySpark experiment analysis and design 

- [Experiment utils](#experiment-utils)
- [Installation](#installation)
- [How to use it](#how-to-use-it)
  - [Experiment Analyzer](#experiment-analyzer)
  - [Power Analysis](#power-analysis)

# Installation

```
pip install git+https://github.com/sdaza/experiment-utils.git
```

# How to use it

## Experiment Analyzer

The DataFrame `df` is a PySpark DataFrame. If it's a Pandas DataFrame, it will transform automatically.

```python
analyzer = ExperimentAnalyzer(
    df,
    treatment_col="treatment",
    outcomes=['registrations', 'visits'],
    covariates=covariates,
    experiment_identifier=["campaign_key"],
    adjustment=None)

analyzer.get_effects()
analyzer.results
```

## Power Analysis


```python
from experiment_utils import PowerSim
p = PowerSim(metric='proportion', relative_effect=False,
	variants=1, nsim=1000, alpha=0.05, alternative='two-tailed')

p.get_power(baseline=[0.33], effect=[0.03], sample_size=[3000])
```

