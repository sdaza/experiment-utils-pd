# %% [markdown]
# # GLM outcome models (logistic, NB)
#
# Binary and overdispersed count outcomes with `ExperimentAnalyzer`
# (`outcome_models`). Compares OLS vs GLM, with and without balance adjustment.
#
# Run:
#
#     uv run python examples/glm_examples.py

# %%
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

np.random.seed(42)
n = 3000

# %% [markdown]
# ## Binary outcome — logistic vs OLS

# %%
data_binary = pd.DataFrame(
    {
        "experiment_id": 1,
        "user_id": np.arange(n),
        "treatment": np.random.binomial(1, 0.5, n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.lognormal(10, 0.5, n),
        "prior_clicks": np.random.poisson(2, n),
    }
)

prob = 1 / (
    1
    + np.exp(
        -(
            -6.0
            + 0.6 * data_binary["treatment"]
            + 0.02 * data_binary["age"]
            + 0.00001 * data_binary["income"]
            + 0.1 * data_binary["prior_clicks"]
        )
    )
)
data_binary["clicked"] = np.random.binomial(1, prob)

analyzer_binary = ExperimentAnalyzer(
    data=data_binary,
    outcomes=["clicked"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    regression_covariates=["age", "income", "prior_clicks"],
    outcome_models={"clicked": ["ols", "logistic"]},
    bootstrap=False,
    bootstrap_iterations=2000,
)

analyzer_binary.get_effects()
print(analyzer_binary.results.iloc[:, :23])

# %% [markdown]
# ## Same binary outcome with balance adjustment (ATT)

# %%
analyzer_binary = ExperimentAnalyzer(
    data=data_binary,
    outcomes=["clicked"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    adjustment="balance",
    target_effect="ATT",
    outcome_models={"clicked": ["ols", "logistic"]},
    bootstrap=False,
    bootstrap_iterations=2000,
)

analyzer_binary.get_effects()
print(analyzer_binary.results.iloc[:, :23])
analyzer_binary.calculate_retrodesign(true_effect=0.005)

# %% [markdown]
# ## Overdispersed counts — Poisson vs negative binomial

# %%
data_count = pd.DataFrame(
    {
        "experiment_id": 1,
        "user_id": np.arange(n),
        "treatment": np.random.binomial(1, 0.5, n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.lognormal(10, 0.5, n),
        "prior_clicks": np.random.poisson(2, n),
    }
)

mu = np.exp(
    -0.9
    + 0.3 * data_count["treatment"]
    + 0.01 * data_count["age"]
    + 0.000005 * data_count["income"]
    + 0.05 * data_count["prior_clicks"]
)
data_count["clicks"] = np.random.negative_binomial(mu, 0.5)

analyzer_count = ExperimentAnalyzer(
    data=data_count,
    outcomes=["clicks"],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    outcome_models={"clicks": ["ols", "poisson", "negative_binomial"]},
    bootstrap=True,
    bootstrap_iterations=2000,
    target_effect="ATE",
)

analyzer_count.get_effects()
print(analyzer_count.results)
print(analyzer_count.calculate_retrodesign(true_effect=0.10))
