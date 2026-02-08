# %% Imports
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# %% first create synthetic data fro survival analysis
np.random.seed(42)
n = 3000

# %% simulate churn, and time to churn
# Create baseline covariates first
data_survival = pd.DataFrame(
    {
        "experiment_id": 1,
        "user_id": np.arange(n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.lognormal(10, 0.5, n),
        "prior_clicks": np.random.poisson(2, n),
    }
)

# Simulate confounded treatment assignment (depends on age and income)
# This creates confounding: treatment is associated with age/income, which affect survival
data_survival["treatment"] = np.random.binomial(
    1, 1 / (1 + np.exp(-0.8 * (data_survival["age"] - 35) / 10 + 0.0001 * data_survival["income"])), n
)

# Simulate time to churn using an exponential distribution
# TRUE treatment effect is ZERO - treatment does NOT causally affect survival
# However, age, income, and prior_clicks DO affect survival (creating confounding)
baseline_hazard = 0.01
treatment_effect = 0.0  # TRUE causal effect is zero
data_survival["time_to_churn"] = np.random.exponential(
    scale=1
    / (
        baseline_hazard
        * np.exp(
            -treatment_effect * data_survival["treatment"]
            + 0.15 * (data_survival["age"] - 35) / 10
            + 0.0001 * data_survival["income"]
            + 0.05 * data_survival["prior_clicks"]
        )
    )
)

# Simulate censoring (e.g., users who haven't churned by the end of the study)
censoring_time = 365  # 1 year
data_survival["churned"] = (data_survival["time_to_churn"] <= censoring_time).astype(int)
data_survival["time_to_churn"] = np.minimum(data_survival["time_to_churn"], censoring_time)

# %% analyze treatment effect WITHOUT adjustment for confounding
# This should show a SPURIOUS effect due to confounding (treatment associated with age/income)
analyzer_survival = ExperimentAnalyzer(
    data=data_survival,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    # regression_covariates=["age", "income", "prior_clicks"],
    outcome_models="cox",  # Use Cox proportional hazards model
    store_fitted_models=True,  # Store models for inspection
    bootstrap=False,
    bootstrap_iterations=1000,
    # pvalue_adjustment='sidak',  # Benjamini-Hochberg for multiple comparisons
)

# %%
analyzer_survival.get_effects()
# %%
print(analyzer_survival.results)
# %%
model = analyzer_survival.get_fitted_models(experiment=(1,), comparison=(1, 0), outcome="time_to_churn")

# %%
analyzer_survival.get_fitted_models()

# %% analyze treatment effect WITH IPW adjustment for confounding
# This should recover the TRUE zero effect by reweighting to balance covariates
analyzer_survival_ipw = ExperimentAnalyzer(
    data=data_survival,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    adjustment="balance",
    balance_method="entropy",
    outcome_models="cox",
    store_fitted_models=True,
    bootstrap=True,
)

# %%
analyzer_survival_ipw.get_effects()

# %%
print(analyzer_survival_ipw.adjusted_balance)

# %%
print(analyzer_survival_ipw.results)
# %%
analyzer_survival_ipw.calculate_retrodesign()
# %%
