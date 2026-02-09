# %% [markdown]
# # Survival Analysis: Unadjusted vs Regression vs IPW
#
# Demonstrates how confounding biases Cox regression results,
# and compares regression adjustment vs IPW for recovering the true effect.
#
# DGP:
# - Treatment assignment depends on age, income (confounded)
# - Survival depends on treatment + age + income + prior_clicks
# - True treatment effect: HR = exp(-0.3) ≈ 0.74 (treatment reduces hazard)

# %% Imports
import numpy as np
import pandas as pd

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

# %% Generate survival data with confounding

np.random.seed(42)
n = 5000

data = pd.DataFrame(
    {
        "experiment_id": 1,
        "user_id": np.arange(n),
        "age": np.random.normal(35, 10, n),
        "income": np.random.lognormal(10, 0.5, n),
        "prior_clicks": np.random.poisson(2, n),
    }
)

# confounded treatment: older and higher-income users more likely to be treated
logit_ps = -0.5 + 0.6 * (data["age"] - 35) / 10 + 0.3 * np.log(data["income"]) / 2
data["treatment"] = np.random.binomial(1, 1 / (1 + np.exp(-logit_ps)), n)

print(f"Treatment rate: {data['treatment'].mean():.2%}")
print(f"Mean age (treated): {data.loc[data['treatment'] == 1, 'age'].mean():.1f}")
print(f"Mean age (control): {data.loc[data['treatment'] == 0, 'age'].mean():.1f}")

# %% Simulate time-to-churn
# TRUE causal effect: treatment REDUCES hazard (protective)
# log(HR) = -0.3, so HR = exp(-0.3) ≈ 0.74

baseline_hazard = 0.01
true_log_hr = -0.3  # treatment is protective

hazard = baseline_hazard * np.exp(
    true_log_hr * data["treatment"]
    + 0.15 * (data["age"] - 35) / 10
    + 0.00005 * data["income"]
    + 0.05 * data["prior_clicks"]
)

data["time_to_churn"] = np.random.exponential(scale=1 / hazard)

# censoring at 1 year
censoring_time = 365
data["churned"] = (data["time_to_churn"] <= censoring_time).astype(int)
data["time_to_churn"] = np.minimum(data["time_to_churn"], censoring_time)

print(f"\nTrue log(HR): {true_log_hr}")
print(f"True HR: {np.exp(true_log_hr):.3f}")
print(f"Churn rate: {data['churned'].mean():.2%}")

# %% 1. Unadjusted (confounded)
print("\n" + "=" * 70)
print("1. UNADJUSTED (no covariate adjustment)")
print("   Expected: biased estimate due to confounding")
print("=" * 70)

analyzer_unadj = ExperimentAnalyzer(
    data=data,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    outcome_models="cox",
    bootstrap=False,
)
analyzer_unadj.get_effects()

r = analyzer_unadj.results
print(f"\n  Estimated log(HR): {r['absolute_effect'].iloc[0]:.4f}  (true: {true_log_hr})")
print(f"  Estimated HR:      {np.exp(r['absolute_effect'].iloc[0]):.4f}  (true: {np.exp(true_log_hr):.4f})")
print(f"  p-value:           {r['pvalue'].iloc[0]:.4f}")
print(f"  Bias:              {r['absolute_effect'].iloc[0] - true_log_hr:+.4f}")

# %% 2. Regression adjustment only
print("\n" + "=" * 70)
print("2. REGRESSION ADJUSTMENT (covariates in Cox model)")
print("   Expected: less biased, covariates absorb confounding")
print("=" * 70)

analyzer_reg = ExperimentAnalyzer(
    data=data,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    regression_covariates=["age", "income", "prior_clicks"],
    outcome_models="cox",
    bootstrap=False,
)
analyzer_reg.get_effects()

r = analyzer_reg.results
print(f"\n  Estimated log(HR): {r['absolute_effect'].iloc[0]:.4f}  (true: {true_log_hr})")
print(f"  Estimated HR:      {np.exp(r['absolute_effect'].iloc[0]):.4f}  (true: {np.exp(true_log_hr):.4f})")
print(f"  p-value:           {r['pvalue'].iloc[0]:.4f}")
print(f"  Bias:              {r['absolute_effect'].iloc[0] - true_log_hr:+.4f}")
print(f"  Balance:           {r['balance'].iloc[0]:.0%}")

# %% 3. IPW only (ps-logistic)
print("\n" + "=" * 70)
print("3. IPW ONLY (propensity score weighting, ps-logistic)")
print("   Expected: unbiased if PS model is correct")
print("=" * 70)

analyzer_ipw = ExperimentAnalyzer(
    data=data,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    adjustment="balance",
    balance_method="ps-logistic",
    target_effect="ATE",
    outcome_models="cox",
    bootstrap=False,
)
analyzer_ipw.get_effects()

r = analyzer_ipw.results
print(f"\n  Estimated log(HR): {r['absolute_effect'].iloc[0]:.4f}  (true: {true_log_hr})")
print(f"  Estimated HR:      {np.exp(r['absolute_effect'].iloc[0]):.4f}  (true: {np.exp(true_log_hr):.4f})")
print(f"  p-value:           {r['pvalue'].iloc[0]:.4f}")
print(f"  Bias:              {r['absolute_effect'].iloc[0] - true_log_hr:+.4f}")
print(f"  Adjusted balance:  {r['balance'].iloc[0]:.0%}")

# %% 4. IPW with entropy balancing
print("\n" + "=" * 70)
print("4. IPW ONLY (entropy balancing)")
print("   Expected: often better balance than logistic PS")
print("=" * 70)

analyzer_entropy = ExperimentAnalyzer(
    data=data,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    adjustment="balance",
    balance_method="entropy",
    target_effect="ATE",
    outcome_models="cox",
    bootstrap=False,
)
analyzer_entropy.get_effects()

r = analyzer_entropy.results
print(f"\n  Estimated log(HR): {r['absolute_effect'].iloc[0]:.4f}  (true: {true_log_hr})")
print(f"  Estimated HR:      {np.exp(r['absolute_effect'].iloc[0]):.4f}  (true: {np.exp(true_log_hr):.4f})")
print(f"  p-value:           {r['pvalue'].iloc[0]:.4f}")
print(f"  Bias:              {r['absolute_effect'].iloc[0] - true_log_hr:+.4f}")
print(f"  Adjusted balance:  {r['balance'].iloc[0]:.0%}")

# %% 5. IPW + Regression (both)
print("\n" + "=" * 70)
print("5. IPW + REGRESSION (ps-logistic + covariates in Cox model)")
print("   Expected: combines both adjustments")
print("=" * 70)

analyzer_both = ExperimentAnalyzer(
    data=data,
    outcomes=[("time_to_churn", "churned")],
    treatment_col="treatment",
    experiment_identifier=["experiment_id"],
    covariates=["age", "income", "prior_clicks"],
    regression_covariates=["age", "income", "prior_clicks"],
    adjustment="balance",
    balance_method="ps-logistic",
    target_effect="ATE",
    outcome_models="cox",
    bootstrap=False,
)
analyzer_both.get_effects()

r = analyzer_both.results
print(f"\n  Estimated log(HR): {r['absolute_effect'].iloc[0]:.4f}  (true: {true_log_hr})")
print(f"  Estimated HR:      {np.exp(r['absolute_effect'].iloc[0]):.4f}  (true: {np.exp(true_log_hr):.4f})")
print(f"  p-value:           {r['pvalue'].iloc[0]:.4f}")
print(f"  Bias:              {r['absolute_effect'].iloc[0] - true_log_hr:+.4f}")
print(f"  Adjusted balance:  {r['balance'].iloc[0]:.0%}")

# %% Summary table
print("\n" + "=" * 70)
print("SUMMARY")
print(f"True log(HR) = {true_log_hr}, True HR = {np.exp(true_log_hr):.3f}")
print("=" * 70)

summary = []
for name, analyzer in [
    ("Unadjusted", analyzer_unadj),
    ("Regression", analyzer_reg),
    ("IPW (logistic)", analyzer_ipw),
    ("IPW (entropy)", analyzer_entropy),
    ("IPW + Regression", analyzer_both),
]:
    r = analyzer.results
    est = r["absolute_effect"].iloc[0]
    summary.append(
        {
            "Method": name,
            "log(HR)": round(est, 4),
            "HR": round(np.exp(est), 4),
            "Bias": round(est - true_log_hr, 4),
            "p-value": round(r["pvalue"].iloc[0], 4),
        }
    )

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# %%
