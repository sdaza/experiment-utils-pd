# %% [markdown]
# # IPW vs Regression Adjustment: When Do They Diverge?
#
# Simulation study comparing 4 estimators under varying conditions:
# - **Unadjusted**: no covariates, no weighting
# - **Regression**: covariates in outcome model
# - **IPW**: propensity score weighting (no covariates in outcome model)
# - **Doubly robust**: IPW + regression covariates
#
# Factors varied:
# 1. **Selection into treatment**: weak (near-random) vs strong (confounded)
# 2. **Covariate prognostic strength**: weak vs strong effect of X on Y
# 3. **Treatment effect heterogeneity**: none vs present

# %% Imports
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiment_utils.experiment_analyzer import ExperimentAnalyzer

warnings.filterwarnings("ignore")


# %% DGP
def generate_data(
    n: int = 3000,
    selection_strength: float = 0.0,
    prognostic_strength: float = 0.0,
    heterogeneity: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """
    Generate binary outcome data with logistic DGP.

    Parameters
    ----------
    n : int
        Sample size
    selection_strength : float
        How much X affects treatment assignment.
        0.0 = randomized, 1.0 = strong confounding.
    prognostic_strength : float
        How much X affects outcome.
        0.0 = weak, 1.0 = strong.
    heterogeneity : bool
        Whether treatment effect varies with X.
    seed : int
        Random seed.

    Returns
    -------
    tuple[pd.DataFrame, float]
        (data, true_ate) where true_ate is the population ATE on probability scale.
    """
    rng = np.random.default_rng(seed)

    # covariates
    x1 = rng.normal(0, 1, n)  # continuous
    x2 = rng.binomial(1, 0.4, n).astype(float)  # binary

    # treatment assignment with selection
    # selection_strength controls confounding: 0 = random, 1 = strong
    logit_ps = -0.1 + selection_strength * (0.8 * x1 + 0.6 * x2)
    ps = 1 / (1 + np.exp(-logit_ps))
    treatment = rng.binomial(1, ps).astype(float)

    # outcome model
    # prognostic_strength controls how much X predicts Y
    beta_x1 = 0.1 + prognostic_strength * 1.5  # range: 0.1 (weak) to 1.6 (strong)
    beta_x2 = 0.05 + prognostic_strength * 0.8  # range: 0.05 (weak) to 0.85 (strong)

    # treatment effect (on logit scale)
    beta_treat = 0.5
    if heterogeneity:
        # treatment effect depends on x1: larger effect for high x1
        treat_effect = beta_treat + 0.5 * x1
    else:
        treat_effect = beta_treat

    logit_y = -1.5 + treat_effect * treatment + beta_x1 * x1 + beta_x2 * x2
    prob_y = 1 / (1 + np.exp(-logit_y))
    outcome = rng.binomial(1, prob_y)

    data = pd.DataFrame(
        {
            "experiment_id": 1,
            "treatment": treatment,
            "x1": x1,
            "x2": x2,
            "outcome": outcome,
        }
    )

    # compute true ATE on probability scale (large sample approximation)
    rng_ate = np.random.default_rng(0)
    n_pop = 200_000
    x1_pop = rng_ate.normal(0, 1, n_pop)
    x2_pop = rng_ate.binomial(1, 0.4, n_pop).astype(float)

    if heterogeneity:
        te_pop = beta_treat + 0.5 * x1_pop
    else:
        te_pop = beta_treat

    logit_y1 = -1.5 + te_pop + beta_x1 * x1_pop + beta_x2 * x2_pop
    logit_y0 = -1.5 + beta_x1 * x1_pop + beta_x2 * x2_pop
    true_ate = float(np.mean(1 / (1 + np.exp(-logit_y1)) - 1 / (1 + np.exp(-logit_y0))))

    return data, true_ate


# %% Estimation
def run_estimators(data: pd.DataFrame) -> dict[str, float]:
    """
    Run 4 estimators on the data and return absolute_effect for each.

    Returns dict with keys: unadjusted, regression, ipw, doubly_robust
    """
    results = {}
    covs = ["x1", "x2"]

    # 1. Unadjusted
    try:
        a = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            outcome_models={"outcome": "logistic"},
            bootstrap=False,
        )
        a.get_effects()
        results["unadjusted"] = a.results["absolute_effect"].iloc[0]
    except Exception:
        results["unadjusted"] = np.nan

    # 2. Regression only
    try:
        a = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            covariates=covs,
            regression_covariates=covs,
            outcome_models={"outcome": "logistic"},
            bootstrap=False,
        )
        a.get_effects()
        results["regression"] = a.results["absolute_effect"].iloc[0]
    except Exception:
        results["regression"] = np.nan

    # 3. IPW only
    try:
        a = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            covariates=covs,
            adjustment="balance",
            target_effect="ATE",
            outcome_models={"outcome": "logistic"},
            bootstrap=False,
        )
        a.get_effects()
        results["ipw"] = a.results["absolute_effect"].iloc[0]
    except Exception:
        results["ipw"] = np.nan

    # 4. Doubly robust (IPW + regression)
    try:
        a = ExperimentAnalyzer(
            data=data,
            outcomes=["outcome"],
            treatment_col="treatment",
            experiment_identifier=["experiment_id"],
            covariates=covs,
            regression_covariates=covs,
            adjustment="balance",
            target_effect="ATE",
            outcome_models={"outcome": "logistic"},
            bootstrap=False,
        )
        a.get_effects()
        results["doubly_robust"] = a.results["absolute_effect"].iloc[0]
    except Exception:
        results["doubly_robust"] = np.nan

    return results


# %% Single simulation rep
def single_rep(
    rep: int,
    n: int,
    selection_strength: float,
    prognostic_strength: float,
    heterogeneity: bool,
) -> dict:
    """Run one Monte Carlo rep."""
    data, true_ate = generate_data(
        n=n,
        selection_strength=selection_strength,
        prognostic_strength=prognostic_strength,
        heterogeneity=heterogeneity,
        seed=rep,
    )
    estimates = run_estimators(data)
    estimates["true_ate"] = true_ate
    estimates["rep"] = rep
    return estimates


# %% Scenario definitions
SCENARIOS = {
    # (selection_strength, prognostic_strength, heterogeneity)
    "Random + Weak X + No HTE": (0.0, 0.0, False),
    "Random + Strong X + No HTE": (0.0, 1.0, False),
    "Random + Strong X + HTE": (0.0, 1.0, True),
    "Selection + Weak X + No HTE": (1.0, 0.0, False),
    "Selection + Strong X + No HTE": (1.0, 1.0, False),
    "Selection + Strong X + HTE": (1.0, 1.0, True),
}

# %% Run simulation
N_REPS = 500
N_OBS = 3000


def run_scenario(name: str, params: tuple) -> pd.DataFrame:
    """Run all reps for one scenario."""
    selection_strength, prognostic_strength, heterogeneity = params

    reps = Parallel(n_jobs=-1, verbose=0)(
        delayed(single_rep)(
            rep=i,
            n=N_OBS,
            selection_strength=selection_strength,
            prognostic_strength=prognostic_strength,
            heterogeneity=heterogeneity,
        )
        for i in range(N_REPS)
    )

    df = pd.DataFrame(reps)
    df["scenario"] = name
    return df


print(f"Running {len(SCENARIOS)} scenarios x {N_REPS} reps each...")
all_results = []
for name, params in SCENARIOS.items():
    print(f"  {name}...")
    df = run_scenario(name, params)
    all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)
print("Done!")


# %% Compute metrics
def compute_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute bias, RMSE for each estimator x scenario."""
    estimators = ["unadjusted", "regression", "ipw", "doubly_robust"]
    rows = []

    for scenario in results_df["scenario"].unique():
        sdf = results_df[results_df["scenario"] == scenario]
        true_ate = sdf["true_ate"].iloc[0]

        for est in estimators:
            vals = sdf[est].dropna()
            bias = vals.mean() - true_ate
            rmse = np.sqrt(((vals - true_ate) ** 2).mean())
            rows.append(
                {
                    "scenario": scenario,
                    "estimator": est,
                    "true_ate": true_ate,
                    "mean_estimate": vals.mean(),
                    "bias": bias,
                    "abs_bias": abs(bias),
                    "rmse": rmse,
                    "std": vals.std(),
                    "n_valid": len(vals),
                }
            )

    return pd.DataFrame(rows)


metrics = compute_metrics(results_df)

# %% Display results
print("\n" + "=" * 90)
print("RESULTS: Bias and RMSE by Scenario and Estimator")
print("=" * 90)

for scenario in SCENARIOS:
    print(f"\n--- {scenario} ---")
    sdf = metrics[metrics["scenario"] == scenario].copy()
    true_ate = sdf["true_ate"].iloc[0]
    print(f"True ATE: {true_ate:.4f}")
    print(
        sdf[["estimator", "mean_estimate", "bias", "rmse", "std"]].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"
        )
    )

# %% Summary: when do IPW and regression diverge?
print("\n" + "=" * 90)
print("SUMMARY: |Bias(regression) - Bias(IPW)| by scenario")
print("=" * 90)

summary_rows = []
for scenario in SCENARIOS:
    sdf = metrics[metrics["scenario"] == scenario]
    reg_bias = sdf[sdf["estimator"] == "regression"]["bias"].iloc[0]
    ipw_bias = sdf[sdf["estimator"] == "ipw"]["bias"].iloc[0]
    dr_bias = sdf[sdf["estimator"] == "doubly_robust"]["bias"].iloc[0]
    unadj_bias = sdf[sdf["estimator"] == "unadjusted"]["bias"].iloc[0]
    summary_rows.append(
        {
            "scenario": scenario,
            "unadjusted_bias": unadj_bias,
            "regression_bias": reg_bias,
            "ipw_bias": ipw_bias,
            "dr_bias": dr_bias,
            "reg_vs_ipw_gap": abs(reg_bias - ipw_bias),
        }
    )

summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

# %%
