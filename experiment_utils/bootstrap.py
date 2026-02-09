"""Bootstrap inference mixin for ExperimentAnalyzer."""

import warnings

import numpy as np
import pandas as pd


class BootstrapMixin:
    """Mixin providing bootstrap resampling, inference, and distribution management."""

    def _stratified_resample(
        self, data: pd.DataFrame, seed: int | None = None, event_col: str | None = None
    ) -> pd.DataFrame:
        """
        Perform stratified resampling with replacement, stratified by treatment group.
        Supports cluster-aware resampling when cluster_col is specified.
        For survival models, can stratify by both treatment and event status.

        Parameters
        ----------
        data : pd.DataFrame
            Data to resample
        seed : int, optional
            Random seed for reproducibility
        event_col : str, optional
            Event column for survival models. When provided, stratifies by
            treatment x event status to ensure sufficient events in each bootstrap sample.

        Returns
        -------
        pd.DataFrame
            Resampled data
        """
        if seed is not None:
            np.random.seed(seed)

        # event-stratified bootstrap for survival models
        if event_col is not None and self._bootstrap_stratify:
            resampled_parts = []
            for treatment_val in [0, 1]:
                for event_val in [0, 1]:
                    stratum = data[(data[self._treatment_col] == treatment_val) & (data[event_col] == event_val)]
                    if len(stratum) > 0:
                        stratum_resample = stratum.sample(n=len(stratum), replace=True)
                        resampled_parts.append(stratum_resample)
            resampled_data = pd.concat(resampled_parts, ignore_index=True)

        # cluster-aware bootstrap
        elif self._cluster_col is not None:
            if self._bootstrap_stratify:
                treated = data[data[self._treatment_col] == 1]
                control = data[data[self._treatment_col] == 0]
                treated_clusters = treated[self._cluster_col].unique()
                control_clusters = control[self._cluster_col].unique()

                # resample clusters with replacement, gathering rows per occurrence
                # to preserve duplicates (isin() would deduplicate)
                resampled_treated_clusters = np.random.choice(
                    treated_clusters, size=len(treated_clusters), replace=True
                )
                resampled_control_clusters = np.random.choice(
                    control_clusters, size=len(control_clusters), replace=True
                )
                treated_parts = [treated[treated[self._cluster_col] == c] for c in resampled_treated_clusters]
                control_parts = [control[control[self._cluster_col] == c] for c in resampled_control_clusters]
                resampled_data = pd.concat(treated_parts + control_parts, ignore_index=True)
            else:
                # simple cluster resampling (no stratification), preserving duplicates
                clusters = data[self._cluster_col].unique()
                resampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
                parts = [data[data[self._cluster_col] == c] for c in resampled_clusters]
                resampled_data = pd.concat(parts, ignore_index=True)

        # standard row-level bootstrap
        elif self._bootstrap_stratify:
            treated = data[data[self._treatment_col] == 1]
            control = data[data[self._treatment_col] == 0]
            treated_resample = treated.sample(n=len(treated), replace=True)
            control_resample = control.sample(n=len(control), replace=True)
            resampled_data = pd.concat([treated_resample, control_resample], ignore_index=True)
        else:
            resampled_data = data.sample(n=len(data), replace=True)

        return resampled_data

    def _bootstrap_single_iteration(
        self,
        data: pd.DataFrame,
        outcome: str,
        adjustment: str | None,
        model_func,
        relevant_covariates: list[str],
        numeric_covariates: list[str],
        binary_covariates: list[str],
        min_binary_count: int,
        model_type: str,
        compute_marginal_effects: str | bool,
        iteration: int,
        event_col_for_resample: str | None = None,
    ) -> tuple[float | None, float | None, str | None]:
        """
        Execute a single bootstrap iteration.

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        outcome : str
            Outcome variable
        adjustment : str | None
            Adjustment method
        model_func : callable
            Model function to use for estimation
        relevant_covariates : list[str]
            Relevant covariates for regression
        numeric_covariates : list[str]
            Numeric covariates
        binary_covariates : list[str]
            Binary covariates
        min_binary_count : int
            Minimum count for binary covariates
        model_type : str
            Model type
        compute_marginal_effects : str | bool
            Whether to compute marginal effects
        iteration : int
            Iteration number for seeding
        event_col_for_resample : str | None
            Event column for survival models

        Returns
        -------
        tuple[float | None, float | None, str | None]
            (absolute_effect, relative_effect, error_info)
        """
        try:
            seed = self._bootstrap_seed + iteration if self._bootstrap_seed is not None else None
            boot_data = self._stratified_resample(data, seed=seed, event_col=event_col_for_resample)

            boot_data = self.impute_missing_values(
                data=boot_data.copy(),
                num_covariates=numeric_covariates,
                bin_covariates=binary_covariates,
            )

            boot_numeric = [c for c in numeric_covariates if boot_data[c].std(ddof=0) != 0]
            boot_binary = [c for c in binary_covariates if boot_data[c].sum() >= min_binary_count]
            boot_binary = [c for c in boot_binary if boot_data[c].std(ddof=0) != 0]
            boot_final_covariates = boot_numeric + boot_binary

            if len(boot_final_covariates) > 0:
                boot_data = self.standardize_covariates(boot_data, boot_final_covariates)

            if adjustment == "balance":
                weight_col = self._target_weights[self._target_effect]
                if len(boot_final_covariates) == 0:
                    # No covariates to balance on: assign uniform weights
                    boot_data[weight_col] = 1.0
                elif self._bootstrap_fixed_weights:
                    if weight_col not in boot_data.columns:
                        self._logger.warning(
                            f"bootstrap_fixed_weights=True but weight column '{weight_col}' not found in data. "
                            f"Falling back to recalculating weights."
                        )
                        get_balance_method = self._estimator.get_balance_method
                        balance_func = get_balance_method(self._balance_method)
                        z_covs = [f"z_{cov}" for cov in boot_final_covariates]
                        boot_data = balance_func(data=boot_data, covariates=z_covs)
                else:
                    get_balance_method = self._estimator.get_balance_method
                    balance_func = get_balance_method(self._balance_method)
                    z_covs = [f"z_{cov}" for cov in boot_final_covariates]
                    boot_data = balance_func(data=boot_data, covariates=z_covs)
            else:
                weight_col = None

            boot_relevant_covariates = set(boot_final_covariates) & set(relevant_covariates)
            estimator_params = {
                "data": boot_data,
                "outcome_variable": outcome,
                "covariates": list(boot_relevant_covariates),
                "cluster_col": self._cluster_col,
                "store_model": False,
                "compute_relative_ci": False,
            }

            if model_type in ["logistic", "poisson", "negative_binomial"]:
                estimator_params["compute_marginal_effects"] = compute_marginal_effects

            if model_type == "cox":
                if outcome in self._outcome_event_cols:
                    estimator_params["event_col"] = self._outcome_event_cols[outcome]
                elif self._event_col:
                    estimator_params["event_col"] = self._event_col
                else:
                    raise ValueError(
                        f"Cox model for outcome '{outcome}' requires event column in bootstrap. "
                        f"Use tuple notation: outcomes=[('{outcome}', 'event_col')] "
                        f"or set event_col parameter."
                    )

            if adjustment == "balance" and weight_col:
                estimator_params["weight_column"] = weight_col

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="cov_type not fully supported with freq_weights")
                output = model_func(**estimator_params)
            return output["absolute_effect"], output["relative_effect"], None

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)[:100]}"
            return None, None, error_info

    def _bootstrap_single_effect(
        self,
        data: pd.DataFrame,
        outcome: str,
        adjustment: str | None,
        model_func,
        relevant_covariates: list[str],
        numeric_covariates: list[str],
        binary_covariates: list[str],
        min_binary_count: int,
        model_type: str = "ols",
        compute_marginal_effects: str | bool = "overall",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap a single effect estimate for one outcome.

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        outcome : str
            Outcome variable
        adjustment : str
            Adjustment method
        model_func : callable
            Model function to use for estimation
        relevant_covariates : list[str]
            Relevant covariates for regression
        numeric_covariates : list[str]
            Numeric covariates
        binary_covariates : list[str]
            Binary covariates
        min_binary_count : int
            Minimum count for binary covariates

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (Bootstrapped absolute effects, Bootstrapped relative effects)
        """
        event_col_for_resample = None
        if model_type == "cox":
            if outcome in self._outcome_event_cols:
                event_col_for_resample = self._outcome_event_cols[outcome]
            elif self._event_col:
                event_col_for_resample = self._event_col

        use_parallel = self._bootstrap_iterations >= 100

        if use_parallel:
            from joblib import Parallel, delayed

            self._logger.info(
                f"Running bootstrap for outcome '{outcome}' model '{model_type}' with "
                f"{self._bootstrap_iterations} iterations in parallel..."
            )

            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(self._bootstrap_single_iteration)(
                    data=data,
                    outcome=outcome,
                    adjustment=adjustment,
                    model_func=model_func,
                    relevant_covariates=relevant_covariates,
                    numeric_covariates=numeric_covariates,
                    binary_covariates=binary_covariates,
                    min_binary_count=min_binary_count,
                    model_type=model_type,
                    compute_marginal_effects=compute_marginal_effects,
                    iteration=i,
                    event_col_for_resample=event_col_for_resample,
                )
                for i in range(self._bootstrap_iterations)
            )

            bootstrap_abs_effects = []
            bootstrap_rel_effects = []
            error_counts = {}

            for _i, (abs_eff, rel_eff, error_info) in enumerate(results):
                if abs_eff is not None:
                    bootstrap_abs_effects.append(abs_eff)
                    bootstrap_rel_effects.append(rel_eff)
                else:
                    if error_info:
                        error_type = error_info.split(":")[0] if ":" in error_info else "UnknownError"
                        error_message = error_info.split(":", 1)[1].strip() if ":" in error_info else error_info
                    else:
                        error_type = "UnknownError"
                        error_message = "Error occurred in parallel execution"
                    if error_type not in error_counts:
                        error_counts[error_type] = {"count": 0, "first_message": error_message}
                    error_counts[error_type]["count"] += 1
        else:
            self._logger.info(
                f"Running bootstrap for outcome '{outcome}' model '{model_type}' with "
                f"{self._bootstrap_iterations} iterations..."
            )

            bootstrap_abs_effects = []
            bootstrap_rel_effects = []
            error_counts = {}

            for i in range(self._bootstrap_iterations):
                abs_eff, rel_eff, error_info = self._bootstrap_single_iteration(
                    data=data,
                    outcome=outcome,
                    adjustment=adjustment,
                    model_func=model_func,
                    relevant_covariates=relevant_covariates,
                    numeric_covariates=numeric_covariates,
                    binary_covariates=binary_covariates,
                    min_binary_count=min_binary_count,
                    model_type=model_type,
                    compute_marginal_effects=compute_marginal_effects,
                    iteration=i,
                    event_col_for_resample=event_col_for_resample,
                )

                if abs_eff is not None:
                    bootstrap_abs_effects.append(abs_eff)
                    bootstrap_rel_effects.append(rel_eff)
                else:
                    if error_info:
                        error_type = error_info.split(":")[0] if ":" in error_info else "UnknownError"
                        error_message = error_info.split(":", 1)[1].strip() if ":" in error_info else error_info
                    else:
                        error_type = "UnknownError"
                        error_message = "Error occurred during execution"
                    if error_type not in error_counts:
                        error_counts[error_type] = {"count": 0, "first_message": error_message}
                    error_counts[error_type]["count"] += 1
                    if error_counts[error_type]["count"] <= 2:
                        self._logger.debug(f"Bootstrap iteration {i} failed: {error_info}")

        if len(bootstrap_abs_effects) < self._bootstrap_iterations * 0.5:
            success_rate = len(bootstrap_abs_effects) / self._bootstrap_iterations * 100
            error_summary = "; ".join([f"{k} ({v['count']}x): {v['first_message']}" for k, v in error_counts.items()])
            base_message = (
                f"More than 50% of bootstrap iterations failed for outcome {outcome} "
                f"(success rate: {success_rate:.1f}%). Error summary: {error_summary}. "
                f"Results may be unreliable."
            )

            if model_type == "cox":
                event_col = (
                    self._outcome_event_cols.get(outcome) if outcome in self._outcome_event_cols else self._event_col
                )
                if event_col and event_col in data.columns:
                    event_rate = data[event_col].mean()
                    n_total_events = data[event_col].sum()
                    recommendations = [
                        f"Event rate: {event_rate:.1%} ({int(n_total_events)} events)",
                        "Recommendations for Cox models:",
                        "  1. Use skip_bootstrap_for_survival=True (Cox already provides robust SEs)",
                        "  2. Reduce number of covariates (fewer predictors = more stable fits)",
                        "  3. Use bootstrap_fixed_weights=True if using IPW",
                    ]
                    if event_rate < 0.3:
                        recommendations.append("  4. Consider collecting more data or longer follow-up")
                    self._logger.warning(f"{base_message} {' '.join(recommendations)}")
                else:
                    self._logger.warning(
                        f"{base_message} Common causes for Cox models: insufficient events in bootstrap samples, "
                        f"convergence issues. Consider skip_bootstrap_for_survival=True."
                    )
            else:
                self._logger.warning(
                    f"{base_message} Common causes: convergence issues, perfect separation in covariates. "
                    f"Consider reducing covariates or increasing sample size."
                )

        return np.array(bootstrap_abs_effects), np.array(bootstrap_rel_effects)

    def _calculate_bootstrap_inference(
        self,
        bootstrap_abs_effects: np.ndarray,
        bootstrap_rel_effects: np.ndarray,
        observed_abs_effect: float,
        observed_rel_effect: float,
    ) -> dict[str, float]:
        """
        Calculate confidence intervals and p-value from bootstrap distribution.

        Parameters
        ----------
        bootstrap_abs_effects : np.ndarray
            Array of bootstrap absolute effect estimates
        bootstrap_rel_effects : np.ndarray
            Array of bootstrap relative effect estimates
        observed_abs_effect : float
            Observed absolute effect estimate from original data
        observed_rel_effect : float
            Observed relative effect estimate from original data

        Returns
        -------
        dict
            Dictionary with standard error, CI, and p-value
        """
        if len(bootstrap_abs_effects) == 0:
            return {
                "standard_error": np.nan,
                "pvalue": np.nan,
                "abs_effect_lower": np.nan,
                "abs_effect_upper": np.nan,
                "rel_effect_lower": np.nan,
                "rel_effect_upper": np.nan,
            }

        valid_abs_idx = ~np.isnan(bootstrap_abs_effects)
        valid_rel_idx = ~np.isnan(bootstrap_rel_effects)
        bootstrap_abs_effects_clean = bootstrap_abs_effects[valid_abs_idx]
        bootstrap_rel_effects_clean = bootstrap_rel_effects[valid_rel_idx]

        if len(bootstrap_abs_effects_clean) == 0:
            return {
                "standard_error": np.nan,
                "pvalue": np.nan,
                "abs_effect_lower": np.nan,
                "abs_effect_upper": np.nan,
                "rel_effect_lower": np.nan,
                "rel_effect_upper": np.nan,
            }

        bootstrap_se = np.std(bootstrap_abs_effects_clean, ddof=1)

        alpha = self._alpha
        if self._bootstrap_ci_method == "percentile":
            abs_ci_lower = np.percentile(bootstrap_abs_effects_clean, alpha / 2 * 100)
            abs_ci_upper = np.percentile(bootstrap_abs_effects_clean, (1 - alpha / 2) * 100)
            if len(bootstrap_rel_effects_clean) > 0:
                rel_ci_lower = np.percentile(bootstrap_rel_effects_clean, alpha / 2 * 100)
                rel_ci_upper = np.percentile(bootstrap_rel_effects_clean, (1 - alpha / 2) * 100)
            else:
                rel_ci_lower = np.nan
                rel_ci_upper = np.nan
        elif self._bootstrap_ci_method == "basic":
            abs_ci_lower = 2 * observed_abs_effect - np.percentile(bootstrap_abs_effects_clean, (1 - alpha / 2) * 100)
            abs_ci_upper = 2 * observed_abs_effect - np.percentile(bootstrap_abs_effects_clean, alpha / 2 * 100)
            if len(bootstrap_rel_effects_clean) > 0:
                rel_ci_lower = 2 * observed_rel_effect - np.percentile(
                    bootstrap_rel_effects_clean, (1 - alpha / 2) * 100
                )
                rel_ci_upper = 2 * observed_rel_effect - np.percentile(bootstrap_rel_effects_clean, alpha / 2 * 100)
            else:
                rel_ci_lower = np.nan
                rel_ci_upper = np.nan
        else:
            abs_ci_lower = np.percentile(bootstrap_abs_effects_clean, alpha / 2 * 100)
            abs_ci_upper = np.percentile(bootstrap_abs_effects_clean, (1 - alpha / 2) * 100)
            if len(bootstrap_rel_effects_clean) > 0:
                rel_ci_lower = np.percentile(bootstrap_rel_effects_clean, alpha / 2 * 100)
                rel_ci_upper = np.percentile(bootstrap_rel_effects_clean, (1 - alpha / 2) * 100)
            else:
                rel_ci_lower = np.nan
                rel_ci_upper = np.nan

        # two-sided bootstrap p-value (already two-tailed via np.abs on both sides)
        pvalue = np.mean(
            np.abs(bootstrap_abs_effects_clean - np.mean(bootstrap_abs_effects_clean)) >= np.abs(observed_abs_effect)
        )

        return {
            "standard_error": bootstrap_se,
            "pvalue": pvalue,
            "abs_effect_lower": abs_ci_lower,
            "abs_effect_upper": abs_ci_upper,
            "rel_effect_lower": rel_ci_lower,
            "rel_effect_upper": rel_ci_upper,
        }

    def clear_bootstrap_distributions(self):
        """Clear stored bootstrap distributions to free memory."""
        self._bootstrap_distributions = {}
