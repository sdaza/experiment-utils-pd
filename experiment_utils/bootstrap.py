"""Bootstrap inference mixin for ExperimentAnalyzer."""

import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm


class BootstrapMixin:
    """Mixin providing bootstrap resampling, inference, and distribution management."""

    def _get_or_create_bootstrap_indices(
        self, data: pd.DataFrame, event_col: str | None = None, cache_key: str | None = None
    ) -> list[np.ndarray]:
        """
        Get cached bootstrap indices or generate new ones if not cached.
        This enables reusing the same bootstrap samples across multiple outcomes/models,
        dramatically reducing redundant resampling (e.g., 5 outcomes × 2 models = 10x savings).

        Parameters
        ----------
        data : pd.DataFrame
            Original data to generate indices for
        event_col : str, optional
            Event column for survival models
        cache_key : str, optional
            Key for caching indices (e.g., 'treat1_vs_control_0'). If provided,
            indices are cached and reused across all outcomes/models with same key.

        Returns
        -------
        list[np.ndarray]
            List of bootstrap sample indices
        """
        # Use cache if key provided
        if cache_key is not None:
            if cache_key in self._bootstrap_indices_cache:
                self._logger.debug(f"Reusing cached bootstrap indices for '{cache_key}'")
                return self._bootstrap_indices_cache[cache_key]

        # Generate new indices
        indices = self._generate_bootstrap_indices(data, event_col=event_col)

        # Cache for reuse
        if cache_key is not None:
            self._bootstrap_indices_cache[cache_key] = indices
            self._logger.debug(f"Cached bootstrap indices for '{cache_key}'")

        return indices

    def _generate_bootstrap_indices(self, data: pd.DataFrame, event_col: str | None = None) -> list[np.ndarray]:
        """
        Pre-generate all bootstrap sample indices for reuse across outcomes and models.
        This avoids redundant resampling when running bootstrap for multiple outcomes/models.

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        event_col : str, optional
            Event column for survival models (stratifies by treatment × event)

        Returns
        -------
        list[np.ndarray]
            List of index arrays, one per bootstrap iteration
        """
        if self._bootstrap_seed is not None:
            np.random.seed(self._bootstrap_seed)

        indices_list = []
        n_total = len(data)

        for iteration in range(self._bootstrap_iterations):
            # Set seed per iteration for reproducibility
            if self._bootstrap_seed is not None:
                np.random.seed(self._bootstrap_seed + iteration)

            # Event-stratified bootstrap for survival models
            if event_col is not None and self._bootstrap_stratify:
                all_indices = []
                for treatment_val in [0, 1]:
                    for event_val in [0, 1]:
                        stratum_mask = (data[self._treatment_col] == treatment_val) & (data[event_col] == event_val)
                        stratum_indices = np.where(stratum_mask)[0]
                        if len(stratum_indices) > 0:
                            resampled = np.random.choice(stratum_indices, size=len(stratum_indices), replace=True)
                            all_indices.append(resampled)
                if all_indices:
                    iteration_indices = np.concatenate(all_indices)
                else:
                    iteration_indices = np.arange(n_total)

            # Cluster-aware bootstrap
            elif self._cluster_col is not None:
                if self._bootstrap_stratify:
                    treated_mask = data[self._treatment_col] == 1
                    control_mask = data[self._treatment_col] == 0
                    treated_clusters = data.loc[treated_mask, self._cluster_col].unique()
                    control_clusters = data.loc[control_mask, self._cluster_col].unique()

                    resampled_treated_clusters = np.random.choice(
                        treated_clusters, size=len(treated_clusters), replace=True
                    )
                    resampled_control_clusters = np.random.choice(
                        control_clusters, size=len(control_clusters), replace=True
                    )

                    treated_indices = []
                    for cluster in resampled_treated_clusters:
                        cluster_indices = np.where((data[self._cluster_col] == cluster) & treated_mask)[0]
                        treated_indices.append(cluster_indices)
                    control_indices = []
                    for cluster in resampled_control_clusters:
                        cluster_indices = np.where((data[self._cluster_col] == cluster) & control_mask)[0]
                        control_indices.append(cluster_indices)

                    iteration_indices = np.concatenate(treated_indices + control_indices)
                else:
                    clusters = data[self._cluster_col].unique()
                    resampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)
                    cluster_parts = []
                    for cluster in resampled_clusters:
                        cluster_indices = np.where(data[self._cluster_col] == cluster)[0]
                        cluster_parts.append(cluster_indices)
                    iteration_indices = np.concatenate(cluster_parts)

            # Standard row-level bootstrap
            elif self._bootstrap_stratify:
                treated_indices = np.where(data[self._treatment_col] == 1)[0]
                control_indices = np.where(data[self._treatment_col] == 0)[0]
                treated_resample = np.random.choice(treated_indices, size=len(treated_indices), replace=True)
                control_resample = np.random.choice(control_indices, size=len(control_indices), replace=True)
                iteration_indices = np.concatenate([treated_resample, control_resample])
            else:
                iteration_indices = np.random.choice(n_total, size=n_total, replace=True)

            indices_list.append(iteration_indices)

        return indices_list

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

    def _prepare_bootstrap_sample(
        self,
        data: pd.DataFrame,
        outcome: str,
        adjustment: str | None,
        relevant_covariates: list[str],
        numeric_covariates: list[str],
        binary_covariates: list[str],
        min_binary_count: int,
        model_type: str,
        iteration: int,
        event_col_for_resample: str | None = None,
        bootstrap_indices: np.ndarray | None = None,
    ) -> tuple[pd.DataFrame | None, list[str], str | None, str | None]:
        """
        Prepare a bootstrap sample (Stage 1: I/O-bound, use threading).

        This method performs data resampling, imputation, standardization, and
        balance weight computation - all memory operations suitable for threading.

        Parameters
        ----------
        data : pd.DataFrame
            Original data
        outcome : str
            Outcome variable
        adjustment : str | None
            Adjustment method
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
        iteration : int
            Iteration number for seeding
        event_col_for_resample : str | None
            Event column for survival models
        bootstrap_indices : np.ndarray | None
            Pre-generated bootstrap sample indices

        Returns
        -------
        tuple[pd.DataFrame | None, list[str], str | None, str | None]
            (prepared_data, boot_relevant_covariates, weight_col, error_info)
        """
        try:
            # Resample data
            if bootstrap_indices is not None:
                boot_data = data.iloc[bootstrap_indices].reset_index(drop=True)
            else:
                seed = self._bootstrap_seed + iteration if self._bootstrap_seed is not None else None
                boot_data = self._stratified_resample(data, seed=seed, event_col=event_col_for_resample)

            # Impute and standardize
            boot_data = self.impute_missing_values(
                data=boot_data,
                num_covariates=numeric_covariates,
                bin_covariates=binary_covariates,
            )

            boot_numeric = [c for c in numeric_covariates if boot_data[c].std(ddof=0) != 0]
            boot_binary = [c for c in binary_covariates if boot_data[c].sum() >= min_binary_count]
            boot_binary = [c for c in boot_binary if boot_data[c].std(ddof=0) != 0]
            boot_final_covariates = boot_numeric + boot_binary

            if len(boot_final_covariates) > 0:
                boot_data = self.standardize_covariates(boot_data, boot_final_covariates)

            # Compute balance weights if needed
            weight_col = None
            if adjustment == "balance":
                weight_col = self._target_weights[self._target_effect]
                if len(boot_final_covariates) == 0:
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

            boot_relevant_covariates = list(set(boot_final_covariates) & set(relevant_covariates))

            return boot_data, boot_relevant_covariates, weight_col, None

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)[:100]}"
            return None, [], None, error_info

    def _fit_bootstrap_model(
        self,
        prepared_data: pd.DataFrame,
        outcome: str,
        model_func,
        boot_relevant_covariates: list[str],
        weight_col: str | None,
        model_type: str,
        compute_marginal_effects: str | bool,
        cluster_col: str | None,
        event_col: str | None = None,
        interaction_covariates: list[str] | None = None,
    ) -> tuple[float | None, float | None, str | None]:
        """
        Fit model on prepared bootstrap sample (Stage 2: CPU-bound, use multiprocessing).

        This method performs the actual model fitting - CPU-intensive statsmodels regression
        suitable for multiprocessing.

        Parameters
        ----------
        prepared_data : pd.DataFrame
            Prepared bootstrap sample from Stage 1
        outcome : str
            Outcome variable
        model_func : callable
            Model function to use for estimation
        boot_relevant_covariates : list[str]
            Relevant covariates for this sample
        weight_col : str | None
            Weight column for balance adjustment
        model_type : str
            Model type
        compute_marginal_effects : str | bool
            Whether to compute marginal effects
        cluster_col : str | None
            Cluster column for clustered standard errors
        event_col : str | None
            Event column for survival models
        interaction_covariates : list[str] | None
            Covariates to include as treatment interactions

        Returns
        -------
        tuple[float | None, float | None, str | None]
            (absolute_effect, relative_effect, error_info)
        """
        try:
            estimator_params = {
                "data": prepared_data,
                "outcome_variable": outcome,
                "covariates": boot_relevant_covariates,
                "cluster_col": cluster_col,
                "store_model": False,
                "compute_relative_ci": False,
            }

            if model_type in ["logistic", "poisson", "negative_binomial"]:
                estimator_params["compute_marginal_effects"] = compute_marginal_effects

            if model_type == "cox":
                if event_col:
                    estimator_params["event_col"] = event_col
                else:
                    raise ValueError(f"Cox model for outcome '{outcome}' requires event column")

            if weight_col:
                estimator_params["weight_column"] = weight_col

            if interaction_covariates and model_type == "ols":
                estimator_params["interaction_covariates"] = interaction_covariates

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                warnings.filterwarnings("ignore", message="cov_type not fully supported with freq_weights")
                output = model_func(**estimator_params)

            return output["absolute_effect"], output["relative_effect"], None

        except Exception as e:
            error_info = f"{type(e).__name__}: {str(e)[:100]}"
            return None, None, error_info

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
        interaction_covariates: list[str] | None = None,
        bootstrap_indices: np.ndarray | None = None,
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
        interaction_covariates : list[str] | None
            Covariates to include as treatment interactions (z_{col} + treatment:z_{col}).
            z_{col} columns are resampled from data alongside the outcome. The regression
            is re-fit on each resample, so interaction coefficients (θ* in CUPED terms)
            are automatically re-estimated — giving proper bootstrap standard errors.
        bootstrap_indices : np.ndarray | None
            Pre-generated bootstrap sample indices. If provided, uses these instead of
            resampling (more efficient when running multiple outcomes/models).

        Returns
        -------
        tuple[float | None, float | None, str | None]
            (absolute_effect, relative_effect, error_info)
        """
        # Stage 1: Prepare bootstrap sample (I/O-bound)
        boot_data, boot_relevant_covariates, weight_col, prep_error = self._prepare_bootstrap_sample(
            data=data,
            outcome=outcome,
            adjustment=adjustment,
            relevant_covariates=relevant_covariates,
            numeric_covariates=numeric_covariates,
            binary_covariates=binary_covariates,
            min_binary_count=min_binary_count,
            model_type=model_type,
            iteration=iteration,
            event_col_for_resample=event_col_for_resample,
            bootstrap_indices=bootstrap_indices,
        )

        if prep_error is not None:
            return None, None, prep_error

        # Stage 2: Fit model (CPU-bound)
        event_col = None
        if model_type == "cox":
            if outcome in self._outcome_event_cols:
                event_col = self._outcome_event_cols[outcome]
            elif self._event_col:
                event_col = self._event_col

        return self._fit_bootstrap_model(
            prepared_data=boot_data,
            outcome=outcome,
            model_func=model_func,
            boot_relevant_covariates=boot_relevant_covariates,
            weight_col=weight_col,
            model_type=model_type,
            compute_marginal_effects=compute_marginal_effects,
            cluster_col=self._cluster_col,
            event_col=event_col,
            interaction_covariates=interaction_covariates,
        )

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
        interaction_covariates: list[str] | None = None,
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
        interaction_covariates : list[str] | None
            Covariates to include as treatment interactions. Forwarded to each iteration
            so the interaction coefficients are re-estimated on every resample.

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

        # Create cache key to reuse indices across outcomes/models
        # Key format: "data_hash_eventcol" - ensures same data+stratification = same indices
        data_hash = str(len(data))  # Simple hash based on data size
        cache_key = f"bootstrap_{data_hash}_{event_col_for_resample}"

        # Get or generate bootstrap indices (cached for reuse across outcomes/models)
        # MAJOR OPTIMIZATION: If you have 5 outcomes × 2 models, this reuses the same 1000 samples
        # instead of generating 10,000 samples (10x speedup!)
        bootstrap_indices_list = self._get_or_create_bootstrap_indices(
            data, event_col=event_col_for_resample, cache_key=cache_key
        )

        if cache_key not in self._bootstrap_indices_cache:
            # First time generating for this cache key
            self._logger.info(
                f"Generated {self._bootstrap_iterations} bootstrap sample indices "
                f"for cache key '{cache_key}' (will reuse across outcomes/models)"
            )

        use_parallel = self._bootstrap_iterations >= 100

        if use_parallel:
            from joblib import Parallel, delayed

            # Hybrid parallelization strategy (two-pass threading approach)
            # Unlike PowerSim's threading+multiprocessing, bootstrap uses threading for both stages
            # because: (1) DataFrames benefit from shared memory, (2) model_func can't be pickled,
            # (3) statsmodels releases GIL so threading is efficient
            if self._bootstrap_backend == "hybrid":
                self._logger.info(
                    f"Running bootstrap for outcome '{outcome}' model '{model_type}' with "
                    f"{self._bootstrap_iterations} iterations using hybrid strategy "
                    f"(two-pass threading: data prep → model fitting)..."
                )

                # Stage 1: Prepare all bootstrap samples using threading (I/O-bound, memory-efficient)
                self._logger.debug("Stage 1: Preparing bootstrap samples with threading...")
                prepared_samples = Parallel(n_jobs=-1, backend="threading")(
                    delayed(self._prepare_bootstrap_sample)(
                        data=data,
                        outcome=outcome,
                        adjustment=adjustment,
                        relevant_covariates=relevant_covariates,
                        numeric_covariates=numeric_covariates,
                        binary_covariates=binary_covariates,
                        min_binary_count=min_binary_count,
                        model_type=model_type,
                        iteration=i,
                        event_col_for_resample=event_col_for_resample,
                        bootstrap_indices=bootstrap_indices_list[i],
                    )
                    for i in range(self._bootstrap_iterations)
                )

                # Get event column for Stage 2
                event_col = None
                if model_type == "cox":
                    if outcome in self._outcome_event_cols:
                        event_col = self._outcome_event_cols[outcome]
                    elif self._event_col:
                        event_col = self._event_col

                # Stage 2: Fit models on prepared samples using threading (CPU-bound but releases GIL)
                # Note: Using threading instead of multiprocessing because statsmodels releases GIL
                # and we avoid expensive DataFrame serialization
                self._logger.debug("Stage 2: Fitting models with threading...")
                valid_samples = [(d, c, w, e) for d, c, w, e in prepared_samples if e is None]
                results = list(
                    tqdm(
                        Parallel(n_jobs=-1, backend="threading", return_as="generator")(
                            delayed(self._fit_bootstrap_model)(
                                prepared_data=boot_data,
                                outcome=outcome,
                                model_func=model_func,
                                boot_relevant_covariates=boot_relevant_covariates,
                                weight_col=weight_col,
                                model_type=model_type,
                                compute_marginal_effects=compute_marginal_effects,
                                cluster_col=self._cluster_col,
                                event_col=event_col,
                                interaction_covariates=interaction_covariates,
                            )
                            for boot_data, boot_relevant_covariates, weight_col, _ in valid_samples
                        ),
                        total=len(valid_samples),
                        desc=f"Bootstrap [{outcome}]",
                        unit="iter",
                    )
                )

                # Collect errors from Stage 1
                prep_errors = [prep_error for _, _, _, prep_error in prepared_samples if prep_error is not None]

                # Combine results and count errors
                bootstrap_abs_effects = []
                bootstrap_rel_effects = []
                error_counts = {}

                # Add Stage 1 errors
                for error_info in prep_errors:
                    error_type = error_info.split(":")[0] if ":" in error_info else "UnknownError"
                    error_message = error_info.split(":", 1)[1].strip() if ":" in error_info else error_info
                    if error_type not in error_counts:
                        error_counts[error_type] = {"count": 0, "first_message": error_message}
                    error_counts[error_type]["count"] += 1

                # Add Stage 2 results and errors
                for abs_eff, rel_eff, error_info in results:
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
                # Non-hybrid: use single backend for entire pipeline (original behavior)
                self._logger.info(
                    f"Running bootstrap for outcome '{outcome}' model '{model_type}' with "
                    f"{self._bootstrap_iterations} iterations in parallel (backend: {self._bootstrap_backend})..."
                )

                # Use configured backend (threading or loky)
                results = list(
                    tqdm(
                        Parallel(n_jobs=-1, backend=self._bootstrap_backend, prefer="threads", return_as="generator")(
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
                                interaction_covariates=interaction_covariates,
                                bootstrap_indices=bootstrap_indices_list[i],
                            )
                            for i in range(self._bootstrap_iterations)
                        ),
                        total=self._bootstrap_iterations,
                        desc=f"Bootstrap [{outcome}]",
                        unit="iter",
                    )
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

            for i in tqdm(range(self._bootstrap_iterations), desc=f"Bootstrap [{outcome}]", unit="iter"):
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
                    interaction_covariates=interaction_covariates,
                    bootstrap_indices=bootstrap_indices_list[i],  # Use pre-generated indices
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
