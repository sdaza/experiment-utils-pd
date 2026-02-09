"""
ExperimentAnalyzer class to analyze and design experiments
"""

import re as _re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.proportion import proportions_ztest

from .bootstrap import BootstrapMixin
from .estimators import Estimators
from .retrodesign import RetrodesignMixin
from .utils import get_logger, log_and_raise_error


def _clean_category_name(cat):
    """Convert category to lowercase and replace spaces/special chars with underscores"""
    cat_str = str(cat).lower()
    cat_str = _re.sub(r"[^\w]+", "_", cat_str)
    cat_str = cat_str.strip("_")
    return cat_str


class ExperimentAnalyzer(BootstrapMixin, RetrodesignMixin):
    """
    Class ExperimentAnalyzer to analyze and design experiments

    Parameters
    ----------
    data : pd.DataFrame
        Pandas Dataframe
    outcomes : List
        List of outcome variables
    covariates : List
        List of covariates. Can include:
        - Numeric columns (continuous)
        - Binary columns (0/1)
        - Categorical columns (object/category dtype or integers with low cardinality)
          Categorical columns are automatically converted to dummy variables.
          Reference category (most frequent) is excluded from regression models
          but included in balance tables.
          Use categorical_max_unique to control the threshold for treating integers as categorical.
    treatment_col : str
        Column name for the treatment variable
    experiment_identifier : List
        List of columns to identify an experiment
    adjustment : str, optional
        Covariate adjustment method ('balance', 'IV'), by default None
    exp_sample_ratio_col : str, optional
        Column name for the expected sample ratio, by default None
    target_ipw_effect : str, optional
        Target IPW effect (ATT, ATE, ATC), by default "ATT"
    balance_method : str, optional
        Balance method ('ps-logistic', 'ps-xgboost', 'entropy'), by default 'ps-logistic'
    min_ps_score : float, optional
        Minimum propensity score, by default 0.05
    max_ps_score : float, optional
        Maximum propensity score, by default 0.95
    polynomial_ipw : bool, optional
        Use polynomial and interaction features for IPW, by default False. It can be slow for large datasets.
    assess_overlap : bool, optional
        Assess overlap between treatment and control groups (slow) when using balance to adjust covariates, by default False
    overlap_plot: bool, optional
        Plot overlap between treatment and control groups, by default False
    assess_overlap : bool, optional
    instrument_col : str, optional
        Column name for the instrument variable, by default None
    alpha : float, optional
        Significance level, by default 0.05
    regression_covariates : List, optional
        List of covariates to include in the final linear regression model, by default None
    unit_identifier : str, optional
        Column name for the unit/user identifier to connect weights to specific units, by default None
    bootstrap : bool, optional
        Whether to use bootstrap inference for p-values and confidence intervals, by default False
    bootstrap_iterations : int, optional
        Number of bootstrap iterations, by default 1000
    bootstrap_ci_method : str, optional
        Bootstrap CI method ('percentile', 'basic'), by default 'percentile'
    bootstrap_stratify : bool, optional
        Whether to stratify bootstrap resampling by treatment group, by default True
    bootstrap_seed : int, optional
        Random seed for bootstrap reproducibility, by default None
    skip_bootstrap_for_survival : bool, optional
        Skip bootstrap for Cox survival models and use model's robust standard errors instead.
        Recommended when event rate < 30% or with IPW adjustment. By default False.
        Even when bootstrap=True, this will skip bootstrap specifically for Cox models.
    bootstrap_fixed_weights : bool, optional
        For IPW adjustment (adjustment='balance'), use fixed weights from original data
        instead of recalculating on each bootstrap sample. Faster and reduces variance.
        By default False (recalculates weights on each sample).
    pvalue_adjustment : str, optional
        P-value adjustment method to apply automatically after get_effects ('bonferroni', 'holm', 'fdr_bh', 'sidak', 'hommel', 'hochberg', 'by', or None for no adjustment), by default 'bonferroni'
    categorical_max_unique : int, optional
        Maximum number of unique values for numeric columns to be treated as categorical.
        Numeric columns (int or float) with 3 to categorical_max_unique unique values
        will be converted to dummy variables.
        Binary numeric columns (2 unique values) are automatically detected and treated as binary covariates.
        Set to 2 to disable categorical treatment for numeric columns (nothing will match 3 <= n <= 2).
        Set to 5, 10, or higher for more inclusive behavior.
        By default 2 (effectively disables categorical treatment for numeric columns).
    outcome_models : dict[str, str | list[str]] | str | list[str] | None, optional
        Specify model type(s) per outcome or globally. By default None (uses OLS for all outcomes).
        - Dict: {"revenue": "ols", "clicked": "logistic", "orders": "poisson"}
        - Dict with lists: {"clicked": ["ols", "logistic"], "orders": ["poisson", "negative_binomial"]}
        - String: "logistic" applies to all outcomes
        - List: ["ols", "logistic"] applies all models to all outcomes
        Supported models: "ols", "logistic", "poisson", "negative_binomial", "cox"
        When multiple models specified for same outcome, results include all model outputs with model_type column
    cluster_col : str, optional
        Column name for clustered standard errors (e.g., "user_id" for repeated measures).
        By default None (no clustering).
    compute_marginal_effects : str | bool, optional
        For GLMs, control marginal effects computation. By default "overall".
        - "overall": Average Marginal Effect (AME) - average across all observations
        - "mean": Marginal Effect at the Mean (MEM) - at mean of covariates
        - "median": Marginal effect at median of covariates
        - True: Same as "overall" (backward compatibility)
        - False: Return odds ratios / rate ratios instead of marginal effects
        Results interpretation:
        - Logistic: change in probability (percentage points) or odds ratio
        - Poisson/NegBin: change in expected count or rate ratio
    store_fitted_models : bool, optional
        Store fitted model instances for retrieval via get_fitted_models(). By default True.
        Enables post-hoc analysis, diagnostics, predictions, and accurate retrodesign simulation.
    event_col : str, optional
        Column name for event indicator in survival analysis (1=event, 0=censored).
        By default None. Alternative: specify Cox outcomes as tuples in the outcomes list.

        Two ways to specify Cox models:
        1. Tuple notation (recommended): outcomes=[("time_col", "event_col")]
        2. Separate parameter: outcomes=["time_col"], event_col="event_col"
    """  # noqa: E501

    def __init__(
        self,
        data: pd.DataFrame,
        outcomes: list[str],
        treatment_col: str,
        experiment_identifier: list[str] | None = None,
        covariates: list[str] | None = None,
        adjustment: str | None = None,
        exp_sample_ratio_col: str | None = None,
        target_effect: str = "ATT",
        balance_method: str = "ps-logistic",
        min_ps_score: float = 0.05,
        max_ps_score: float = 0.95,
        polynomial_ipw: bool = False,
        instrument_col: str | None = None,
        alpha: float = 0.05,
        regression_covariates: list[str] | None = None,
        assess_overlap: bool = False,
        overlap_plot: bool = False,
        unit_identifier: str | None = None,
        bootstrap: bool = False,
        bootstrap_iterations: int = 1000,
        bootstrap_ci_method: str = "percentile",
        bootstrap_stratify: bool = True,
        bootstrap_seed: int | None = None,
        skip_bootstrap_for_survival: bool = False,
        bootstrap_fixed_weights: bool = False,
        treatment_comparisons: list[tuple] | None = None,
        pvalue_adjustment: str | None = None,
        categorical_max_unique: int = 2,
        outcome_models: dict[str, str | list[str]] | str | list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = True,
        store_fitted_models: bool = True,
        event_col: str | None = None,
    ) -> None:
        self._logger = get_logger("Experiment Analyzer")
        self._data = data.copy()

        # process outcomes - can be strings or tuples for Cox models
        self._outcomes = []
        self._outcome_event_cols = {}  # Map outcome -> event_col for Cox models
        for outcome in self.__ensure_list(outcomes):
            if isinstance(outcome, tuple):
                # cox model: (time_col, event_col)
                if len(outcome) != 2:
                    log_and_raise_error(
                        self._logger,
                        f"Cox outcome tuple must have exactly 2 elements: (time_col, event_col). Got: {outcome}",
                    )
                time_col, event_col_name = outcome
                self._outcomes.append(time_col)
                self._outcome_event_cols[time_col] = event_col_name
            else:
                # regular outcome
                self._outcomes.append(outcome)

        self._covariates = self.__ensure_list(covariates)
        self._treatment_col = treatment_col
        self._experiment_identifier = self.__ensure_list(experiment_identifier)
        self._adjustment = adjustment
        self._exp_sample_ratio_col = exp_sample_ratio_col
        self._balance_method = balance_method
        self._target_effect = target_effect
        self._assess_overlap = assess_overlap
        self._overlap_plot = overlap_plot
        self._instrument_col = instrument_col
        self._regression_covariates = self.__ensure_list(regression_covariates)
        self._unit_identifier = unit_identifier
        self._bootstrap = bootstrap
        self._bootstrap_iterations = bootstrap_iterations
        self._bootstrap_ci_method = bootstrap_ci_method
        self._bootstrap_stratify = bootstrap_stratify
        self._bootstrap_seed = bootstrap_seed
        self._skip_bootstrap_for_survival = skip_bootstrap_for_survival
        self._bootstrap_fixed_weights = bootstrap_fixed_weights
        self._pvalue_adjustment = pvalue_adjustment
        self._treatment_comparisons = treatment_comparisons
        self._categorical_max_unique = categorical_max_unique
        self._outcome_models = outcome_models
        self._cluster_col = cluster_col
        self._compute_marginal_effects = compute_marginal_effects
        self._store_fitted_models = store_fitted_models
        self._event_col = event_col
        self.__check_input()
        self._alpha = alpha
        self._results = None
        self._balance = []
        self._adjusted_balance = []
        self._weights = None
        self._final_covariates = []
        self._fitted_models = {}
        self._bootstrap_distributions = {}  # Store bootstrap distributions for MCP adjustment
        self._target_weights = {
            "ATT": "tips_stabilized_weight",
            "ATE": "ips_stabilized_weight",
            "ATC": "cips_stabilized_weight",
        }  # noqa: E501
        self._estimator = Estimators(
            treatment_col,
            instrument_col,
            target_effect,
            self._target_weights,
            alpha,
            min_ps_score,
            max_ps_score,
            polynomial_ipw,
        )  # noqa: E501

    def __check_input(self) -> None:
        if self._data.empty:
            log_and_raise_error(self._logger, "Dataframe is empty!")

        if (len(self._covariates) == 0) & (len(self._regression_covariates) > 0):
            self._covariates = self._regression_covariates

        string_cols = [c for c in self._covariates if pd.api.types.is_string_dtype(self._data[c])]
        if string_cols:
            self._logger.info(f"Detected categorical covariates (will create dummies): {string_cols}")

        if len(self._regression_covariates) > 0:
            if not set(self._regression_covariates).issubset(set(self._covariates)):
                log_and_raise_error(self._logger, "Regression covariates should be a subset of covariates")

        if len(self._experiment_identifier) == 0:
            self._data["experiment_id"] = 1
            self._experiment_identifier = ["experiment_id"]
            self._logger.warning("No experiment identifier, assuming data is from a single experiment!")

        required_columns = (
            self._experiment_identifier
            + [self._treatment_col]
            + self._outcomes
            + list(self._outcome_event_cols.values())  # Event columns from tuple notation
            + self._covariates
            + ([self._instrument_col] if self._instrument_col is not None else [])
            + ([self._exp_sample_ratio_col] if self._exp_sample_ratio_col is not None else [])
            + ([self._unit_identifier] if self._unit_identifier is not None else [])
            + ([self._cluster_col] if self._cluster_col is not None else [])
            + ([self._event_col] if self._event_col is not None else [])
        )

        missing_columns = set(required_columns) - set(self._data.columns)

        if missing_columns:
            log_and_raise_error(
                self._logger, f"The following required columns are missing from the dataframe: {missing_columns}"
            )  # noqa: E501
        if len(self._covariates) == 0:
            self._logger.warning("No covariates specified, balance can't be assessed!")

        if self._outcome_models is not None:
            valid_models = {"ols", "logistic", "poisson", "negative_binomial", "cox"}
            if isinstance(self._outcome_models, dict):
                invalid_outcomes = set(self._outcome_models.keys()) - set(self._outcomes)
                if invalid_outcomes:
                    log_and_raise_error(
                        self._logger,
                        f"outcome_models contains invalid outcome names: {invalid_outcomes}. "
                        f"Valid outcomes: {self._outcomes}",
                    )
                all_invalid_models = set()
                for outcome, models in self._outcome_models.items():
                    if isinstance(models, str):
                        if models not in valid_models:
                            all_invalid_models.add(models)
                    elif isinstance(models, list):
                        if len(models) != len(set(models)):
                            log_and_raise_error(
                                self._logger, f"outcome_models for '{outcome}' contains duplicate models: {models}"
                            )
                        if len(models) == 0:
                            log_and_raise_error(self._logger, f"outcome_models for '{outcome}' cannot be an empty list")
                        for model in models:
                            if not isinstance(model, str):
                                log_and_raise_error(
                                    self._logger,
                                    f"outcome_models for '{outcome}' must contain strings. Got: {type(model)}",
                                )
                            if model not in valid_models:
                                all_invalid_models.add(model)
                    else:
                        log_and_raise_error(
                            self._logger,
                            f"outcome_models values must be str or list[str]. Got {type(models)} for '{outcome}'",
                        )
                if all_invalid_models:
                    log_and_raise_error(
                        self._logger,
                        f"outcome_models contains invalid model types: {all_invalid_models}. "
                        f"Valid models: {valid_models}",
                    )
            elif isinstance(self._outcome_models, str):
                if self._outcome_models not in valid_models:
                    log_and_raise_error(
                        self._logger,
                        f"outcome_models '{self._outcome_models}' is invalid. Valid models: {valid_models}",
                    )
            elif isinstance(self._outcome_models, list):
                if len(self._outcome_models) != len(set(self._outcome_models)):
                    log_and_raise_error(
                        self._logger, f"outcome_models list contains duplicates: {self._outcome_models}"
                    )
                if len(self._outcome_models) == 0:
                    log_and_raise_error(self._logger, "outcome_models cannot be an empty list")
                for model in self._outcome_models:
                    if not isinstance(model, str):
                        log_and_raise_error(
                            self._logger, f"outcome_models list must contain strings. Got: {type(model)}"
                        )
                    if model not in valid_models:
                        log_and_raise_error(
                            self._logger,
                            f"outcome_models contains invalid model type '{model}'. Valid models: {valid_models}",
                        )
            else:
                log_and_raise_error(
                    self._logger, f"outcome_models must be dict, str, list, or None. Got: {type(self._outcome_models)}"
                )

        if self._outcome_models is not None:
            models_to_check = []
            if isinstance(self._outcome_models, dict):
                for models in self._outcome_models.values():
                    if isinstance(models, str):
                        models_to_check.append(models)
                    elif isinstance(models, list):
                        models_to_check.extend(models)
            elif isinstance(self._outcome_models, str):
                models_to_check = [self._outcome_models]
            elif isinstance(self._outcome_models, list):
                models_to_check = self._outcome_models

            if "cox" in models_to_check:
                cox_outcomes = []
                if isinstance(self._outcome_models, dict):
                    for k, v in self._outcome_models.items():
                        if v == "cox" or (isinstance(v, list) and "cox" in v):
                            cox_outcomes.append(k)
                elif self._outcome_models == "cox" or (
                    isinstance(self._outcome_models, list) and "cox" in self._outcome_models
                ):
                    cox_outcomes = self._outcomes

                for outcome in cox_outcomes:
                    if outcome not in self._outcome_event_cols and self._event_col is None:
                        log_and_raise_error(
                            self._logger,
                            f"Cox model for outcome '{outcome}' requires event column. "
                            f"Use tuple notation: outcomes=[('{outcome}', 'event_col')] "
                            f"or set event_col parameter.",
                        )

        self._normalized_outcome_models = {}
        if self._outcome_models is not None:
            if isinstance(self._outcome_models, str):
                self._normalized_outcome_models = {outcome: [self._outcome_models] for outcome in self._outcomes}
            elif isinstance(self._outcome_models, list):
                self._normalized_outcome_models = {outcome: self._outcome_models.copy() for outcome in self._outcomes}
            elif isinstance(self._outcome_models, dict):
                for outcome in self._outcomes:
                    models = self._outcome_models.get(outcome, ["ols"])
                    if isinstance(models, str):
                        self._normalized_outcome_models[outcome] = [models]
                    else:
                        self._normalized_outcome_models[outcome] = models.copy()
        else:
            self._normalized_outcome_models = {outcome: ["ols"] for outcome in self._outcomes}

        valid_me_values = {False, "overall", "mean", "median", True}
        if self._compute_marginal_effects not in valid_me_values:
            log_and_raise_error(
                self._logger,
                f"compute_marginal_effects must be one of {valid_me_values}. Got: {self._compute_marginal_effects}",
            )
        if self._compute_marginal_effects is True:
            self._compute_marginal_effects = "overall"

        self._data = self._data[required_columns]

    def __get_binary_covariates(self, data: pd.DataFrame, exclude_categoricals: set[str] = None) -> list[str]:
        """Get binary covariates, optionally excluding categorical columns that were converted to dummies.

        Detects any numeric column with exactly 2 unique values (e.g., 0/1, 10.0/11.0).
        """
        binary_covariates = []
        if exclude_categoricals is None:
            exclude_categoricals = set()
        if self._covariates is not None:
            for c in self._covariates:
                if c in exclude_categoricals:
                    continue  # Skip categorical columns that are now dummies
                if pd.api.types.is_numeric_dtype(data[c]) and data[c].nunique() == 2:
                    binary_covariates.append(c)
        return binary_covariates

    def __get_numeric_covariates(self, data: pd.DataFrame, exclude_categoricals: set[str] = None) -> list[str]:
        """Get numeric covariates, optionally excluding categorical columns that were converted to dummies."""
        numeric_covariates = []
        if exclude_categoricals is None:
            exclude_categoricals = set()
        if self._covariates is not None:
            for c in self._covariates:
                if c in exclude_categoricals:
                    continue  # Skip categorical columns that are now dummies
                if data[c].nunique() > 2:
                    numeric_covariates.append(c)
        return numeric_covariates

    def __get_categorical_covariates(self, data: pd.DataFrame) -> dict[str, list]:
        """
        Detect categorical covariates and return their unique categories.

        Detection rules:
        - object/category dtype columns
        - numeric columns (int or float) with 3 to categorical_max_unique unique values
          (excludes binary columns with 2 values since those are handled by __get_binary_covariates)

        Returns dict: {covariate_name: [list_of_categories]}
        """
        categorical_info = {}
        if self._covariates is not None:
            for c in self._covariates:
                is_object = pd.api.types.is_object_dtype(data[c]) or isinstance(data[c].dtype, pd.CategoricalDtype)

                # treat numeric columns (int or float) as categorical if they have
                # 3 to categorical_max_unique unique values
                # exclude binary (2 values) since those are handled by __get_binary_covariates
                is_low_cardinality_numeric = (
                    pd.api.types.is_numeric_dtype(data[c]) and 3 <= data[c].nunique() <= self._categorical_max_unique
                )

                if is_object or is_low_cardinality_numeric:
                    categories = sorted(data[c].dropna().unique())
                    categorical_info[c] = categories

        return categorical_info

    def __create_dummy_variables(
        self,
        data: pd.DataFrame,
        categorical_info: dict[str, list],
        reference_categories: dict[str, any] | None = None,
        include_reference: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, str], dict[str, any]]:
        """
        Create dummy variables for categorical covariates.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        categorical_info : dict
            Dict of {covariate: [categories]}
        reference_categories : dict, optional
            Dict of {covariate: reference_category}. If not provided, uses most frequent.
        include_reference : bool
            If True, include reference category dummy (for balance tables)
            If False, drop reference category (for regression models)

        Returns
        -------
        data : pd.DataFrame
            Data with dummy columns added
        dummy_cols_map : dict
            Mapping of {dummy_col_name: original_covariate}
        reference_map : dict
            Mapping of {covariate: reference_category_used}
        """
        dummy_cols_map = {}
        reference_map = {}

        for covariate, categories in categorical_info.items():
            if reference_categories and covariate in reference_categories:
                ref_cat = reference_categories[covariate]
                if ref_cat not in categories:
                    self._logger.warning(f"Reference category {ref_cat} not found in {covariate}, using most frequent")
                    ref_cat = data[covariate].value_counts().idxmax()
            else:
                ref_cat = data[covariate].value_counts().idxmax()

            reference_map[covariate] = ref_cat

            for cat in categories:
                if not include_reference and cat == ref_cat:
                    continue

                cat_clean = _clean_category_name(cat)
                dummy_col = f"{covariate}_{cat_clean}"
                data[dummy_col] = (data[covariate] == cat).astype(int)
                dummy_cols_map[dummy_col] = covariate

        return data, dummy_cols_map, reference_map

    def impute_missing_values(
        self, data: pd.DataFrame, num_covariates: list[str] | None = None, bin_covariates: list[str] | None = None
    ) -> pd.DataFrame:  # noqa: E501
        """ "
        Impute missing values for numeric and binary covariates
        """
        for cov in num_covariates:
            if data[cov].isna().all():
                log_and_raise_error(self._logger, f"Column {cov} has only missing values")
            data[cov] = data[cov].fillna(data[cov].mean())

        for cov in bin_covariates:
            if data[cov].isna().all():
                log_and_raise_error(self._logger, f"Column {cov} has only missing values.")
            data[cov] = data[cov].fillna(data[cov].mode()[0])

        return data

    def __handle_missing_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Handle missing data in treatment and categorical covariates.

        Missing data handling strategy:
        - Treatment column: Drop rows with missing values (REQUIRED)
        - Categorical covariates: Create explicit 'Missing' category

        Parameters
        ----------
        data : pd.DataFrame
            Input data

        Returns
        -------
        data : pd.DataFrame
            Data with missing values handled
        summary : dict
            Summary of missing data handling
        """
        original_n = len(data)
        rows_dropped = 0
        missing_info = []

        treatment_missing = data[self._treatment_col].isna().sum()
        if treatment_missing > 0:
            pct_missing = (treatment_missing / original_n) * 100
            self._logger.warning(
                f"Treatment column '{self._treatment_col}' has {treatment_missing} missing values "
                f"({pct_missing:.1f}%). Dropping these rows."
            )
            data = data[data[self._treatment_col].notna()].copy()
            rows_dropped += treatment_missing
            missing_info.append(f"treatment: {treatment_missing} rows")

        if self._covariates:
            for cov in self._covariates:
                if cov not in data.columns:
                    continue

                is_likely_categorical = (
                    pd.api.types.is_object_dtype(data[cov])
                    or isinstance(data[cov].dtype, pd.CategoricalDtype)
                    or (
                        pd.api.types.is_numeric_dtype(data[cov])
                        and 3 <= data[cov].nunique() <= self._categorical_max_unique
                    )
                )

                cov_missing = data[cov].isna().sum()
                if cov_missing > 0 and is_likely_categorical:
                    pct_missing = (cov_missing / len(data)) * 100
                    self._logger.warning(
                        f"Categorical covariate '{cov}' has {cov_missing} missing values "
                        f"({pct_missing:.1f}%). Creating explicit 'Missing' category."
                    )
                    data[cov] = data[cov].fillna("Missing")
                    missing_info.append(f"{cov}: {cov_missing} values")

        summary = {
            "rows_dropped": rows_dropped,
            "pct_dropped": (rows_dropped / original_n * 100) if original_n > 0 else 0,
            "missing_details": missing_info,
        }

        return data, summary

    def standardize_covariates(self, data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        """
        Standardize covariates in the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data to standardize
        covariates : list[str]
            List of covariates to standardize

        Returns
        -------
        pd.DataFrame
            Data with standardized covariates
        """

        for covariate in covariates:
            data[f"z_{covariate}"] = (data[covariate] - data[covariate].mean()) / data[covariate].std()
        return data

    def calculate_smd(
        self,
        data: pd.DataFrame,
        treatment_col: str = None,
        covariates: list[str] | None = None,
        weights_col: str = "weights",
        threshold: float = 0.1,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Calculate standardized mean differences (SMDs) between treatment and control groups.

        Parameters
        ----------
        data : DataFrame, optional
            DataFrame containing the data to calculate SMDs on. If None, uses the data from the class.
        treatment_col : str, optional
            Name of the column containing the treatment assignment.
        covariates : list, optional
            List of column names to calculate SMDs for. If None, uses all numeric and binary covariates.
        weights_col : str, optional
            Name of the column to use for weighting the means. Defaults to 'weights'.
        threshold : float, optional
            Threshold to determine if a covariate is balanced. Defaults to 0.1.

        Returns
        -------
        DataFrame
            DataFrame containing the SMDs and balance flags for each covariate.
        """

        if treatment_col is None:
            treatment_col = self._treatment_col

        treated = data[data[treatment_col] == 1]
        control = data[data[treatment_col] == 0]

        if covariates is None:
            covariates = self._final_covariates

        smd_results = []
        for cov in covariates:
            mean_treated = np.average(treated[cov], weights=treated[weights_col])
            mean_control = np.average(control[cov], weights=control[weights_col])

            var_treated = np.average((treated[cov] - mean_treated) ** 2, weights=treated[weights_col])
            var_control = np.average((control[cov] - mean_control) ** 2, weights=control[weights_col])

            pooled_std = np.sqrt((var_treated + var_control) / 2)

            smd = (mean_treated - mean_control) / pooled_std if pooled_std != 0 else 0

            balance_flag = 1 if abs(smd) <= threshold else 0

            smd_results.append(
                {
                    "covariate": cov,
                    "mean_treated": mean_treated,
                    "mean_control": mean_control,
                    "smd": smd,
                    "balance_flag": balance_flag,
                }
            )

        smd_df = pd.DataFrame(smd_results)

        return smd_df

    def __get_estimator_function(self, model_type: str, adjustment: str | None, outcome: str):
        """
        Get the appropriate estimator function based on model type and adjustment.

        Parameters
        ----------
        model_type : str
            Model type: 'ols', 'logistic', 'poisson', 'negative_binomial', 'cox'
        adjustment : str | None
            Adjustment type: None, 'balance', 'IV'
        outcome : str
            Outcome name (for error messages)

        Returns
        -------
        callable
            Estimator function
        """
        if adjustment == "IV" and model_type != "ols":
            self._logger.warning(
                f"IV adjustment not supported for {model_type} model. "
                f"Falling back to unadjusted estimation for outcome '{outcome}'."
            )
            adjustment = None

        if adjustment == "aipw" and model_type == "cox":
            self._logger.warning(
                "AIPW adjustment not supported for Cox survival models "
                "(requires specialized survival AIPW methodology). "
                f"Falling back to unadjusted estimation for outcome '{outcome}'."
            )
            adjustment = None

        if adjustment == "balance" and model_type == "cox":
            regression_covs_for_outcome = set(self._final_covariates) & set(self._regression_covariates)
            if len(regression_covs_for_outcome) == 0:
                self._logger.warning(
                    "IPW without regression covariates for Cox models estimates the marginal hazard ratio, "
                    "which differs from the conditional HR due to non-collapsibility of the hazard ratio. "
                    "Consider adding regression_covariates for IPW+Regression to recover the conditional HR."
                )

        estimator_map = {
            ("ols", None): self._estimator.linear_regression,
            ("ols", "balance"): self._estimator.weighted_least_squares,
            ("ols", "IV"): self._estimator.iv_regression,
            ("ols", "aipw"): self._estimator.aipw_ols,
            ("logistic", None): self._estimator.logistic_regression,
            ("logistic", "balance"): self._estimator.weighted_logistic_regression,
            ("logistic", "aipw"): self._estimator.aipw_logistic,
            ("poisson", None): self._estimator.poisson_regression,
            ("poisson", "balance"): self._estimator.weighted_poisson_regression,
            ("poisson", "aipw"): self._estimator.aipw_poisson,
            ("negative_binomial", None): self._estimator.negative_binomial_regression,
            ("negative_binomial", "balance"): self._estimator.weighted_negative_binomial_regression,
            ("negative_binomial", "aipw"): self._estimator.aipw_negative_binomial,
            ("cox", None): self._estimator.cox_proportional_hazards,
            ("cox", "balance"): self._estimator.weighted_cox_proportional_hazards,
        }

        key = (model_type, adjustment)
        if key not in estimator_map:
            log_and_raise_error(
                self._logger, f"No estimator found for model_type='{model_type}' with adjustment='{adjustment}'"
            )

        return estimator_map[key]

    def get_overlap_coefficient(
        self,
        treatment_scores: np.ndarray,
        control_scores: np.ndarray,
        grid_points: int = 1000,
        bw_method: float | None = None,
    ) -> float:  # noqa: E501
        """
        Calculate the Overlap Coefficient between treatment and control propensity scores.

        Parameters
        ----------
        treatment_scores : array-like
            Array of treatment propensity scores.
        control_scores : array-like
            Array of control propensity scores.
        grid_points : int, optional
            number of points to evaluate KDE on (default is 10000 for higher resolution)
        bw_method : float, optional
            Bandwidth method for the KDE estimation. Defaults to 0.1

        Returns
        -------
        float
        """

        kde_treatment = gaussian_kde(treatment_scores, bw_method=bw_method)
        kde_control = gaussian_kde(control_scores, bw_method=bw_method)

        min_score = min(treatment_scores.min(), control_scores.min())
        max_score = max(treatment_scores.max(), control_scores.max())
        x_grid = np.linspace(min_score, max_score, grid_points)
        kde_treatment_values = kde_treatment(x_grid)
        kde_control_values = kde_control(x_grid)

        overlap_coefficient = np.trapz(np.minimum(kde_treatment_values, kde_control_values), x_grid)

        return overlap_coefficient

    def get_effects(
        self,
        min_binary_count: int = 100,
        adjustment: str | None = None,
        bootstrap: bool | None = None,
        compute_marginal_effects: str | bool | None = None,
    ) -> None:
        """ "
        Calculate effects (uplifts), given the data and experimental units.

        Parameters
        ----------
        min_binary_count : int, optional
            The minimum number of observations required for a binary covariate to be included in the analysis. Defaults to 100.
        adjustment : str, optional
            The type of adjustment to apply to estimation: 'IPW', 'IV'. Default is None.
        bootstrap : bool, optional
            Whether to use bootstrap inference. If None, uses the value set in __init__. Default is None.
        compute_marginal_effects : str | bool | None, optional
            Override the marginal effects setting from __init__. If None, uses the value set in __init__.
            - "overall": Average Marginal Effect (AME)
            - "mean": Marginal Effect at the Mean (MEM)
            - "median": Marginal effect at median
            - True: Same as "overall"
            - False: Return odds ratios / rate ratios

        Updates
        -------
        self._results: A Pandas DataFrame with effects.
        self._balance: A list of Pandas DataFrames with balance metrics per experiment.
        self._adjusted_balance: A list of Pandas DataFrames with adjusted balance metrics per experiment.

        Results
        -------
        The results are stored in self._results, which contains the following columns:
        - experiment: The experiment identifier.
        - outcome: The outcome variable.
        - treatment_group: The treatment/comparison group.
        - control_group: The control/reference group.
        - sample_ratio: The sample ratio of the treatment group to the control group.
        - adjustment: The type of adjustment applied.
        - inference_method: "asymptotic" or "bootstrap"
        - model_type: Type of statistical model used ("ols", "logistic", "poisson", "negative_binomial", "cox")
        - effect_type: Type of effect reported in absolute_effect column.
          - "mean_difference": OLS models (difference in means)
          - "probability_change": Logistic with marginal effects (percentage points)
          - "count_change": Poisson/NegBin with marginal effects (change in expected count)
          - "log_hazard_ratio": Cox models (log HR, the coefficient)
          - "log_odds": Logistic without marginal effects (log odds, the coefficient)
          - "log_rate_ratio": Poisson/NegBin without marginal effects (log IRR, the coefficient)
        - balance: The balance metric for the covariates (if applicable).
        - treatment_units: The number of units in the treatment group.
        - control_units: The number of units in the control group.
        - control_value: The mean value of the outcome variable in the control group.
        - treatment_value: The mean value of the outcome variable in the treatment group.
        - absolute_effect: The absolute effect of the treatment vs control.
        - standard_error: The standard error of the effect estimate.
        - pvalue: The p-value of the effect estimate.
        - stat_significance: The statistical significance of the effect.
        - abs_effect_lower: Lower bound of absolute effect CI.
        - abs_effect_upper: Upper bound of absolute effect CI.
        - relative_effect: The relative effect of the treatment.
        - rel_effect_lower: Lower bound of relative effect CI (if available).
        - rel_effect_upper: Upper bound of relative effect CI (if available).
        - srm_detected: Whether sample ratio mismatch was detected.
        - srm_pvalue: The p-value for the sample ratio mismatch test.

        Note:
        - Columns with all NaN values are automatically removed from results.
        - Model-specific metrics (odds_ratio, hazard_ratio, etc.) are not included in results.
        - Use effect_type to understand what absolute_effect and relative_effect represent.
        """  # noqa: E501

        if compute_marginal_effects is not None:
            valid_me_values = {False, "overall", "mean", "median", True}
            if compute_marginal_effects not in valid_me_values:
                log_and_raise_error(
                    self._logger,
                    f"compute_marginal_effects must be one of {valid_me_values}. Got: {compute_marginal_effects}",
                )
            if compute_marginal_effects is True:
                compute_marginal_effects = "overall"
            me_setting = compute_marginal_effects
        else:
            me_setting = self._compute_marginal_effects

        get_balance_method = self._estimator.get_balance_method

        temp_results = []
        output = {}
        weights_list = []

        if adjustment is None:
            adjustment = self._adjustment

        if bootstrap is not None:
            original_bootstrap = self._bootstrap
            self._bootstrap = bootstrap
        else:
            original_bootstrap = None

        self._balance = []
        self._adjusted_balance = []

        if self._experiment_identifier:
            grouped_data = self._data.groupby(self._experiment_identifier)
        else:
            grouped_data = [(None, self._data)]

        for experiment_tuple, temp_pd in grouped_data:
            if experiment_tuple is None:
                self._logger.info("Processing experiment")
            elif isinstance(experiment_tuple, tuple):
                if len(experiment_tuple) == 1:
                    self._logger.info(f"Processing experiment: {experiment_tuple[0]}")
                else:
                    self._logger.info(f"Processing experiment: {experiment_tuple}")
            else:
                self._logger.info(f"Processing experiment: {experiment_tuple}")

            if self._pvalue_adjustment:
                self._logger.info(f"P-value adjustment: {self._pvalue_adjustment}")
            final_covariates = []

            temp_pd, missing_summary = self.__handle_missing_data(temp_pd)
            if missing_summary["rows_dropped"] > 0:
                self._logger.warning(
                    f"Dropped {missing_summary['rows_dropped']} rows due to missing values "
                    f"({missing_summary['pct_dropped']:.1f}% of data)"
                )

            categorical_info = self.__get_categorical_covariates(data=temp_pd)
            dummy_map = {}
            ref_map = {}
            categorical_col_names = set()

            if categorical_info:
                categorical_col_names = set(categorical_info.keys())
                temp_pd, dummy_map, ref_map = self.__create_dummy_variables(
                    data=temp_pd,
                    categorical_info=categorical_info,
                    reference_categories=None,
                    include_reference=False,
                )
                self._logger.info(
                    f"Created {len(dummy_map)} dummy variables from {len(categorical_info)} categorical covariates"
                )
                for cov, ref in ref_map.items():
                    self._logger.info(f"  {cov}: reference category = {ref}")

            numeric_covariates = self.__get_numeric_covariates(data=temp_pd, exclude_categoricals=categorical_col_names)
            binary_covariates = self.__get_binary_covariates(data=temp_pd, exclude_categoricals=categorical_col_names)

            if dummy_map:
                binary_covariates.extend(list(dummy_map.keys()))

            treatvalues = set(temp_pd[self._treatment_col].unique())
            if len(treatvalues) < 2:
                self._logger.warning(f"Skipping {experiment_tuple} as there are not enough treatment groups!")
                continue

            comparison_pairs = self.__get_comparison_pairs(treatvalues, temp_pd)
            if not comparison_pairs:
                self._logger.warning(f"Skipping {experiment_tuple} as no valid comparison pairs were found!")
                continue

            for treatment_val, control_val in comparison_pairs:
                self._logger.info(f"Processing comparison: {treatment_val} vs {control_val}")

                comparison_data = temp_pd[temp_pd[self._treatment_col].isin([treatment_val, control_val])].copy()

                if comparison_data.empty:
                    self._logger.warning(f"No data for comparison {treatment_val} vs {control_val}. Skipping.")
                    continue

                # Check if we have both treatment and control groups before proceeding
                n_treatment_before = (comparison_data[self._treatment_col] == treatment_val).sum()
                n_control_before = (comparison_data[self._treatment_col] == control_val).sum()

                if n_treatment_before == 0:
                    self._logger.warning(
                        f"No treatment units ({treatment_val}) found for comparison {treatment_val} vs {control_val}. "
                        f"Cannot estimate treatment effect. Skipping."
                    )
                    continue

                if n_control_before == 0:
                    self._logger.warning(
                        f"No control units ({control_val}) found for comparison {treatment_val} vs {control_val}. "
                        f"Cannot estimate treatment effect. Skipping."
                    )
                    continue

                comparison_data[self._treatment_col] = (comparison_data[self._treatment_col] == treatment_val).astype(
                    int
                )

                sample_ratio, srm_detected, srm_pvalue = self.__check_sample_ratio_mismatch(comparison_data)

                comparison_data = self.impute_missing_values(
                    data=comparison_data.copy(),
                    num_covariates=numeric_covariates,
                    bin_covariates=binary_covariates,
                )

                comp_numeric_covariates = [c for c in numeric_covariates if comparison_data[c].std(ddof=0) != 0]
                comp_binary_covariates = [c for c in binary_covariates if comparison_data[c].sum() >= min_binary_count]
                comp_binary_covariates = [c for c in comp_binary_covariates if comparison_data[c].std(ddof=0) != 0]

                # exclude dummies that only appear in one treatment group
                removed_single_group = []
                if dummy_map:
                    treated_mask = comparison_data[self._treatment_col] == 1
                    control_mask = comparison_data[self._treatment_col] == 0
                    comp_binary_covariates_filtered = []
                    for c in comp_binary_covariates:
                        if c in dummy_map:
                            t_sum = comparison_data.loc[treated_mask, c].sum()
                            c_sum = comparison_data.loc[control_mask, c].sum()
                            if t_sum > 0 and c_sum > 0:
                                comp_binary_covariates_filtered.append(c)
                            else:
                                removed_single_group.append(c)
                        else:
                            comp_binary_covariates_filtered.append(c)
                    comp_binary_covariates = comp_binary_covariates_filtered

                removed_numeric_var = set(numeric_covariates) - set(comp_numeric_covariates)
                removed_binary_freq = [c for c in binary_covariates if comparison_data[c].sum() < min_binary_count]
                removed_binary_var = [
                    c for c in binary_covariates if c not in removed_binary_freq and comparison_data[c].std(ddof=0) == 0
                ]

                final_covariates = comp_numeric_covariates + comp_binary_covariates
                self._final_covariates = final_covariates

                if removed_numeric_var or removed_binary_freq or removed_binary_var or removed_single_group:
                    self._logger.warning(f"Removed covariates for comparison {treatment_val} vs {control_val}:")
                    if removed_numeric_var:
                        self._logger.warning(f"  - Zero variance (numeric): {sorted(removed_numeric_var)}")
                    if removed_binary_freq:
                        self._logger.warning(f"  - Low frequency (< {min_binary_count}): {sorted(removed_binary_freq)}")
                    if removed_binary_var:
                        self._logger.warning(f"  - Zero variance (binary): {sorted(removed_binary_var)}")
                    if removed_single_group:
                        self._logger.warning(f"  - Single group only (dummy): {sorted(removed_single_group)}")

                if len(final_covariates) == 0 & len(self._covariates if self._covariates is not None else []) > 0:
                    self._logger.warning(
                        f"No valid covariates for comparison {treatment_val} vs {control_val}, "
                        f"balance can't be assessed!"
                    )

                balance = pd.DataFrame()
                adjusted_balance = pd.DataFrame()

                if len(final_covariates) > 0 or categorical_info:
                    comparison_data["weights"] = 1

                    if categorical_info:
                        comparison_data_balance = comparison_data.copy()
                        comparison_data_balance, _, _ = self.__create_dummy_variables(
                            data=comparison_data_balance,
                            categorical_info=categorical_info,
                            reference_categories=ref_map,  # Use same references as models
                            include_reference=True,  # Include all for balance
                        )

                        balance_covariates = []

                        for cov in categorical_info.keys():
                            for cat in categorical_info[cov]:
                                cat_clean = _clean_category_name(cat)
                                dummy_col = f"{cov}_{cat_clean}"
                                if dummy_col in comparison_data_balance.columns:
                                    if comparison_data_balance[dummy_col].sum() >= min_binary_count:
                                        if comparison_data_balance[dummy_col].std(ddof=0) != 0:
                                            # Also check that dummy appears in both treatment and control groups
                                            treated_sum = comparison_data_balance[
                                                comparison_data_balance[self._treatment_col] == 1
                                            ][dummy_col].sum()
                                            control_sum = comparison_data_balance[
                                                comparison_data_balance[self._treatment_col] == 0
                                            ][dummy_col].sum()
                                            if treated_sum > 0 and control_sum > 0:
                                                balance_covariates.append(dummy_col)

                        for cov in final_covariates:
                            if cov not in balance_covariates:
                                balance_covariates.append(cov)

                        if balance_covariates:
                            comparison_data_balance = self.standardize_covariates(
                                comparison_data_balance, balance_covariates
                            )
                            balance = self.calculate_smd(data=comparison_data_balance, covariates=balance_covariates)

                        if len(final_covariates) > 0:
                            comparison_data = self.standardize_covariates(comparison_data, final_covariates)
                    else:
                        comparison_data = self.standardize_covariates(comparison_data, final_covariates)
                        balance = self.calculate_smd(data=comparison_data, covariates=final_covariates)

                    if not balance.empty:
                        balance["experiment"] = [experiment_tuple] * balance.shape[0]
                        balance = self.__transform_tuple_column(balance, "experiment", self._experiment_identifier)
                        self._balance.append(balance)
                        balance_mean = balance["balance_flag"].mean()
                        self._logger.info(f"Balance: {balance_mean:.2%}")

                if len(final_covariates) > 0 and adjustment in ("balance", "aipw"):
                    balance_func = get_balance_method(self._balance_method)
                    comparison_data = balance_func(
                        data=comparison_data, covariates=[f"z_{cov}" for cov in final_covariates]
                    )

                    adjusted_balance = self.calculate_smd(
                        data=comparison_data,
                        covariates=final_covariates,
                        weights_col=self._target_weights[self._target_effect],
                    )
                    adjusted_balance["experiment"] = [experiment_tuple] * adjusted_balance.shape[0]
                    adjusted_balance = self.__transform_tuple_column(
                        adjusted_balance, "experiment", self._experiment_identifier
                    )
                    self._adjusted_balance.append(adjusted_balance)
                    adj_balance_mean = adjusted_balance["balance_flag"].mean() if not adjusted_balance.empty else np.nan
                    self._logger.info(f"Adjusted balance: {adj_balance_mean:.2%}")
                    if self._assess_overlap:
                        if "propensity_score" in comparison_data.columns:
                            treatment_scores = comparison_data.loc[
                                comparison_data[self._treatment_col] == 1, "propensity_score"
                            ]
                            control_scores = comparison_data.loc[
                                comparison_data[self._treatment_col] == 0, "propensity_score"
                            ]
                            if not treatment_scores.empty and not control_scores.empty:
                                overlap = self.get_overlap_coefficient(treatment_scores.values, control_scores.values)
                                self._logger.info("::::: Overlap: %.2f", np.round(overlap, 2))
                            else:
                                self._logger.warning(
                                    "Could not calculate overlap due to missing scores for treatment or control."
                                )  # noqa: E501
                        else:
                            self._logger.warning("Propensity score column not found, skipping overlap assessment.")
                    if self._overlap_plot:
                        # Check if overlap plot can be generated (requires propensity scores from balance adjustment)
                        if adjustment not in ("balance", "aipw") or len(final_covariates) == 0:
                            self._logger.warning(
                                "Overlap plot requested but no balance adjustment method or covariates specified. "
                                "Overlap plots require propensity scores from balance adjustment. Skipping plot."
                            )
                        elif "propensity_score" in comparison_data.columns:
                            self._plot_common_support(
                                comparison_data,
                                treatment_col=self._treatment_col,
                                propensity_col="propensity_score",
                                experiment_id=experiment_tuple,
                            )
                        else:
                            self._logger.warning("Propensity score column not found, skipping overlap plot.")
                elif len(final_covariates) == 0 and adjustment in ("balance", "aipw"):
                    # No covariates to balance on: assign uniform weights so the pipeline
                    # proceeds without error and produces results equivalent to unadjusted analysis.
                    weight_col = self._target_weights[self._target_effect]
                    comparison_data[weight_col] = 1.0
                    self._logger.info(
                        "No covariates for balance adjustment — assigning uniform weights (1.0). "
                        "Results will be equivalent to unadjusted analysis."
                    )

                if len(final_covariates) > 0 and adjustment == "balance":
                    weight_col = self._target_weights[self._target_effect]
                    if weight_col in comparison_data.columns:
                        weight_columns = [*self._experiment_identifier, weight_col]
                        if self._unit_identifier and self._unit_identifier in comparison_data.columns:
                            weight_columns.insert(-1, self._unit_identifier)
                        weights_df = comparison_data[weight_columns].copy()
                        weights_df["treatment_group"] = treatment_val
                        weights_df["control_group"] = control_val
                        weights_list.append(weights_df)

                if len(final_covariates) > 0 and adjustment == "IV":
                    if self._instrument_col is None:
                        log_and_raise_error(self._logger, "Instrument column is required for IV estimation!")
                    iv_balance = self.calculate_smd(
                        data=comparison_data, treatment_col=self._instrument_col, covariates=final_covariates
                    )
                    iv_balance_mean = iv_balance["balance_flag"].mean() if not iv_balance.empty else np.nan
                    self._logger.info(f"IV Balance: {iv_balance_mean:.2%}")

                if adjustment == "aipw":
                    # AIPW always uses all covariates in the outcome model
                    relevant_covariates = set(self._final_covariates)
                else:
                    relevant_covariates = set(self._final_covariates) & set(self._regression_covariates)

                adjustment_labels = {"balance": "balance", "IV": "IV", "aipw": "aipw"}

                if adjustment == "aipw":
                    adjustment_label = "aipw"
                elif adjustment in adjustment_labels and len(relevant_covariates) > 0:
                    adjustment_label = adjustment_labels[adjustment] + "+regression"
                elif adjustment in adjustment_labels:
                    adjustment_label = adjustment_labels[adjustment]
                elif len(relevant_covariates) > 0:
                    adjustment_label = "regression"
                else:
                    adjustment_label = "no adjustment"

                for outcome in self._outcomes:
                    if not pd.api.types.is_numeric_dtype(comparison_data[outcome]):
                        self._logger.warning(
                            f"Outcome '{outcome}' is not numeric for comparison {treatment_val} vs {control_val}. "
                            f"Skipping."
                        )
                        continue
                    if comparison_data[outcome].var() == 0:
                        self._logger.warning(
                            f"Outcome '{outcome}' has zero variance for comparison {treatment_val} vs {control_val}. "
                            f"Skipping."
                        )
                        continue

                    model_types = self._normalized_outcome_models.get(outcome, ["ols"])

                    for model_type in model_types:
                        estimator_func = self.__get_estimator_function(model_type, adjustment, outcome)

                        try:
                            estimator_params = {
                                "data": comparison_data,
                                "outcome_variable": outcome,
                                "covariates": list(relevant_covariates),
                                "cluster_col": self._cluster_col,
                                "store_model": self._store_fitted_models,
                            }

                            if model_type in ["logistic", "poisson", "negative_binomial"]:
                                estimator_params["compute_marginal_effects"] = me_setting

                            if model_type == "cox":
                                if outcome in self._outcome_event_cols:
                                    estimator_params["event_col"] = self._outcome_event_cols[outcome]
                                elif self._event_col:
                                    estimator_params["event_col"] = self._event_col
                                else:
                                    log_and_raise_error(
                                        self._logger,
                                        f"Cox model for outcome '{outcome}' requires event column. "
                                        f"Use tuple notation: outcomes=[('{outcome}', 'event_col')] "
                                        f"or set event_col parameter.",
                                    )

                            if adjustment == "balance":
                                weight_col = self._target_weights[self._target_effect]
                                if weight_col not in comparison_data.columns:
                                    log_and_raise_error(
                                        self._logger,
                                        f"Weight column '{weight_col}' not found after balance calculation.",
                                    )
                                if len(final_covariates) > 0 and comparison_data[weight_col].var() == 0:
                                    self._logger.warning(
                                        f"Weight column '{weight_col}' has zero variance for comparison {treatment_val} vs {control_val}. Results might be unreliable."  # noqa: E501
                                    )
                                estimator_params["weight_column"] = weight_col

                            output = estimator_func(**estimator_params)

                            # Compute control group SD for retrodesign simulation
                            control_mask = comparison_data[self._treatment_col] == control_val
                            output["control_std"] = comparison_data.loc[control_mask, outcome].std()

                            if self._store_fitted_models and "fitted_model" in output:
                                if experiment_tuple not in self._fitted_models:
                                    self._fitted_models[experiment_tuple] = {}
                                if (treatment_val, control_val) not in self._fitted_models[experiment_tuple]:
                                    self._fitted_models[experiment_tuple][(treatment_val, control_val)] = {}
                                if outcome not in self._fitted_models[experiment_tuple][(treatment_val, control_val)]:
                                    self._fitted_models[experiment_tuple][(treatment_val, control_val)][outcome] = {}
                                fitted = output.pop("fitted_model")
                                comp_key = (treatment_val, control_val)
                                if len(model_types) > 1:
                                    self._fitted_models[experiment_tuple][comp_key][outcome][model_type] = fitted
                                else:
                                    self._fitted_models[experiment_tuple][comp_key][outcome] = fitted

                            if self._bootstrap:
                                skip_bootstrap = False
                                if model_type == "cox" and self._skip_bootstrap_for_survival:
                                    self._logger.info(
                                        f"Skipping bootstrap for survival outcome '{outcome}' "
                                        f"(skip_bootstrap_for_survival=True). Using Cox model robust standard errors."
                                    )
                                    skip_bootstrap = True

                                if not skip_bootstrap and model_type == "cox":
                                    event_col = (
                                        self._outcome_event_cols.get(outcome)
                                        if outcome in self._outcome_event_cols
                                        else self._event_col
                                    )
                                    if event_col and event_col in comparison_data.columns:
                                        event_rate = comparison_data[event_col].mean()
                                        n_events_treated = comparison_data[
                                            (comparison_data[self._treatment_col] == treatment_val)
                                            & (comparison_data[event_col] == 1)
                                        ].shape[0]
                                        n_events_control = comparison_data[
                                            (comparison_data[self._treatment_col] == control_val)
                                            & (comparison_data[event_col] == 1)
                                        ].shape[0]

                                        if event_rate < 0.3:
                                            self._logger.warning(
                                                f"Low event rate ({event_rate:.1%}) for outcome '{outcome}'. "
                                                f"Events: {n_events_treated} treated, {n_events_control} control. "
                                                f"Bootstrap may fail frequently with Cox models. "
                                                f"Consider: (1) skip_bootstrap_for_survival=True, "
                                                f"(2) reduce bootstrap_iterations, (3) remove covariates."
                                            )

                                        min_events = min(n_events_treated, n_events_control)
                                        if min_events < 10:
                                            self._logger.warning(
                                                f"Very few events in one group ({n_events_treated} treated, "
                                                f"{n_events_control} control) for outcome '{outcome}'. "
                                                f"Bootstrap will likely fail. "
                                                f"Recommend skip_bootstrap_for_survival=True."
                                            )

                                if not skip_bootstrap:
                                    bootstrap_abs_effects, bootstrap_rel_effects = self._bootstrap_single_effect(
                                        data=comparison_data,
                                        outcome=outcome,
                                        adjustment=adjustment,
                                        model_func=estimator_func,
                                        relevant_covariates=list(relevant_covariates),
                                        numeric_covariates=comp_numeric_covariates,
                                        binary_covariates=comp_binary_covariates,
                                        min_binary_count=min_binary_count,
                                        model_type=model_type,
                                        compute_marginal_effects=me_setting,
                                    )
                                    bootstrap_results = self._calculate_bootstrap_inference(
                                        bootstrap_abs_effects,
                                        bootstrap_rel_effects,
                                        output["absolute_effect"],
                                        output["relative_effect"],
                                    )
                                    output["standard_error"] = bootstrap_results["standard_error"]
                                    output["pvalue"] = bootstrap_results["pvalue"]
                                    output["abs_effect_lower"] = bootstrap_results["abs_effect_lower"]
                                    output["abs_effect_upper"] = bootstrap_results["abs_effect_upper"]
                                    output["rel_effect_lower"] = bootstrap_results["rel_effect_lower"]
                                    output["rel_effect_upper"] = bootstrap_results["rel_effect_upper"]
                                    output["stat_significance"] = 1 if output["pvalue"] < self._alpha else 0

                                    if self._pvalue_adjustment:
                                        if experiment_tuple not in self._bootstrap_distributions:
                                            self._bootstrap_distributions[experiment_tuple] = {}
                                        comp_key = (treatment_val, control_val)
                                        if comp_key not in self._bootstrap_distributions[experiment_tuple]:
                                            self._bootstrap_distributions[experiment_tuple][comp_key] = {}
                                        boot_key = f"{outcome}_{model_type}" if len(model_types) > 1 else outcome
                                        self._bootstrap_distributions[experiment_tuple][comp_key][boot_key] = {
                                            "abs_effects": bootstrap_abs_effects,
                                            "rel_effects": bootstrap_rel_effects,
                                        }
                            else:
                                output["abs_effect_lower"] = output.get("abs_effect_lower", np.nan)
                                output["abs_effect_upper"] = output.get("abs_effect_upper", np.nan)
                                output["rel_effect_lower"] = output.get("rel_effect_lower", np.nan)
                                output["rel_effect_upper"] = output.get("rel_effect_upper", np.nan)

                            output["inference_method"] = "bootstrap" if self._bootstrap else "asymptotic"
                            output["adjustment"] = adjustment_label
                            if adjustment in ("balance", "aipw"):
                                output["method"] = self._balance_method
                            if adjustment in ("balance", "aipw") and not adjusted_balance.empty:
                                output["balance"] = np.round(adjusted_balance["balance_flag"].mean(), 2)
                            elif not balance.empty:  # Use initial balance if no adjustment or balance failed
                                output["balance"] = np.round(balance["balance_flag"].mean(), 2)
                            else:
                                output["balance"] = np.nan  # No balance calculated

                            if adjustment == "balance":
                                weight_col = self._target_weights[self._target_effect]
                                if weight_col in comparison_data.columns:
                                    treat_mask = comparison_data[self._treatment_col] == 1
                                    control_mask = comparison_data[self._treatment_col] == 0
                                    treat_weights = comparison_data.loc[treat_mask, weight_col]
                                    control_weights = comparison_data.loc[control_mask, weight_col]
                                    ess_treat = self.compute_ess(weights=treat_weights)
                                    ess_control = self.compute_ess(weights=control_weights)
                                    n_treat = treat_mask.sum()
                                    n_control = control_mask.sum()
                                    output["ess_treatment"] = np.floor(ess_treat)
                                    output["ess_control"] = np.floor(ess_control)
                                    output["ess_treatment_reduction"] = (
                                        np.round(1 - (ess_treat / n_treat), 3) if n_treat > 0 else np.nan
                                    )  # noqa: E501
                                    output["ess_control_reduction"] = (
                                        np.round(1 - (ess_control / n_control), 3) if n_control > 0 else np.nan
                                    )  # noqa: E501
                                else:
                                    output["ess_treatment"] = np.nan
                                    output["ess_control"] = np.nan
                                    output["ess_treatment_reduction"] = np.nan
                                    output["ess_control_reduction"] = np.nan

                            output["treatment_group"] = treatment_val
                            output["control_group"] = control_val
                            output["experiment"] = experiment_tuple
                            output["sample_ratio"] = sample_ratio
                            output["srm_detected"] = srm_detected
                            output["srm_pvalue"] = srm_pvalue
                            temp_results.append(output)

                        except Exception as e:
                            self._logger.error(
                                f"Error processing outcome '{outcome}' model '{model_type}' for comparison {treatment_val} vs {control_val} with adjustment '{adjustment_label}': {e}"  # noqa: E501
                            )  # noqa: E501
                            error_output = {
                                "outcome": outcome,
                                "model_type": model_type,
                                "adjustment": adjustment_label,
                                "treatment_group": treatment_val,
                                "control_group": control_val,
                                "treatment_units": np.nan,
                                "control_units": np.nan,
                                "control_value": np.nan,
                                "treatment_value": np.nan,
                                "absolute_effect": np.nan,
                                "relative_effect": np.nan,
                                "stat_significance": np.nan,
                                "standard_error": np.nan,
                                "pvalue": np.nan,
                                "experiment": experiment_tuple,
                                "sample_ratio": sample_ratio,
                                "srm_detected": srm_detected,
                                "srm_pvalue": srm_pvalue,
                            }
                            temp_results.append(error_output)

        if not temp_results:
            self._logger.warning("No results were generated. Check input data and experiment configurations.")
            self._results = pd.DataFrame()
            return

        clean_temp_results = pd.DataFrame(temp_results)

        if weights_list:
            self._weights = pd.concat(weights_list, ignore_index=True)
        else:
            self._weights = None

        result_columns = [
            "experiment",
            "outcome",
            "treatment_group",
            "control_group",
            "sample_ratio",
            "adjustment",
            "inference_method",
            "model_type",
            "effect_type",
            "treatment_units",
            "control_units",
            "control_value",
            "treatment_value",
            "absolute_effect",
            "abs_effect_lower",
            "abs_effect_upper",
            "relative_effect",
            "rel_effect_lower",
            "rel_effect_upper",
            "standard_error",
            "pvalue",
            "stat_significance",
            "se_intercept",
            "cov_coef_intercept",
            "alpha_param",
            "control_std",
        ]

        balance_calculated = len(self._balance) > 0 or len(self._adjusted_balance) > 0
        if balance_calculated and "balance" in clean_temp_results.columns:
            index_to_insert = result_columns.index("adjustment") + 1
            result_columns.insert(index_to_insert, "method")
            index_to_insert = result_columns.index("method") + 1
            result_columns.insert(index_to_insert, "balance")
            result_columns.extend(
                [
                    "ess_treatment",
                    "ess_control",
                    "ess_treatment_reduction",
                    "ess_control_reduction",
                ]
            )
        if srm_detected is not None and "srm_detected" not in result_columns:
            result_columns.append("srm_detected")
        if srm_pvalue is not None and "srm_pvalue" not in result_columns:
            result_columns.append("srm_pvalue")

        final_result_columns = [col for col in result_columns if col in clean_temp_results.columns]
        clean_temp_results = clean_temp_results[final_result_columns]

        if self._balance:
            self._balance = pd.concat(self._balance, ignore_index=True)
        else:
            self._balance = pd.DataFrame()

        if self._adjusted_balance:
            self._adjusted_balance = pd.concat(self._adjusted_balance, ignore_index=True)
        else:
            self._adjusted_balance = pd.DataFrame()

        results_df = self.__transform_tuple_column(clean_temp_results, "experiment", self._experiment_identifier)

        if not results_df.empty and "absolute_effect" in results_df.columns and "standard_error" in results_df.columns:
            alpha = getattr(self, "_alpha", 0.05)
            z_critical = stats.norm.ppf(1 - alpha / 2)

            if "abs_effect_lower" not in results_df.columns:
                results_df["abs_effect_lower"] = np.nan
            if "abs_effect_upper" not in results_df.columns:
                results_df["abs_effect_upper"] = np.nan

            if "inference_method" in results_df.columns:
                async_mask = results_df["inference_method"] == "asymptotic"
                if async_mask.any():
                    results_df.loc[async_mask, "abs_effect_lower"] = (
                        results_df.loc[async_mask, "absolute_effect"]
                        - z_critical * results_df.loc[async_mask, "standard_error"]
                    )
                    results_df.loc[async_mask, "abs_effect_upper"] = (
                        results_df.loc[async_mask, "absolute_effect"]
                        + z_critical * results_df.loc[async_mask, "standard_error"]
                    )

        self._results = results_df

        if not results_df.empty:
            cols_to_check = ["rel_effect_lower", "rel_effect_upper", "se_intercept", "cov_coef_intercept"]
            cols_to_drop = []
            for col in cols_to_check:
                if col in results_df.columns and results_df[col].isna().all():
                    cols_to_drop.append(col)

            if cols_to_drop:
                self._logger.debug(f"Removing columns with all NaN values: {cols_to_drop}")
                self._results = self._results.drop(columns=cols_to_drop)

        if self._pvalue_adjustment:
            self._logger.info(f"Applying {self._pvalue_adjustment} p-value adjustment")
            self.adjust_pvalues(method=self._pvalue_adjustment)

        if original_bootstrap is not None:
            self._bootstrap = original_bootstrap

    def test_non_inferiority(
        self, absolute_margin: float | None = None, relative_margin: float | None = None, alpha: float = 0.05
    ) -> None:
        """
        Performs a non-inferiority test on the results.

        This test determines if the test group is not unacceptably worse than the control group.
        You must provide either an absolute or a relative margin.

        Parameters
        ----------
        absolute_margin : float, optional
            The absolute margin of non-inferiority.
        relative_margin : float, optional
            The relative margin of non-inferiority, calculated as a percentage of the control group's value.
        alpha : float, optional
            The significance level for the one-sided test, by default 0.05.

        Updates
        -------
        self._results : pd.DataFrame
            Adds non-inferiority test columns to the results DataFrame.
        """
        if self._results is None:
            log_and_raise_error(self._logger, "Must run get_effects() before testing for non-inferiority.")

        if not (0 < alpha < 1):
            log_and_raise_error(self._logger, "Alpha must be between 0 and 1 (exclusive).")

        if absolute_margin is not None and relative_margin is not None:
            log_and_raise_error(
                self._logger, "Please provide either an absolute_margin or a relative_margin, not both."
            )  # noqa: E501

        if absolute_margin is None and relative_margin is None:
            log_and_raise_error(self._logger, "Please provide either an absolute_margin or a relative_margin.")

        if absolute_margin is not None and absolute_margin <= 0:
            log_and_raise_error(self._logger, "absolute_margin must be a positive value.")

        if relative_margin is not None and relative_margin <= 0:
            log_and_raise_error(self._logger, "relative_margin must be a positive value.")

        results_df = self._results.copy()
        if relative_margin is not None:
            results_df["non_inferiority_margin"] = relative_margin * results_df["control_value"].abs()
        else:
            results_df["non_inferiority_margin"] = absolute_margin

        z_critical = stats.norm.ppf(1 - alpha)
        results_df["ci_lower_bound"] = results_df["absolute_effect"] - z_critical * results_df["standard_error"]

        results_df["is_non_inferior"] = results_df["ci_lower_bound"] > -results_df["non_inferiority_margin"]

        self._results = results_df

    def combine_effects(self, data: pd.DataFrame | None = None, grouping_cols: list[str] | None = None) -> pd.DataFrame:
        """
        Combine effects across experiments using fixed effects meta-analysis.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame containing the results. Defaults to self._results
        grouping_cols : list, optional
            The columns to group by. Defaults to experiment_identifer + ['outcome']
        effect : str, optional
            The method to use for combining results (fixed or random). Defaults to 'fixed'.

        Returns
        -------
        A Pandas DataFrame with combined results
        """

        if data is None:
            data = self._results

        if grouping_cols is None:
            self._logger.warning("No grouping columns specified, using only outcome!")
            grouping_cols = ["outcome"]
        else:
            grouping_cols = self.__ensure_list(grouping_cols)
            if "outcome" not in grouping_cols:
                grouping_cols.append("outcome")

        if any(data.groupby(grouping_cols).size() < 2):
            self._logger.warning("There are some combinations with only one experiment!")

        pooled_results = (
            data.groupby(grouping_cols)
            .apply(lambda df: pd.Series(self.__get_fixed_meta_analysis_estimate(df)), include_groups=False)
            .reset_index()
        )

        result_columns = grouping_cols + [
            "experiments",
            "control_units",
            "treatment_units",
            "absolute_effect",
            "relative_effect",
            "stat_significance",
            "standard_error",
            "pvalue",
        ]
        if "balance" in data.columns:
            index_to_insert = len(grouping_cols)
            result_columns.insert(index_to_insert + 1, "balance")
        pooled_results["stat_significance"] = pooled_results["stat_significance"].astype(int)

        self._logger.info("Combining effects using fixed-effects meta-analysis!")
        return pooled_results[result_columns]

    def __get_fixed_meta_analysis_estimate(self, data: pd.DataFrame) -> dict[str, int | float]:
        weights = 1 / (data["standard_error"] ** 2)
        absolute_estimate = np.sum(weights * data["absolute_effect"]) / np.sum(weights)
        pooled_standard_error = np.sqrt(1 / np.sum(weights))
        relative_estimate = np.sum(weights * data["relative_effect"]) / np.sum(weights)

        np.seterr(invalid="ignore")
        try:
            pvalue = stats.norm.sf(abs(absolute_estimate / pooled_standard_error)) * 2
        except FloatingPointError:
            pvalue = np.nan

        meta_results = {
            "experiments": int(data.shape[0]),
            "control_units": int(data["control_units"].sum()),
            "treatment_units": int(data["treatment_units"].sum()),
            "absolute_effect": absolute_estimate,
            "relative_effect": relative_estimate,
            "standard_error": pooled_standard_error,
            "pvalue": pvalue,
        }

        if "balance" in data.columns:
            meta_results["balance"] = data["balance"].mean()
        meta_results["stat_significance"] = 1 if meta_results["pvalue"] < self._alpha else 0
        return meta_results

    def aggregate_effects(
        self, data: pd.DataFrame | None = None, grouping_cols: list[str] | None = None
    ) -> pd.DataFrame:  # noqa: E501
        """
        Aggregate effects using a weighted average based on the size of the treatment group.

        Parameters
        ----------
        data : pd.DataFrame
        The DataFrame containing the results.
        grouping_cols : list, optional
        The columns to group by. Defaults to ['outcome']

        Returns
        -------
        A Pandas DataFrame with combined results
        """

        if data is None:
            data = self._results

        if grouping_cols is None:
            self._logger.warning("No grouping columns specified, using only outcome!")
            grouping_cols = ["outcome"]
        else:
            grouping_cols = self.__ensure_list(grouping_cols)
            if "outcome" not in grouping_cols:
                grouping_cols.append("outcome")

        aggregate_results = data.groupby(grouping_cols).apply(self.__compute_weighted_effect).reset_index()

        self._logger.info("Aggregating effects using weighted averages!")
        self._logger.info("For a better standard error estimation, use meta-analysis or `combine_effects`")

        result_columns = grouping_cols + ["experiments", "balance"]
        existing_columns = [col for col in result_columns if col in aggregate_results.columns]
        remaining_columns = [col for col in aggregate_results.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        return aggregate_results[final_columns]

    def __compute_weighted_effect(self, group: pd.DataFrame) -> pd.Series:
        group["gweight"] = group["treatment_units"].astype(int)
        absolute_effect = np.sum(group["absolute_effect"] * group["gweight"]) / np.sum(group["gweight"])
        relative_effect = np.sum(group["relative_effect"] * group["gweight"]) / np.sum(group["gweight"])
        variance = (group["standard_error"] ** 2) * group["gweight"]

        pooled_variance = np.sum(variance) / np.sum(group["gweight"])
        combined_se = np.sqrt(pooled_variance)
        z_score = absolute_effect / combined_se
        combined_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        output = pd.Series(
            {
                "experiments": int(group.shape[0]),
                "treatment_units": int(np.sum(group["gweight"])),
                "absolute_effect": absolute_effect,
                "relative_effect": relative_effect,
                "stat_significance": 1 if combined_p_value < self._alpha else 0,
                "standard_error": combined_se,
                "pvalue": combined_p_value,
            }
        )

        if "balance" in group.columns:
            combined_balance = np.sum(group["balance"] * group["gweight"]) / np.sum(group["gweight"])
            output["balance"] = combined_balance

        return output

    @property
    def imbalance(self) -> pd.DataFrame | None:
        """
        Returns the imbalance DataFrame. Checks adjusted balance first, then unadjusted.
        """
        if not self._adjusted_balance.empty:
            ab = self._adjusted_balance[self._adjusted_balance.balance_flag == 0]
            if not ab.empty:
                self._logger.info("Imbalance after adjustments!")
                return ab
        elif not self._balance.empty:
            b = self._balance[self._balance.balance_flag == 0]
            if not b.empty:
                self._logger.info("Imbalance without adjustments!")
                return b
        else:
            self._logger.warning("No balance information available to determine imbalance!")
            return None

    def __transform_tuple_column(self, df: pd.DataFrame, tuple_column: str, new_columns: list[str]) -> pd.DataFrame:
        """
        Transforms a column containing tuples or single values
        into separate columns named according to new_columns.
        Handles the case where a single identifier results in a single-element tuple.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the column to transform.
        tuple_column (str): The name of the column with tuples or single values.
        new_columns (list): A list of new column names. Should match self._experiment_identifier.

        Returns:
        pd.DataFrame: A new DataFrame with the elements as separate columns.
        """
        if df.empty or tuple_column not in df.columns:
            return df

        if not new_columns:
            self._logger.warning("No new column names provided for transformation. Skipping.")
            return df

        df = df.copy()

        if len(new_columns) == 1:
            new_col_name = new_columns[0]

            def extract_single_element(x):
                if isinstance(x, tuple) and len(x) == 1:
                    return x[0]  # Extract the first (and only) element, regardless of type
                return x  # Return as is if not a single-element tuple

            df.loc[:, tuple_column] = df[tuple_column].apply(extract_single_element)

            df = df.rename(columns={tuple_column: new_col_name})

            cols_order = [new_col_name] + [col for col in df.columns if col != new_col_name]
            df = df[cols_order]

        elif len(new_columns) > 1:

            def check_tuple(x):
                return isinstance(x, tuple) and len(x) == len(new_columns)

            if not df.empty and df[tuple_column].apply(check_tuple).all():
                columns_to_keep = [col for col in df.columns if col != tuple_column]
                try:
                    split_cols = pd.DataFrame(df[tuple_column].tolist(), index=df.index, columns=new_columns)
                    df_transformed = pd.concat([split_cols, df[columns_to_keep]], axis=1)
                    ordered_columns = new_columns + columns_to_keep
                    df = df_transformed[ordered_columns]
                except Exception as e:
                    self._logger.error(f"Failed to split tuple column '{tuple_column}' into {new_columns}. Error: {e}")
            elif not df.empty:
                self._logger.warning(
                    f"Column '{tuple_column}' does not contain consistent tuples of length {len(new_columns)}. Transformation skipped."  # noqa: E501
                )  # noqa: E501

        return df

    def __ensure_list(self, item: str | list[str] | None) -> list[str]:
        """Ensure the input is a list."""
        if item is None:
            return []
        return item if isinstance(item, list) else [item]

    @property
    def results(self) -> pd.DataFrame | None:
        """
        Returns the results DataFrame
        """
        if self._results is not None:
            internal_columns = ["se_intercept", "cov_coef_intercept", "alpha_param"]
            return self._results.drop(columns=[col for col in internal_columns if col in self._results.columns])
        else:
            self._logger.warning("Run the `get_effects` function first!")
            return None

    @property
    def balance(self) -> pd.DataFrame | None:
        """
        Returns the balance DataFrame
        """
        if isinstance(self._balance, pd.DataFrame) and not self._balance.empty:
            return self._balance
        else:
            if self._covariates:
                self._logger.warning(
                    "No balance information available! "
                    "If you have covariates, use check_balance() or run get_effects() first."
                )
            else:
                self._logger.warning("No balance information available!")
            return None

    @property
    def adjusted_balance(self) -> pd.DataFrame | None:
        """
        Returns the adjusted balance DataFrame
        """
        if isinstance(self._adjusted_balance, pd.DataFrame) and not self._adjusted_balance.empty:
            return self._adjusted_balance
        else:
            if self._covariates:
                self._logger.warning(
                    "No adjusted balance information available! "
                    "Run get_effects() with covariate adjustment to get adjusted balance."
                )
            else:
                self._logger.warning("No adjusted balance information available!")
            return None

    @property
    def weights(self) -> pd.DataFrame | None:
        """
        Returns the weights DataFrame from balance adjustment
        """
        if self._weights is not None:
            return self._weights
        else:
            self._logger.warning("No weights available! Weights are only saved when balance adjustment is used.")
            return None

    def check_balance(
        self,
        threshold: float = 0.1,
        min_binary_count: int = 5,
    ) -> pd.DataFrame | None:
        """
        Check covariate balance between treatment and control groups.

        This method can be called independently of get_effects() to assess balance
        on the loaded experiment data. It performs the same preprocessing as get_effects()
        including categorical variable handling, covariate filtering, and standardization.

        Parameters
        ----------
        threshold : float, optional
            SMD threshold for balance flag (default 0.1). Covariates with
            |SMD| < threshold are considered balanced.
        min_binary_count : int, optional
            Minimum count required for binary covariates (default 5).
            Binary covariates with fewer observations are excluded.

        Returns
        -------
        pd.DataFrame or None
            Balance metrics DataFrame with columns:
            - experiment columns (if experiment_identifier was specified)
            - covariate: str - covariate name
            - mean_treated: float - weighted mean for treatment group
            - mean_control: float - weighted mean for control group
            - smd: float - standardized mean difference
            - balance_flag: int - 1 if balanced (|SMD| < threshold), 0 if imbalanced

            Returns None if no covariates are specified.

        Examples
        --------
        >>> analyzer = ExperimentAnalyzer(
        ...     data=df,
        ...     outcomes=['conversion'],
        ...     treatment_col='treatment',
        ...     covariates=['age', 'income', 'region']
        ... )
        >>> balance_df = analyzer.check_balance(threshold=0.1)
        >>> print(f"Balanced: {balance_df['balance_flag'].mean():.2%}")
        >>> imbalanced = balance_df[balance_df['balance_flag'] == 0]
        >>> print(f"Imbalanced covariates: {imbalanced['covariate'].tolist()}")

        Notes
        -----
        - Categorical covariates are automatically detected and dummy-encoded
        - Covariates with zero variance or low frequency are filtered out
        - The method handles experiment identifiers (checks balance per experiment)
        - Can be called before or after get_effects()
        """
        from .utils import check_covariate_balance

        if self._covariates is None or len(self._covariates) == 0:
            self._logger.warning("No covariates specified, balance cannot be assessed!")
            return None

        # Identify categorical covariates
        categorical_info = self.__get_categorical_covariates(data=self._data)

        # Get experiment groups
        if self._experiment_identifier:
            experiment_groups = self._data.groupby(self._experiment_identifier)
        else:
            experiment_groups = [(None, self._data)]

        balance_results = []

        for experiment_tuple, temp_data in experiment_groups:
            if experiment_tuple is None:
                experiment_tuple = (1,)

            self._logger.info(f"Checking balance for experiment: {experiment_tuple}")

            # Get treatment values
            treatvalues = set(temp_data[self._treatment_col].unique())
            if len(treatvalues) < 2:
                self._logger.warning(f"Skipping {experiment_tuple}: not enough treatment groups!")
                continue

            # Get comparison pairs
            comparison_pairs = self.__get_comparison_pairs(treatvalues, temp_data)
            if not comparison_pairs:
                self._logger.warning(f"Skipping {experiment_tuple}: no valid comparison pairs found!")
                continue

            for treatment_val, control_val in comparison_pairs:
                self._logger.info(f"Checking balance: {treatment_val} vs {control_val}")

                # Get comparison data
                comparison_data = temp_data[temp_data[self._treatment_col].isin([treatment_val, control_val])].copy()

                if comparison_data.empty:
                    self._logger.warning(f"No data for comparison {treatment_val} vs {control_val}. Skipping.")
                    continue

                # Check if we have both treatment and control groups
                n_treatment = (comparison_data[self._treatment_col] == treatment_val).sum()
                n_control = (comparison_data[self._treatment_col] == control_val).sum()

                if n_treatment == 0 or n_control == 0:
                    self._logger.warning(
                        f"Missing treatment ({n_treatment}) or control ({n_control}) units for comparison "
                        f"{treatment_val} vs {control_val}. Cannot check balance. Skipping."
                    )
                    continue

                # Check balance using standalone function
                balance_df = check_covariate_balance(
                    data=comparison_data,
                    treatment_col=self._treatment_col,
                    covariates=self._covariates,
                    categorical_covariates=categorical_info,
                    min_binary_count=min_binary_count,
                    threshold=threshold,
                    treatment_value=treatment_val,
                    control_value=control_val,
                    categorical_max_unique=self._categorical_max_unique,
                    logger=self._logger,
                )

                if not balance_df.empty:
                    balance_df["experiment"] = [experiment_tuple] * len(balance_df)
                    balance_df = self.__transform_tuple_column(balance_df, "experiment", self._experiment_identifier)

                    # Log balance summary
                    balance_mean = balance_df["balance_flag"].mean()
                    self._logger.info(f"Balance: {balance_mean:.2%}")

                    balance_results.append(balance_df)

        if balance_results:
            final_balance = pd.concat(balance_results, ignore_index=True)
            return final_balance
        else:
            self._logger.warning("No balance results generated")
            return pd.DataFrame()

    def get_attribute(self, attribute: str) -> str | None:
        """
        Get an attribute of the class.

        Parameters:
        attribute (str): The name of the attribute to get.

        Returns:
        str: The value of the attribute.
        """

        private_attribute = f"_{attribute}"
        if hasattr(self, private_attribute):
            return getattr(self, private_attribute)
        elif hasattr(self, attribute):
            return getattr(self, attribute)
        else:
            log_and_raise_error(self._logger, f"Attribute {attribute} not found!")

    def get_fitted_models(
        self,
        experiment: tuple | None = None,
        comparison: tuple | None = None,
        outcome: str | None = None,
        model_type: str | None = None,
    ) -> dict | object | None:
        """
        Retrieve fitted model instances.

        This method allows access to the underlying statistical models fitted during
        analysis when store_fitted_models=True was set. Useful for post-hoc analysis,
        diagnostics, predictions, and model-specific tests.

        Parameters
        ----------
        experiment : tuple, optional
            Filter by experiment identifier tuple (e.g., ('US', 'mobile'))
        comparison : tuple, optional
            Filter by (treatment, control) comparison tuple (e.g., (1, 0))
        outcome : str, optional
            Filter by outcome name (e.g., 'conversion')
        model_type : str, optional
            Filter by model type when multiple models fitted for same outcome (e.g., 'ols', 'logistic')

        Returns
        -------
        dict or object or None
            - If no filters: returns full nested dict structure
            - If filters specified: returns filtered dict, single model object, or None
            Structure: {experiment_tuple: {(treatment, control): {outcome: model_object or {model_type: model_object}}}}

        Examples
        --------
        >>> # Get all models
        >>> all_models = analyzer.get_fitted_models()

        >>> # Get models for specific outcome
        >>> clicked_models = analyzer.get_fitted_models(outcome='clicked')

        >>> # Get specific model (single model for outcome)
        >>> model = analyzer.get_fitted_models(
        ...     experiment=(1,),
        ...     comparison=(1, 0),
        ...     outcome='conversion'
        ... )
        >>> model.summary()

        >>> # Get specific model when multiple models per outcome
        >>> ols_model = analyzer.get_fitted_models(
        ...     experiment=(1,),
        ...     comparison=(1, 0),
        ...     outcome='clicked',
        ...     model_type='ols'
        ... )
        >>> logit_model = analyzer.get_fitted_models(
        ...     experiment=(1,),
        ...     comparison=(1, 0),
        ...     outcome='clicked',
        ...     model_type='logistic'
        ... )
        """
        if not self._store_fitted_models:
            self._logger.warning(
                "No models stored. Set store_fitted_models=True when initializing "
                "ExperimentAnalyzer to store fitted models."
            )
            return None

        if not self._fitted_models:
            self._logger.warning("No models have been fitted yet. Run get_effects() first.")
            return None

        if experiment is None and comparison is None and outcome is None:
            return self._fitted_models

        result = self._fitted_models
        if experiment is not None:
            result = {exp: models for exp, models in result.items() if exp == experiment}
            if not result:
                return None
            if comparison is None and outcome is None:
                return result

        def _extract_model_by_type(model_obj, model_type_filter):
            """Extract specific model from potentially nested structure."""
            if model_type_filter is None:
                return model_obj
            if isinstance(model_obj, dict) and model_type_filter in model_obj:
                return model_obj[model_type_filter]
            return model_obj if model_type_filter is None else None

        if comparison is not None:
            if experiment is not None:
                if experiment in self._fitted_models:
                    comp_models = self._fitted_models[experiment].get(comparison, {})
                    if outcome is None:
                        return comp_models if comp_models else None
                    outcome_model = comp_models.get(outcome)
                    return _extract_model_by_type(outcome_model, model_type)
                return None
            else:
                result = {
                    exp: {comp: models for comp, models in exp_models.items() if comp == comparison}
                    for exp, exp_models in result.items()
                }
                result = {exp: comps for exp, comps in result.items() if comps}
                if not result:
                    return None
                if outcome is None:
                    return result

        if outcome is not None and experiment is None and comparison is None:
            result = {}
            for exp, exp_models in self._fitted_models.items():
                exp_result = {}
                for comp, comp_models in exp_models.items():
                    if outcome in comp_models:
                        outcome_model = _extract_model_by_type(comp_models[outcome], model_type)
                        if outcome_model is not None:
                            exp_result[comp] = outcome_model
                if exp_result:
                    result[exp] = exp_result
            return result if result else None

        if outcome is not None and experiment is not None and comparison is None:
            if experiment in self._fitted_models:
                exp_result = {}
                for comp, comp_models in self._fitted_models[experiment].items():
                    if outcome in comp_models:
                        outcome_model = _extract_model_by_type(comp_models[outcome], model_type)
                        if outcome_model is not None:
                            exp_result[comp] = outcome_model
                return exp_result if exp_result else None
            return None

        if outcome is not None and comparison is not None:
            pass

        return result if result else None

    def __get_comparison_pairs(self, treatvalues: set, data: pd.DataFrame) -> list[tuple]:
        """
        Generate comparison pairs for categorical treatment analysis.

        Parameters
        ----------
        treatvalues : set
            Set of unique treatment values in the data
        data : pd.DataFrame
            DataFrame containing the treatment column

        Returns
        -------
        list[tuple]
            List of (treatment_val, control_val) tuples to compare
        """
        import itertools

        if self._treatment_comparisons is not None:
            valid_pairs = []
            for treatment_val, control_val in self._treatment_comparisons:
                if treatment_val in treatvalues and control_val in treatvalues:
                    valid_pairs.append((treatment_val, control_val))
                else:
                    self._logger.warning(
                        f"Skipping comparison ({treatment_val}, {control_val}) - one or both groups not found in data"
                    )
            return valid_pairs

        if treatvalues == {0, 1}:
            return [(1, 0)]

        sorted_values = sorted(list(treatvalues))
        pairs = list(itertools.combinations(sorted_values, 2))
        return [(b, a) for a, b in pairs]

    def __check_sample_ratio_mismatch(self, df: pd.DataFrame):
        """
        Performs a Sample Ratio Mismatch (SRM) test to check if the observed
        treatment ratio significantly differs from the expected ratio.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the treatment assignment and expected sample ratio.
        """

        n_total = len(df)
        observed_count = df[self._treatment_col].sum()
        if n_total > 0:
            sample_ratio = observed_count / n_total
        else:
            sample_ratio = None

        if self._exp_sample_ratio_col is not None:
            unique_expected_ratio = df[self._exp_sample_ratio_col].unique()

            if len(unique_expected_ratio) > 1:
                log_and_raise_error(
                    self._logger, "Multiple unique values by experiment found in expected ratio column!"
                )  # noqa: E501
            if not (0 < unique_expected_ratio < 1):
                log_and_raise_error(self._logger, "Expected ratio is not between 0 and 1!")

            try:
                srm_detected = False
                srm_pvalue = None

                z_stat, p_value = proportions_ztest(
                    count=observed_count, nobs=n_total, value=unique_expected_ratio[0], alternative="two-sided"
                )
                srm_pvalue = p_value

                if p_value < self._alpha:
                    srm_detected = True
                    self._logger.info(
                        f"Significant mismatch detected (p < {self._alpha:.3f}). Observed ratio differs statistically from expected ratio."  # noqa: E501
                    )  # noqa: E501

                return sample_ratio, srm_detected, srm_pvalue

            except Exception as e:
                self._logger.error(f"SRM error during proportions_ztest: {e}")
        else:
            return sample_ratio, None, None

    def compute_ess(self, weights: np.ndarray) -> float:
        """
        Compute effective sample size (ESS) given a vector of weights.
        ESS = (sum(w))^2 / sum(w^2)
        """
        weights = np.asarray(weights)
        if np.all(weights == 0):
            return 0.0
        return (weights.sum()) ** 2 / (np.sum(weights**2) + 1e-12)

    def _plot_common_support(
        self, df, propensity_col, treatment_col, bw_method=None, grid_size=100, figsize=(10, 6), experiment_id=None
    ):
        """
        Create a mirror density plot for common support showing the distribution
        of propensity scores for treatment (up) and control (down).

        Parameters:
        - df: pandas DataFrame containing the data.
        - propensity_col: string, name of the column with propensity scores.
        - treatment_col: string, name of the column with treatment indicator (binary).
        - bw_method: bandwidth method for gaussian_kde (default None).
        - grid_size: number of grid points for density estimation.
        - figsize: tuple, figure size.
        - experiment_id: optional, identifier for the experiment (used in title).
        """

        treatment_scores = df.loc[df[treatment_col] == 1, propensity_col].dropna()
        control_scores = df.loc[df[treatment_col] == 0, propensity_col].dropna()

        all_scores = df[propensity_col].dropna()
        x_min, x_max = all_scores.min(), all_scores.max()
        x_grid = np.linspace(x_min, x_max, grid_size)

        kde_treat = gaussian_kde(treatment_scores, bw_method=bw_method)
        kde_control = gaussian_kde(control_scores, bw_method=bw_method)
        density_treat = kde_treat(x_grid)
        density_control = kde_control(x_grid)

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(x_grid, density_treat, alpha=0.3, color="blue", label="Treatment")
        ax.plot(x_grid, density_treat, color="blue")

        ax.fill_between(x_grid, -density_control, alpha=0.3, color="red", label="Control")
        ax.plot(x_grid, -density_control, color="red")

        ax.axhline(0, color="black", linewidth=0.8)

        ax.set_xlabel("Propensity Score")
        ax.set_ylabel("Density")
        ax.set_title(f"Common Support {f'for Experiment {experiment_id}' if experiment_id else ''}")
        ax.legend(loc="best")

        from matplotlib.ticker import FuncFormatter

        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{abs(x):.2f}"))

        plt.tight_layout()
        return fig

    def _compute_fieller_ci_adjusted(
        self,
        coefficient: float,
        intercept: float,
        se_coef: float,
        se_intercept: float,
        cov: float,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Compute Fieller confidence interval with custom alpha.

        This is a simplified version used for MCP-adjusted relative CIs.
        Uses the same Fieller's theorem as in estimators.py but with adjusted alpha.

        Parameters
        ----------
        coefficient : float
            Treatment effect (numerator)
        intercept : float
            Control mean (denominator)
        se_coef : float
            Standard error of coefficient
        se_intercept : float
            Standard error of intercept
        cov : float
            Covariance between coefficient and intercept
        alpha : float
            Adjusted significance level

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) or (nan, nan) if cannot compute
        """
        from scipy import stats

        if intercept == 0 or abs(intercept) < 1e-10:
            return (np.nan, np.nan)

        z_crit = stats.norm.ppf(1 - alpha / 2)
        g_sq = z_crit**2

        a = intercept**2 - g_sq * se_intercept**2
        b = -(2 * intercept * coefficient - 2 * g_sq * cov)
        c = coefficient**2 - g_sq * se_coef**2

        if abs(a) < 1e-10:
            return (np.nan, np.nan)

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return (np.nan, np.nan)

        sqrt_disc = np.sqrt(discriminant)
        root1 = (-b - sqrt_disc) / (2 * a)
        root2 = (-b + sqrt_disc) / (2 * a)

        if a > 0:
            ci_lower = min(root1, root2)
            ci_upper = max(root1, root2)
        else:
            return (np.nan, np.nan)

        return (ci_lower, ci_upper)

    def adjust_pvalues(
        self,
        method: str = "bonferroni",
        groupby_cols: list[str] | None = None,
        outcomes: list[str] | None = None,
        experiments: list | None = None,
        alpha: float | None = 0.05,
    ):
        """
        Adjust p-values for multiple comparisons and compute adjusted confidence intervals.

        This method applies multiple comparison corrections to p-values and recalculates
        confidence intervals using method-appropriate alpha adjustments. Results are added
        to the existing results dataframe with '_mcp' suffix (multiple comparison procedure).

        Parameters
        ----------
        method : str, optional
            Multiple comparison adjustment method, by default "bonferroni"
            Options: 'bonferroni', 'holm', 'fdr_bh', 'sidak', 'hommel', 'hochberg', 'by'
        groupby_cols : list[str], optional
            Columns to group by when applying adjustment (default: experiment_identifier)
        outcomes : list[str], optional
            Filter to specific outcomes for adjustment
        experiments : list, optional
            Filter to specific experiments for adjustment
        alpha : float, optional
            Family-wise error rate or false discovery rate threshold, by default 0.05

        Notes
        -----
        Column naming:
        - '_mcp' suffix indicates multiple comparison procedure adjustments
        - This distinguishes from 'adjustment' parameter (balance/IV covariate adjustment)

        Confidence interval adjustments by method:
        - **Bonferroni**: Uses α_CI = α/k for simultaneous coverage
        - **Sidak**: Uses α_CI = 1-(1-α)^(1/k) for independent tests
        - **Sequential methods** (Holm/Hochberg/Hommel): Use conservative Bonferroni
          approach (α/k) since no single per-comparison alpha exists
        - **FDR methods** (BH/BY): Use conservative Bonferroni approach. Note that FDR
          controls expected proportion of false discoveries, not family-wise error rate,
          so simultaneous CI coverage is not directly comparable

        Relative effect CIs:
        - Relative effect CIs are computed using Fieller's method with adjusted alpha
        - Requires stored covariance information (se_intercept, cov_coef_intercept)
        - If covariance info unavailable (e.g., bootstrap results), CIs set to NaN

        Added columns:
        - pvalue_mcp: Adjusted p-values
        - stat_significance_mcp: Significance indicator using adjusted p-values
        - mcp_method: Method used for adjustment
        - abs_effect_lower_mcp: Adjusted lower bound for absolute effect
        - abs_effect_upper_mcp: Adjusted upper bound for absolute effect
        - rel_effect_lower_mcp: Adjusted lower bound for relative effect (Fieller method)
        - rel_effect_upper_mcp: Adjusted upper bound for relative effect (Fieller method)

        Examples
        --------
        >>> analyzer.get_effects()
        >>> analyzer.adjust_pvalues(method='holm')
        >>> results = analyzer.results
        >>> print(results[['pvalue', 'pvalue_mcp', 'stat_significance_mcp']])
        """
        if self._results is None:
            log_and_raise_error(self._logger, "Run get_effects() first.")

        df = self._results.copy()
        group_cols = groupby_cols or self._experiment_identifier or ["experiment"]
        if not all(col in df.columns for col in group_cols):
            log_and_raise_error(self._logger, f"Grouping columns {group_cols} not found in results.")

        mask = pd.Series([True] * len(df), index=df.index)
        if outcomes is not None:
            mask &= df["outcome"].isin(outcomes)
        if experiments is not None:
            for col, vals in zip(group_cols, zip(*experiments, strict=False), strict=False):
                mask &= df[col].isin(vals)
        df_adj = df[mask].copy()
        df_rest = df[~mask].copy()

        def calculate_adjustments(group):
            pvals = group["pvalue"].values
            m = method.lower()
            if m == "bonferroni":
                pvals_adj = np.minimum(pvals * len(pvals), 1.0)
            elif m in {"holm", "fdr_bh", "sidak", "hommel", "hochberg", "by"}:
                from statsmodels.stats.multitest import multipletests

                pvals_adj = multipletests(pvals, method=m)[1]
            else:
                raise ValueError(f"Unknown adjustment method: {method}")

            thres = alpha if alpha is not None else self._alpha

            from scipy import stats

            n_comparisons = len(pvals)

            if m == "bonferroni":
                alpha_ci = thres / n_comparisons
                ci_note = f"Using Bonferroni α/k = {alpha_ci:.6f} for {n_comparisons} comparisons"
            elif m == "sidak":
                alpha_ci = 1 - (1 - thres) ** (1 / n_comparisons)
                ci_note = f"Using Sidak 1-(1-α)^(1/k) = {alpha_ci:.6f} for {n_comparisons} comparisons"
            elif m in {"holm", "hochberg", "hommel"}:
                alpha_ci = thres / n_comparisons
                ci_note = (
                    f"Using conservative Bonferroni α/k = {alpha_ci:.6f} for CIs "
                    f"(sequential method {m} has no single per-comparison alpha)"
                )
                self._logger.warning(ci_note)
            elif m == "fdr_bh" or m == "by":
                alpha_ci = thres / n_comparisons
                ci_note = (
                    f"Using conservative Bonferroni α/k = {alpha_ci:.6f} for CIs "
                    f"(FDR method {m} controls false discovery rate, not family-wise error)"
                )
                self._logger.warning(ci_note)
            else:
                alpha_ci = thres / n_comparisons
                ci_note = f"Using default Bonferroni α/k = {alpha_ci:.6f}"

            z_crit_adj = stats.norm.ppf(1 - alpha_ci / 2)

            result_dict = {
                "pvalue_mcp": pvals_adj,
                "stat_significance_mcp": (pvals_adj < thres).astype(int),
                "mcp_method": method,
            }

            abs_effect = group["absolute_effect"].values
            se = group["standard_error"].values
            result_dict["abs_effect_lower_mcp"] = abs_effect - z_crit_adj * se
            result_dict["abs_effect_upper_mcp"] = abs_effect + z_crit_adj * se

            if "se_intercept" in group.columns and "cov_coef_intercept" in group.columns:
                rel_lower_adj = []
                rel_upper_adj = []

                for idx in range(len(group)):
                    coef = abs_effect[idx]
                    intercept = group["control_value"].values[idx]
                    se_coef = se[idx]
                    se_int = group["se_intercept"].values[idx]
                    cov = group["cov_coef_intercept"].values[idx]

                    rel_ci_lower, rel_ci_upper = self._compute_fieller_ci_adjusted(
                        coef, intercept, se_coef, se_int, cov, alpha_ci
                    )
                    rel_lower_adj.append(rel_ci_lower)
                    rel_upper_adj.append(rel_ci_upper)

                result_dict["rel_effect_lower_mcp"] = rel_lower_adj
                result_dict["rel_effect_upper_mcp"] = rel_upper_adj

            elif self._bootstrap_distributions:
                rel_lower_adj = []
                rel_upper_adj = []
                exp_key = group.name

                for idx in range(len(group)):
                    row = group.iloc[idx]
                    comp_key = (row["treatment_group"], row["control_group"])
                    outcome_key = row["outcome"]

                    if (
                        exp_key in self._bootstrap_distributions
                        and comp_key in self._bootstrap_distributions[exp_key]
                        and outcome_key in self._bootstrap_distributions[exp_key][comp_key]
                    ):
                        rel_dist = self._bootstrap_distributions[exp_key][comp_key][outcome_key]["rel_effects"]
                        rel_dist_clean = rel_dist[~np.isnan(rel_dist)]

                        if len(rel_dist_clean) > 0:
                            if self._bootstrap_ci_method == "percentile":
                                rel_lower = np.percentile(rel_dist_clean, alpha_ci / 2 * 100)
                                rel_upper = np.percentile(rel_dist_clean, (1 - alpha_ci / 2) * 100)
                            elif self._bootstrap_ci_method == "basic":
                                observed_rel = row["relative_effect"]
                                rel_lower = 2 * observed_rel - np.percentile(rel_dist_clean, (1 - alpha_ci / 2) * 100)
                                rel_upper = 2 * observed_rel - np.percentile(rel_dist_clean, alpha_ci / 2 * 100)
                            else:
                                rel_lower = np.percentile(rel_dist_clean, alpha_ci / 2 * 100)
                                rel_upper = np.percentile(rel_dist_clean, (1 - alpha_ci / 2) * 100)
                        else:
                            rel_lower = np.nan
                            rel_upper = np.nan
                    else:
                        rel_lower = np.nan
                        rel_upper = np.nan

                    rel_lower_adj.append(rel_lower)
                    rel_upper_adj.append(rel_upper)

                result_dict["rel_effect_lower_mcp"] = rel_lower_adj
                result_dict["rel_effect_upper_mcp"] = rel_upper_adj

            else:
                result_dict["rel_effect_lower_mcp"] = np.nan
                result_dict["rel_effect_upper_mcp"] = np.nan

            return pd.DataFrame(result_dict, index=group.index)

        if "se_intercept" in df_adj.columns and "cov_coef_intercept" in df_adj.columns:
            self._logger.info(
                "Computing adjusted relative effect CIs using Fieller's method with adjusted alpha. "
                "This properly accounts for uncertainty in both numerator and denominator."
            )
        elif self._bootstrap_distributions:
            self._logger.info(
                "Computing adjusted relative effect CIs from bootstrap distributions with adjusted alpha."
            )
        else:
            self._logger.warning(
                "Covariance information (se_intercept, cov_coef_intercept) not found in results, "
                "and no bootstrap distributions available. "
                "Relative effect CIs with MCP adjustment will be set to NaN. "
                "To obtain adjusted relative CIs, re-run get_effects() with bootstrap=True or bootstrap=False."
            )

        adjustments = df_adj.groupby(group_cols, group_keys=False).apply(calculate_adjustments, include_groups=False)

        cols_to_add = [
            "pvalue_mcp",
            "stat_significance_mcp",
            "mcp_method",
            "abs_effect_lower_mcp",
            "abs_effect_upper_mcp",
            "rel_effect_lower_mcp",
            "rel_effect_upper_mcp",
        ]

        df_adj = df_adj.drop(columns=cols_to_add, errors="ignore")
        df_adj = df_adj.join(adjustments)
        df_final = pd.concat([df_adj, df_rest]).sort_index()

        mcp_cols_to_check = ["rel_effect_lower_mcp", "rel_effect_upper_mcp"]
        mcp_cols_to_drop = []
        for col in mcp_cols_to_check:
            if col in df_final.columns and df_final[col].isna().all():
                mcp_cols_to_drop.append(col)

        if mcp_cols_to_drop:
            self._logger.debug(f"Removing MCP columns with all NaN values: {mcp_cols_to_drop}")
            df_final = df_final.drop(columns=mcp_cols_to_drop)

        self._results = df_final
