"""
ExperimentAnalyzer class to analyze and design experiments
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.proportion import proportions_ztest

from .estimators import Estimators
from .utils import get_logger, log_and_raise_error


class ExperimentAnalyzer:
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
    pvalue_adjustment : str, optional
        P-value adjustment method to apply automatically after get_effects ('bonferroni', 'holm', 'fdr_bh', 'sidak', 'hommel', 'hochberg', 'by', or None for no adjustment), by default 'bonferroni'
    categorical_max_unique : int, optional
        Maximum number of unique values for integer/numeric columns to be treated as categorical.
        Integer columns with 3 to categorical_max_unique unique values will be converted to dummy variables.
        Binary integers (2 values) are excluded and kept as-is.
        Set to 2 to disable categorical treatment for integers (nothing will match 3 <= n <= 2).
        Set to 5, 10, or higher for more inclusive behavior.
        By default 2 (effectively disables categorical treatment for integer columns).
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
        treatment_comparisons: list[tuple] | None = None,
        pvalue_adjustment: str | None = "bonferroni",
        categorical_max_unique: int = 2,
    ) -> None:
        self._logger = get_logger("Experiment Analyzer")
        self._data = data.copy()
        self._outcomes = self.__ensure_list(outcomes)
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
        self._pvalue_adjustment = pvalue_adjustment
        self._treatment_comparisons = treatment_comparisons
        self._categorical_max_unique = categorical_max_unique
        self.__check_input()
        self._alpha = alpha
        self._results = None
        self._balance = []
        self._adjusted_balance = []
        self._weights = None
        self._final_covariates = []
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
        # dataframe is empty
        if self._data.empty:
            log_and_raise_error(self._logger, "Dataframe is empty!")

        # impute covariates from regression covariates
        if (len(self._covariates) == 0) & (len(self._regression_covariates) > 0):
            self._covariates = self._regression_covariates

        # check for categorical columns that will be automatically converted to dummies
        string_cols = [c for c in self._covariates if pd.api.types.is_string_dtype(self._data[c])]
        if string_cols:
            self._logger.info(f"Detected categorical covariates (will create dummies): {string_cols}")

        # regression covariates has to be a subset of covariates
        if len(self._regression_covariates) > 0:
            if not set(self._regression_covariates).issubset(set(self._covariates)):
                log_and_raise_error(self._logger, "Regression covariates should be a subset of covariates")

        # create an experiment id if there is not one
        if len(self._experiment_identifier) == 0:
            self._data["experiment_id"] = 1
            self._experiment_identifier = ["experiment_id"]
            self._logger.warning("No experiment identifier, assuming data is from a single experiment!")

        required_columns = (
            self._experiment_identifier
            + [self._treatment_col]
            + self._outcomes
            + self._covariates
            + ([self._instrument_col] if self._instrument_col is not None else [])
            + ([self._exp_sample_ratio_col] if self._exp_sample_ratio_col is not None else [])
            + ([self._unit_identifier] if self._unit_identifier is not None else [])
        )

        missing_columns = set(required_columns) - set(self._data.columns)

        if missing_columns:
            log_and_raise_error(
                self._logger, f"The following required columns are missing from the dataframe: {missing_columns}"
            )  # noqa: E501
        if len(self._covariates) == 0:
            self._logger.warning("No covariates specified, balance can't be assessed!")

        self._data = self._data[required_columns]

    def __get_binary_covariates(self, data: pd.DataFrame, exclude_categoricals: set[str] = None) -> list[str]:
        """Get binary covariates, optionally excluding categorical columns that were converted to dummies."""
        binary_covariates = []
        if exclude_categoricals is None:
            exclude_categoricals = set()
        if self._covariates is not None:
            for c in self._covariates:
                if c in exclude_categoricals:
                    continue  # Skip categorical columns that are now dummies
                if data[c].nunique() == 2 and data[c].max() == 1:
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
        - integer columns with 3-10 unique values (excludes binary 0/1 columns)

        Returns dict: {covariate_name: [list_of_categories]}
        """
        categorical_info = {}
        if self._covariates is not None:
            for c in self._covariates:
                is_object = pd.api.types.is_object_dtype(data[c]) or isinstance(data[c].dtype, pd.CategoricalDtype)

                # treat integers as categorical if they have 3 to categorical_max_unique values
                # exclude binary (2 values) since those don't need dummy encoding
                is_low_cardinality_int = (
                    pd.api.types.is_integer_dtype(data[c]) and 3 <= data[c].nunique() <= self._categorical_max_unique
                )

                if is_object or is_low_cardinality_int:
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
        import re

        def _clean_category_name(cat):
            """Convert category to lowercase and replace spaces/special chars with underscores"""
            cat_str = str(cat).lower()
            cat_str = re.sub(r"[^\w]+", "_", cat_str)
            cat_str = cat_str.strip("_")
            return cat_str

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
                        pd.api.types.is_integer_dtype(data[cov])
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
                    # Create explicit 'Missing' category
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
        self, min_binary_count: int = 100, adjustment: str | None = None, bootstrap: bool | None = None
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
        - balance: The balance metric for the covariates.
        - treatment_units: The number of units in the treatment group.
        - control_units: The number of units in the control group.
        - control_value: The mean value of the outcome variable in the control group.
        - treatment_value: The mean value of the outcome variable in the treatment group.
        - absolute_effect: The absolute effect of the treatment vs control.
        - relative_effect: The relative effect of the treatment.
        - stat_significance: The statistical significance of the effect.
        - standard_error: The standard error of the effect estimate.
        - pvalue: The p-value of the effect estimate.
        - srm_detected: Whether sample ratio mismatch was detected.
        - srm_pvalue: The p-value for the sample ratio mismatch test.
        """  # noqa: E501

        model = {
            None: self._estimator.linear_regression,
            "balance": self._estimator.weighted_least_squares,
            "IV": self._estimator.iv_regression,
        }

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
            self._logger.info("Processing: %s", experiment_tuple)
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

                removed_numeric_var = set(numeric_covariates) - set(comp_numeric_covariates)
                removed_binary_freq = [c for c in binary_covariates if comparison_data[c].sum() < min_binary_count]
                removed_binary_var = [
                    c for c in binary_covariates if c not in removed_binary_freq and comparison_data[c].std(ddof=0) == 0
                ]

                final_covariates = comp_numeric_covariates + comp_binary_covariates
                self._final_covariates = final_covariates

                if removed_numeric_var or removed_binary_freq or removed_binary_var:
                    self._logger.warning(f"Removed covariates for comparison {treatment_val} vs {control_val}:")
                    if removed_numeric_var:
                        self._logger.warning(f"  - Zero variance (numeric): {sorted(removed_numeric_var)}")
                    if removed_binary_freq:
                        self._logger.warning(f"  - Low frequency (< {min_binary_count}): {sorted(removed_binary_freq)}")
                    if removed_binary_var:
                        self._logger.warning(f"  - Zero variance (binary): {sorted(removed_binary_var)}")

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
                                dummy_col = f"{cov}_{cat}"
                                # Check if this dummy passes the same filtering criteria
                                if dummy_col in comparison_data_balance.columns:
                                    if comparison_data_balance[dummy_col].sum() >= min_binary_count:
                                        if comparison_data_balance[dummy_col].std(ddof=0) != 0:
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
                        self._logger.info("::::: Balance: %.2f", np.round(balance_mean, 2))

                if len(final_covariates) > 0 and adjustment == "balance":
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
                    self._logger.info("::::: Adjusted balance: %.2f", np.round(adj_balance_mean, 2))
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
                        if "propensity_score" in comparison_data.columns:
                            self._plot_common_support(
                                comparison_data,
                                treatment_col=self._treatment_col,
                                propensity_col="propensity_score",
                                experiment_id=experiment_tuple,
                            )
                        else:
                            self._logger.warning("Propensity score column not found, skipping overlap plot.")

                # save weights for this experiment when balance adjustment is used
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
                    self._logger.info("::::: IV Balance: %.2f", np.round(iv_balance_mean, 2))

                # create adjustment label
                relevant_covariates = set(self._final_covariates) & set(self._regression_covariates)

                adjustment_labels = {"balance": "balance", "IV": "IV"}

                if adjustment in adjustment_labels and len(relevant_covariates) > 0:
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

                    try:
                        if adjustment == "balance":
                            weight_col = self._target_weights[self._target_effect]
                            if weight_col not in comparison_data.columns:
                                log_and_raise_error(
                                    self._logger,
                                    f"Weight column '{weight_col}' not found after balance calculation.",
                                )
                            if comparison_data[weight_col].var() == 0:
                                self._logger.warning(
                                    f"Weight column '{weight_col}' has zero variance for comparison {treatment_val} vs {control_val}. Results might be unreliable."  # noqa: E501
                                )
                            output = model[adjustment](
                                data=comparison_data,
                                outcome_variable=outcome,
                                covariates=list(relevant_covariates),
                                weight_column=weight_col,
                            )
                        else:
                            output = model[adjustment](
                                data=comparison_data, outcome_variable=outcome, covariates=list(relevant_covariates)
                            )

                        if self._bootstrap:
                            self._logger.info(
                                f"Running bootstrap for outcome '{outcome}' with "
                                f"{self._bootstrap_iterations} iterations..."
                            )
                            bootstrap_abs_effects, bootstrap_rel_effects = self.__bootstrap_single_effect(
                                data=comparison_data,
                                outcome=outcome,
                                adjustment=adjustment,
                                model_func=model[adjustment],
                                relevant_covariates=list(relevant_covariates),
                                numeric_covariates=comp_numeric_covariates,
                                binary_covariates=comp_binary_covariates,
                                min_binary_count=min_binary_count,
                            )
                            bootstrap_results = self.__calculate_bootstrap_inference(
                                bootstrap_abs_effects,
                                bootstrap_rel_effects,
                                output["absolute_effect"],
                                output["relative_effect"],
                            )
                            # Replace asymptotic inference with bootstrap inference
                            output["standard_error"] = bootstrap_results["standard_error"]
                            output["pvalue"] = bootstrap_results["pvalue"]
                            output["abs_effect_lower"] = bootstrap_results["abs_effect_lower"]
                            output["abs_effect_upper"] = bootstrap_results["abs_effect_upper"]
                            output["rel_effect_lower"] = bootstrap_results["rel_effect_lower"]
                            output["rel_effect_upper"] = bootstrap_results["rel_effect_upper"]
                            output["stat_significance"] = 1 if output["pvalue"] < self._alpha else 0
                        else:
                            # For asymptotic, add CI columns for consistency
                            output["abs_effect_lower"] = output.get("abs_effect_lower", np.nan)
                            output["abs_effect_upper"] = output.get("abs_effect_upper", np.nan)
                            output["rel_effect_lower"] = output.get("rel_effect_lower", np.nan)
                            output["rel_effect_upper"] = output.get("rel_effect_upper", np.nan)

                        # Add inference_method after all inference columns for consistent ordering
                        output["inference_method"] = "bootstrap" if self._bootstrap else "asymptotic"
                        output["adjustment"] = adjustment_label
                        if adjustment == "balance":
                            output["method"] = self._balance_method
                        if adjustment == "balance" and not adjusted_balance.empty:
                            output["balance"] = np.round(adjusted_balance["balance_flag"].mean(), 2)
                        elif not balance.empty:  # Use initial balance if no adjustment or balance failed
                            output["balance"] = np.round(balance["balance_flag"].mean(), 2)
                        else:
                            output["balance"] = np.nan  # No balance calculated

                        # Compute ESS for treatment and control if balance adjustment
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

                        # Add treatment and control group identifiers
                        output["treatment_group"] = treatment_val
                        output["control_group"] = control_val
                        output["experiment"] = experiment_tuple
                        output["sample_ratio"] = sample_ratio
                        output["srm_detected"] = srm_detected
                        output["srm_pvalue"] = srm_pvalue
                        temp_results.append(output)

                    except Exception as e:
                        self._logger.error(
                            f"Error processing outcome '{outcome}' for comparison {treatment_val} vs {control_val} with adjustment '{adjustment_label}': {e}"  # noqa: E501
                        )  # noqa: E501
                        error_output = {
                            "outcome": outcome,
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
                            "balance": output.get("balance", np.nan),
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

        # Save all weights if balance adjustment was used
        if weights_list:
            self._weights = pd.concat(weights_list, ignore_index=True)
        else:
            self._weights = None

        # Define result columns, ensure 'balance' is included conditionally
        result_columns = [
            "experiment",
            "outcome",
            "treatment_group",
            "control_group",
            "sample_ratio",
            "adjustment",
            "inference_method",
            "treatment_units",
            "control_units",
            "control_value",
            "treatment_value",
            "absolute_effect",
            "standard_error",
            "pvalue",
            "stat_significance",
            "abs_effect_lower",
            "abs_effect_upper",
            "relative_effect",
            "rel_effect_lower",
            "rel_effect_upper",
            "se_intercept",
            "cov_coef_intercept",
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

        # Transform tuple column in the final results dataframe
        results_df = self.__transform_tuple_column(clean_temp_results, "experiment", self._experiment_identifier)

        # Add confidence intervals based on inference method
        if not results_df.empty and "absolute_effect" in results_df.columns and "standard_error" in results_df.columns:
            alpha = getattr(self, "_alpha", 0.05)
            z_critical = stats.norm.ppf(1 - alpha / 2)

            # Add CI columns if they don't exist (from bootstrap)
            if "abs_effect_lower" not in results_df.columns:
                results_df["abs_effect_lower"] = np.nan
            if "abs_effect_upper" not in results_df.columns:
                results_df["abs_effect_upper"] = np.nan

            # Calculate asymptotic CIs for rows using asymptotic inference
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

        # Apply p-value adjustment if specified
        if self._pvalue_adjustment:
            self._logger.info(f"Applying {self._pvalue_adjustment} p-value adjustment")
            self.adjust_pvalues(method=self._pvalue_adjustment)

        # Restore bootstrap setting if it was temporarily overridden
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

        # Validate alpha
        if not (0 < alpha < 1):
            log_and_raise_error(self._logger, "Alpha must be between 0 and 1 (exclusive).")

        # Validate margin inputs
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

        # pvalue
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

        # keep initial order
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

    def __stratified_resample(self, data: pd.DataFrame, seed: int | None = None) -> pd.DataFrame:
        """
        Perform stratified resampling with replacement, stratified by treatment group.

        Parameters
        ----------
        data : pd.DataFrame
            Data to resample
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Resampled data
        """
        if seed is not None:
            np.random.seed(seed)

        if self._bootstrap_stratify:
            # Stratified resampling by treatment group
            treated = data[data[self._treatment_col] == 1]
            control = data[data[self._treatment_col] == 0]

            treated_resample = treated.sample(n=len(treated), replace=True)
            control_resample = control.sample(n=len(control), replace=True)

            resampled_data = pd.concat([treated_resample, control_resample], ignore_index=True)
        else:
            # Simple resampling
            resampled_data = data.sample(n=len(data), replace=True)

        return resampled_data

    def __bootstrap_single_effect(
        self,
        data: pd.DataFrame,
        outcome: str,
        adjustment: str | None,
        model_func,
        relevant_covariates: list[str],
        numeric_covariates: list[str],
        binary_covariates: list[str],
        min_binary_count: int,
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
        bootstrap_abs_effects = []
        bootstrap_rel_effects = []

        for i in range(self._bootstrap_iterations):
            seed = self._bootstrap_seed + i if self._bootstrap_seed is not None else None
            boot_data = self.__stratified_resample(data, seed=seed)

            try:
                # Impute missing values
                boot_data = self.impute_missing_values(
                    data=boot_data.copy(),
                    num_covariates=numeric_covariates,
                    bin_covariates=binary_covariates,
                )

                # Filter covariates
                boot_numeric = [c for c in numeric_covariates if boot_data[c].std(ddof=0) != 0]
                boot_binary = [c for c in binary_covariates if boot_data[c].sum() >= min_binary_count]
                boot_binary = [c for c in boot_binary if boot_data[c].std(ddof=0) != 0]

                boot_final_covariates = boot_numeric + boot_binary

                if len(boot_final_covariates) > 0:
                    boot_data = self.standardize_covariates(boot_data, boot_final_covariates)

                # Recalculate weights if needed
                if adjustment == "balance":
                    get_balance_method = self._estimator.get_balance_method
                    balance_func = get_balance_method(self._balance_method)
                    boot_data = balance_func(data=boot_data, covariates=[f"z_{cov}" for cov in boot_final_covariates])
                    weight_col = self._target_weights[self._target_effect]
                else:
                    weight_col = None

                # Estimate effect
                boot_relevant_covariates = set(boot_final_covariates) & set(relevant_covariates)

                if adjustment == "balance" and weight_col:
                    output = model_func(
                        data=boot_data,
                        outcome_variable=outcome,
                        covariates=list(boot_relevant_covariates),
                        weight_column=weight_col,
                    )
                else:
                    output = model_func(
                        data=boot_data, outcome_variable=outcome, covariates=list(boot_relevant_covariates)
                    )

                bootstrap_abs_effects.append(output["absolute_effect"])
                bootstrap_rel_effects.append(output["relative_effect"])

            except Exception as e:
                # Skip failed bootstrap iterations
                self._logger.debug(f"Bootstrap iteration {i} failed: {e}")
                continue

        if len(bootstrap_abs_effects) < self._bootstrap_iterations * 0.5:
            self._logger.warning(
                f"More than 50% of bootstrap iterations failed for outcome {outcome}. Results may be unreliable."
            )

        return np.array(bootstrap_abs_effects), np.array(bootstrap_rel_effects)

    def __calculate_bootstrap_inference(
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
            Dictionary with standard error, CI, and p-value (using standard column names)
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
            # Relative effect CI from bootstrap distribution
            if len(bootstrap_rel_effects_clean) > 0:
                rel_ci_lower = np.percentile(bootstrap_rel_effects_clean, alpha / 2 * 100)
                rel_ci_upper = np.percentile(bootstrap_rel_effects_clean, (1 - alpha / 2) * 100)
            else:
                rel_ci_lower = np.nan
                rel_ci_upper = np.nan
        elif self._bootstrap_ci_method == "basic":
            # Basic bootstrap CI
            abs_ci_lower = 2 * observed_abs_effect - np.percentile(bootstrap_abs_effects_clean, (1 - alpha / 2) * 100)
            abs_ci_upper = 2 * observed_abs_effect - np.percentile(bootstrap_abs_effects_clean, alpha / 2 * 100)
            if len(bootstrap_rel_effects_clean) > 0:
                rel_ci_lower = 2 * observed_rel_effect - np.percentile(bootstrap_rel_effects_clean, (1 - alpha / 2) * 100)
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

        pvalue = (
            np.mean(np.abs(bootstrap_abs_effects_clean - np.mean(bootstrap_abs_effects_clean)) >= np.abs(observed_abs_effect)) * 2
        )
        pvalue = min(pvalue, 1.0)  # Ensure p-value doesn't exceed 1

        return {
            "standard_error": bootstrap_se,
            "pvalue": pvalue,
            "abs_effect_lower": abs_ci_lower,
            "abs_effect_upper": abs_ci_upper,
            "rel_effect_lower": rel_ci_lower,
            "rel_effect_upper": rel_ci_upper,
        }

    @property
    def results(self) -> pd.DataFrame | None:
        """
        Returns the results DataFrame
        """
        if self._results is not None:
            return self._results
        else:
            self._logger.warning("Run the `get_effects` function first!")
            return None

    @property
    def balance(self) -> pd.DataFrame | None:
        """
        Returns the balance DataFrame
        """
        if not self._balance.empty:
            return self._balance
        else:
            self._logger.warning("No balance information available!")
            return None

    @property
    def adjusted_balance(self) -> pd.DataFrame | None:
        """
        Returns the adjusted balance DataFrame
        """
        if not self._adjusted_balance.empty:
            return self._adjusted_balance
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
        # Return as (higher, lower) for each pair
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
            else:
                result_dict["rel_effect_lower_mcp"] = np.nan
                result_dict["rel_effect_upper_mcp"] = np.nan

            return pd.DataFrame(result_dict, index=group.index)

        if "se_intercept" in df_adj.columns and "cov_coef_intercept" in df_adj.columns:
            self._logger.info(
                "Computing adjusted relative effect CIs using Fieller's method with adjusted alpha. "
                "This properly accounts for uncertainty in both numerator and denominator."
            )
        else:
            self._logger.warning(
                "Covariance information (se_intercept, cov_coef_intercept) not found in results. "
                "Relative effect CIs with MCP adjustment will be set to NaN. "
                "This may occur with bootstrap results. To obtain adjusted relative CIs, "
                "re-run get_effects() with bootstrap=False or with appropriate alpha parameter."
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
        self._results = df_final

    def calculate_retrodesign(
        self,
        true_effect: float | dict | None = None,
        alpha: float | None = None,
        outcomes: list[str] | None = None,
        experiments: list | None = None,
    ) -> pd.DataFrame:
        """
        Calculate retrodesign metrics (Gelman & Carlin) for existing results.

        Retrodesign analysis calculates:
        - Power: probability of achieving statistical significance given true effect
        - Type S error: probability of getting the wrong sign (sign error rate)
        - Type M error: expected exaggeration ratio (magnitude error)
        - Relative bias: expected bias ratio preserving signs (Jaksic et al. 2026)

        The exaggeration ratio tells you how much the effect size is expected to be
        overestimated when you get a statistically significant result from an
        underpowered study.

        Parameters
        ----------
        true_effect : float, dict, or None
            The hypothesized true effect size. Can be:
            - A single float to apply to all results
            - A dict mapping outcome names to their true effects:
              {'outcome1': 0.02, 'outcome2': 0.03}
            - A dict mapping (treatment_group, control_group) tuples to true effects:
              {('treatment_a', 'control'): 0.02, ('treatment_b', 'control'): 0.05}
            - A dict mapping (outcome, treatment_group, control_group) tuples:
              {('outcome1', 'treatment_a', 'control'): 0.02,
               ('outcome1', 'treatment_b', 'control'): 0.03}
            - None to use the observed effect as the assumed true effect (conservative)
        alpha : float, optional
            Significance level. If None, uses self._alpha
        outcomes : list of str, optional
            Filter to specific outcomes. If None, uses all outcomes
        experiments : list, optional
            Filter to specific experiments. If None, uses all experiments

        Returns
        -------
        pd.DataFrame
            Original results with added columns:
            - true_effect: The assumed true effect size
            - power: Probability of significance given true effect
            - type_s_error: Probability of wrong sign
            - type_m_error: Expected exaggeration ratio using absolute values
            - relative_bias: Expected bias ratio preserving signs (Jaksic et al. 2026)
              This is typically lower than type_m_error because negative significant
              results (Type S errors) partially offset positive overestimates

        Examples
        --------
        >>> analyzer.get_effects()
        >>> # Use observed effects as true effects (conservative)
        >>> retro = analyzer.calculate_retrodesign()
        >>> # Single true effect for all
        >>> retro = analyzer.calculate_retrodesign(true_effect=0.05)
        >>> # Different true effects per outcome
        >>> retro = analyzer.calculate_retrodesign(
        ...     true_effect={'outcome1': 0.05, 'outcome2': 0.03}
        ... )
        >>> # Different true effects per comparison
        >>> retro = analyzer.calculate_retrodesign(
        ...     true_effect={
        ...         ('treatment_a', 'control'): 0.02,
        ...         ('treatment_b', 'control'): 0.05,
        ...         ('treatment_b', 'treatment_a'): 0.03
        ...     }
        ... )
        >>> # Different true effects per outcome AND comparison
        >>> retro = analyzer.calculate_retrodesign(
        ...     true_effect={
        ...         ('outcome1', 'treatment_a', 'control'): 0.02,
        ...         ('outcome1', 'treatment_b', 'control'): 0.05,
        ...         ('outcome2', 'treatment_a', 'control'): 0.01,
        ...     }
        ... )
        """
        if self._results is None:
            log_and_raise_error(self._logger, "Run get_effects() first.")

        df = self._results.copy()
        alpha_val = alpha if alpha is not None else self._alpha

        # Filter by outcomes and experiments if specified
        mask = pd.Series([True] * len(df))
        if outcomes is not None:
            mask &= df["outcome"].isin(outcomes)
        if experiments is not None:
            group_cols = self._experiment_identifier or ["experiment"]
            for col, vals in zip(group_cols, zip(*experiments, strict=False), strict=False):
                mask &= df[col].isin(vals)

        df_filtered = df[mask].copy()

        # Drop existing retrodesign columns if they exist (from previous calls)
        retro_cols = ["true_effect", "power", "type_s_error", "type_m_error", "relative_bias"]
        existing_retro_cols = [col for col in retro_cols if col in df_filtered.columns]
        if existing_retro_cols:
            df_filtered = df_filtered.drop(columns=existing_retro_cols)

        # Determine true effect for each row
        if true_effect is None:
            # Use observed effect as assumed true effect (conservative)
            self._logger.info(
                "No true_effect specified. Using observed effects as assumed true effects (conservative approach)."
            )
            df_filtered["true_effect"] = df_filtered["absolute_effect"]
        elif isinstance(true_effect, dict):
            # Check if dict keys are tuples to determine mapping type
            sample_key = next(iter(true_effect.keys()))

            if isinstance(sample_key, tuple):
                # Tuple-based mapping
                if len(sample_key) == 2:
                    # (treatment_group, control_group) mapping
                    self._logger.info(
                        f"Using true_effect mapping by (treatment_group, control_group): {len(true_effect)} comparison(s)"  # noqa: E501
                    )
                    df_filtered["true_effect"] = df_filtered.apply(
                        lambda row: true_effect.get(
                            (row["treatment_group"], row["control_group"]),
                            row["absolute_effect"],  # fallback to observed
                        ),
                        axis=1,
                    )
                elif len(sample_key) == 3:
                    # (outcome, treatment_group, control_group) mapping
                    self._logger.info(
                        f"Using true_effect mapping by (outcome, treatment_group, control_group): {len(true_effect)} combination(s)"  # noqa: E501
                    )
                    df_filtered["true_effect"] = df_filtered.apply(
                        lambda row: true_effect.get(
                            (row["outcome"], row["treatment_group"], row["control_group"]),
                            row["absolute_effect"],  # fallback to observed
                        ),
                        axis=1,
                    )
                else:
                    log_and_raise_error(
                        self._logger,
                        f"Dict keys must be tuples of length 2 (treatment_group, control_group) "
                        f"or 3 (outcome, treatment_group, control_group), got length {len(sample_key)}",
                    )
            else:
                # String-based mapping (outcome names)
                self._logger.info(f"Using true_effect mapping by outcome: {true_effect}")
                df_filtered["true_effect"] = df_filtered["outcome"].map(true_effect)
                # Fill any unmapped with observed effect
                df_filtered.loc[:, "true_effect"] = df_filtered["true_effect"].fillna(df_filtered["absolute_effect"])
        else:
            # Single value for all
            self._logger.info(f"Using single true_effect value for all comparisons: {true_effect}")
            df_filtered["true_effect"] = true_effect

        # Calculate retrodesign metrics
        z_crit = stats.norm.ppf(1 - alpha_val / 2)  # Two-tailed critical value

        # Calculate for each row
        results = []
        for _idx, row in df_filtered.iterrows():
            te = row["true_effect"]  # true effect
            se = row["standard_error"]

            if pd.isna(te) or pd.isna(se) or se <= 0:
                # Can't calculate retrodesign
                power = np.nan
                type_s = np.nan
                type_m = np.nan
                relative_bias = np.nan
            else:
                noncentrality = te / se
                power = 1 - (stats.norm.cdf(z_crit - noncentrality) - stats.norm.cdf(-z_crit - noncentrality))

                if te == 0:
                    type_s = 0.5  # Equally likely to be positive or negative
                else:
                    prob_wrong_sign = stats.norm.cdf(-z_crit - noncentrality)
                    type_s = prob_wrong_sign / power if power > 0 else np.nan

                upper_tail = 1 - stats.norm.cdf(z_crit - noncentrality)
                lower_tail = stats.norm.cdf(-z_crit - noncentrality)

                if te == 0:
                    type_m = np.inf
                    relative_bias = np.inf
                else:
                    if power > 0:
                        if upper_tail > 0:
                            e_upper = noncentrality + stats.norm.pdf(z_crit - noncentrality) / upper_tail
                        else:
                            e_upper = 0

                        if lower_tail > 0:
                            e_lower = noncentrality - stats.norm.pdf(-z_crit - noncentrality) / lower_tail
                        else:
                            e_lower = 0

                        expected_abs_z = (upper_tail * abs(e_upper) + lower_tail * abs(e_lower)) / power
                        expected_abs_estimate = expected_abs_z * se
                        type_m = abs(expected_abs_estimate / te)

                        expected_z_signed = (upper_tail * e_upper + lower_tail * e_lower) / power
                        expected_estimate_signed = expected_z_signed * se
                        relative_bias = expected_estimate_signed / te
                    else:
                        type_m = np.nan
                        relative_bias = np.nan

            results.append(
                {
                    "power": power,
                    "type_s_error": type_s,
                    "type_m_error": type_m,
                    "relative_bias": relative_bias,
                }
            )

        retro_df = pd.DataFrame(results, index=df_filtered.index)
        df_filtered = pd.concat([df_filtered, retro_df], axis=1)

        return df_filtered
