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
        List of covariates
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

        # check if any covariate is a string using pandas dtypes
        if any(pd.api.types.is_string_dtype(self._data[c]) for c in self._covariates):
            log_and_raise_error(
                self._logger, "Covariates should be numeric, for categorical columns use dummy variables!"
            )

        # regression covariates has to be a subset of covariates
        if len(self._regression_covariates) > 0:
            if not set(self._regression_covariates).issubset(set(self._covariates)):
                log_and_raise_error(self._logger, "Regression covariates should be a subset of covariates")

        # create an experiment id if there is not one
        if len(self._experiment_identifier) == 0:
            self._data["experiment_id"] = 1
            self._experiment_identifier = ["experiment_id"]
            self._logger.warning("No experiment identifier, assuming data is from a single experiment!")

        # check if all required columns are present
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

        # Select only required columns using pandas indexing
        self._data = self._data[required_columns]

    def __get_binary_covariates(self, data: pd.DataFrame) -> list[str]:
        binary_covariates = []
        if self._covariates is not None:
            for c in self._covariates:
                if data[c].nunique() == 2 and data[c].max() == 1:
                    binary_covariates.append(c)
        return binary_covariates

    def __get_numeric_covariates(self, data: pd.DataFrame) -> list[str]:
        numeric_covariates = []
        if self._covariates is not None:
            for c in self._covariates:
                if data[c].nunique() > 2:
                    numeric_covariates.append(c)
        return numeric_covariates

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
        - sample_ratio: The sample ratio of the treatment group to the control group.
        - adjustment: The type of adjustment applied.
        - balance: The balance metric for the covariates.
        - treated_units: The number of treated units.
        - control_units: The number of control units.
        - control_value: The mean value of the outcome variable in the control group.
        - treatment_value: The mean value of the outcome variable in the treatment group.
        - absolute_effect: The absolute effect of the treatment.
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

        # Use Estimators.get_balance_method for modularity
        get_balance_method = self._estimator.get_balance_method

        temp_results = []
        output = {}  # Ensure output is always defined for error handling
        weights_list = []

        if adjustment is None:
            adjustment = self._adjustment

        # Allow overriding bootstrap setting for this call
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
            final_covariates = []
            numeric_covariates = self.__get_numeric_covariates(data=temp_pd)
            binary_covariates = self.__get_binary_covariates(data=temp_pd)

            treatvalues = set(temp_pd[self._treatment_col].unique())
            if len(treatvalues) != 2:
                self._logger.warning(f"Skipping {experiment_tuple} as there are no valid treatment-control groups!")
                continue
            if not (0 in treatvalues and 1 in treatvalues):
                log_and_raise_error(
                    self._logger, f"The treatment column {self._treatment_col} must contain only 0 and 1"
                )  # noqa: E501

            sample_ratio, srm_detected, srm_pvalue = self.__check_sample_ratio_mismatch(temp_pd)

            temp_pd = self.impute_missing_values(
                data=temp_pd.copy(),
                num_covariates=numeric_covariates,
                bin_covariates=binary_covariates,
            )

            # remove constant or low frequency covariates
            numeric_covariates = [c for c in numeric_covariates if temp_pd[c].std(ddof=0) != 0]
            binary_covariates = [c for c in binary_covariates if temp_pd[c].sum() >= min_binary_count]
            binary_covariates = [c for c in binary_covariates if temp_pd[c].std(ddof=0) != 0]

            final_covariates = numeric_covariates + binary_covariates
            self._final_covariates = final_covariates
            if len(self._covariates) > len(self._final_covariates):
                self._logger.warning(
                    f"Some covariates were removed due to low variance or frequency in {experiment_tuple}!"
                )

            if len(final_covariates) == 0 & len(self._covariates if self._covariates is not None else []) > 0:
                self._logger.warning(f"No valid covariates for {experiment_tuple}, balance can't be assessed!")

            balance = pd.DataFrame()
            adjusted_balance = pd.DataFrame()

            if len(final_covariates) > 0:
                temp_pd["weights"] = 1
                temp_pd = self.standardize_covariates(temp_pd, final_covariates)
                balance = self.calculate_smd(data=temp_pd, covariates=final_covariates)
                balance["experiment"] = [experiment_tuple] * balance.shape[0]
                balance = self.__transform_tuple_column(balance, "experiment", self._experiment_identifier)
                self._balance.append(balance)
                balance_mean = balance["balance_flag"].mean() if not balance.empty else np.nan
                self._logger.info("::::: Balance: %.2f", np.round(balance_mean, 2))

                if adjustment == "balance":
                    balance_func = get_balance_method(self._balance_method)
                    temp_pd = balance_func(data=temp_pd, covariates=[f"z_{cov}" for cov in final_covariates])

                    adjusted_balance = self.calculate_smd(
                        data=temp_pd,
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
                        if "propensity_score" in temp_pd.columns:
                            treatment_scores = temp_pd.loc[temp_pd[self._treatment_col] == 1, "propensity_score"]
                            control_scores = temp_pd.loc[temp_pd[self._treatment_col] == 0, "propensity_score"]
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
                        if "propensity_score" in temp_pd.columns:
                            self._plot_common_support(
                                temp_pd,
                                treatment_col=self._treatment_col,
                                propensity_col="propensity_score",
                                experiment_id=experiment_tuple,
                            )
                        else:
                            self._logger.warning("Propensity score column not found, skipping overlap plot.")

                # Save weights for this experiment when balance adjustment is used
                if adjustment == "balance":
                    weight_col = self._target_weights[self._target_effect]
                    if weight_col in temp_pd.columns:
                        weight_columns = [*self._experiment_identifier, weight_col]
                        if self._unit_identifier and self._unit_identifier in temp_pd.columns:
                            weight_columns.insert(-1, self._unit_identifier)
                        weights_df = temp_pd[weight_columns].copy()
                        weights_list.append(weights_df)

                if adjustment == "IV":
                    if self._instrument_col is None:
                        log_and_raise_error(self._logger, "Instrument column is required for IV estimation!")
                    iv_balance = self.calculate_smd(
                        data=temp_pd, treatment_col=self._instrument_col, covariates=final_covariates
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
                # Ensure outcome column is numeric
                if not pd.api.types.is_numeric_dtype(temp_pd[outcome]):
                    self._logger.warning(
                        f"Outcome '{outcome}' is not numeric for experiment {experiment_tuple}. Skipping."
                    )  # noqa: E501
                    continue
                # Ensure outcome has variance
                if temp_pd[outcome].var() == 0:
                    self._logger.warning(
                        f"Outcome '{outcome}' has zero variance for experiment {experiment_tuple}. Skipping."
                    )  # noqa: E501
                    continue

                try:
                    if adjustment == "balance":
                        # Use correct target_effect for weight column
                        weight_col = self._target_weights[self._target_effect]
                        if weight_col not in temp_pd.columns:
                            log_and_raise_error(
                                self._logger,
                                f"Weight column '{weight_col}' not found after balance calculation.",
                            )
                        if temp_pd[weight_col].var() == 0:
                            self._logger.warning(
                                f"Weight column '{weight_col}' has zero variance for experiment {experiment_tuple}. Results might be unreliable."  # noqa: E501
                            )
                        output = model[adjustment](
                            data=temp_pd,
                            outcome_variable=outcome,
                            covariates=list(relevant_covariates),
                            weight_column=weight_col,
                        )
                    else:
                        output = model[adjustment](
                            data=temp_pd, outcome_variable=outcome, covariates=list(relevant_covariates)
                        )

                    # Bootstrap inference if requested
                    if self._bootstrap:
                        self._logger.info(
                            f"Running bootstrap for outcome '{outcome}' with {self._bootstrap_iterations} iterations..."
                        )
                        bootstrap_effects = self.__bootstrap_single_effect(
                            data=temp_pd,
                            outcome=outcome,
                            adjustment=adjustment,
                            model_func=model[adjustment],
                            relevant_covariates=list(relevant_covariates),
                            numeric_covariates=numeric_covariates,
                            binary_covariates=binary_covariates,
                            min_binary_count=min_binary_count,
                        )
                        bootstrap_results = self.__calculate_bootstrap_inference(
                            bootstrap_effects, output["absolute_effect"]
                        )
                        # Replace asymptotic inference with bootstrap inference
                        output["standard_error"] = bootstrap_results["standard_error"]
                        output["pvalue"] = bootstrap_results["pvalue"]
                        output["abs_effect_lower"] = bootstrap_results["abs_effect_lower"]
                        output["abs_effect_upper"] = bootstrap_results["abs_effect_upper"]
                        output["stat_significance"] = 1 if output["pvalue"] < self._alpha else 0
                    else:
                        # For asymptotic, add CI columns for consistency
                        output["abs_effect_lower"] = output.get("abs_effect_lower", np.nan)
                        output["abs_effect_upper"] = output.get("abs_effect_upper", np.nan)

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
                        if weight_col in temp_pd.columns:
                            treat_mask = temp_pd[self._treatment_col] == 1
                            control_mask = temp_pd[self._treatment_col] == 0
                            treat_weights = temp_pd.loc[treat_mask, weight_col]
                            control_weights = temp_pd.loc[control_mask, weight_col]
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

                    output["experiment"] = experiment_tuple
                    output["sample_ratio"] = sample_ratio
                    output["srm_detected"] = srm_detected
                    output["srm_pvalue"] = srm_pvalue
                    temp_results.append(output)

                except Exception as e:
                    self._logger.error(
                        f"Error processing outcome '{outcome}' for experiment {experiment_tuple} with adjustment '{adjustment_label}': {e}"  # noqa: E501
                    )  # noqa: E501
                    # Optionally append a row with NaNs or skip
                    error_output = {
                        "outcome": outcome,
                        "adjustment": adjustment_label,
                        "treated_units": np.nan,
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
            "sample_ratio",
            "adjustment",
            "inference_method",
            "treated_units",
            "control_units",
            "control_value",
            "treatment_value",
            "absolute_effect",
            "relative_effect",
            "standard_error",
            "pvalue",
            "stat_significance",
            "abs_effect_lower",
            "abs_effect_upper",
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

        # Restore original bootstrap setting if it was overridden
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
            .apply(lambda df: pd.Series(self.__get_fixed_meta_analysis_estimate(df)))
            .reset_index()
        )

        result_columns = grouping_cols + [
            "experiments",
            "control_units",
            "treated_units",
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
            "treated_units": int(data["treated_units"].sum()),
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
        group["gweight"] = group["treated_units"].astype(int)
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
                "treated_units": int(np.sum(group["gweight"])),
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
    ) -> float:
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
        float
            Bootstrapped absolute effect estimate
        """
        bootstrap_effects = []

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

                bootstrap_effects.append(output["absolute_effect"])

            except Exception as e:
                # Skip failed bootstrap iterations
                self._logger.debug(f"Bootstrap iteration {i} failed: {e}")
                continue

        if len(bootstrap_effects) < self._bootstrap_iterations * 0.5:
            self._logger.warning(
                f"More than 50% of bootstrap iterations failed for outcome {outcome}. Results may be unreliable."
            )

        return np.array(bootstrap_effects)

    def __calculate_bootstrap_inference(
        self, bootstrap_effects: np.ndarray, observed_effect: float
    ) -> dict[str, float]:
        """
        Calculate confidence intervals and p-value from bootstrap distribution.

        Parameters
        ----------
        bootstrap_effects : np.ndarray
            Array of bootstrap effect estimates
        observed_effect : float
            Observed effect estimate from original data

        Returns
        -------
        dict
            Dictionary with standard error, CI, and p-value (using standard column names)
        """
        if len(bootstrap_effects) == 0:
            return {
                "standard_error": np.nan,
                "pvalue": np.nan,
                "abs_effect_lower": np.nan,
                "abs_effect_upper": np.nan,
            }

        # Remove NaN values
        bootstrap_effects = bootstrap_effects[~np.isnan(bootstrap_effects)]

        if len(bootstrap_effects) == 0:
            return {
                "standard_error": np.nan,
                "pvalue": np.nan,
                "abs_effect_lower": np.nan,
                "abs_effect_upper": np.nan,
            }

        # Calculate bootstrap standard error
        bootstrap_se = np.std(bootstrap_effects, ddof=1)

        # Confidence intervals
        alpha = self._alpha
        if self._bootstrap_ci_method == "percentile":
            ci_lower = np.percentile(bootstrap_effects, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)
        elif self._bootstrap_ci_method == "basic":
            # Basic bootstrap CI
            ci_lower = 2 * observed_effect - np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)
            ci_upper = 2 * observed_effect - np.percentile(bootstrap_effects, alpha / 2 * 100)
        else:
            # Default to percentile
            ci_lower = np.percentile(bootstrap_effects, alpha / 2 * 100)
            ci_upper = np.percentile(bootstrap_effects, (1 - alpha / 2) * 100)

        # P-value (two-sided)
        # Count how many bootstrap effects are as extreme as observed
        pvalue = np.mean(np.abs(bootstrap_effects - np.mean(bootstrap_effects)) >= np.abs(observed_effect)) * 2
        pvalue = min(pvalue, 1.0)  # Ensure p-value doesn't exceed 1

        return {
            "standard_error": bootstrap_se,
            "pvalue": pvalue,
            "abs_effect_lower": ci_lower,
            "abs_effect_upper": ci_upper,
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

    def adjust_pvalues(
        self,
        method: str = "bonferroni",
        groupby_cols: list[str] | None = None,
        outcomes: list[str] | None = None,
        experiments: list | None = None,
        alpha: float | None = 0.05,
    ):
        if self._results is None:
            log_and_raise_error(self._logger, "Run get_effects() first.")

        df = self._results.copy()
        group_cols = groupby_cols or self._experiment_identifier or ["experiment"]
        if not all(col in df.columns for col in group_cols):
            log_and_raise_error(self._logger, f"Grouping columns {group_cols} not found in results.")

        mask = pd.Series([True] * len(df))
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

            return pd.DataFrame(
                {
                    "pvalue_adj": pvals_adj,
                    "stat_significance_adj": (pvals_adj < thres).astype(int),
                    "adj_method": method,
                },
                index=group.index,
            )

        adjustments = df_adj.groupby(group_cols, group_keys=False).apply(calculate_adjustments, include_groups=False)
        cols_to_add = ["pvalue_adj", "stat_significance_adj", "adj_method"]

        df_adj = df_adj.drop(columns=cols_to_add, errors="ignore")
        df_adj = df_adj.join(adjustments)
        df_final = pd.concat([df_adj, df_rest], ignore_index=True).sort_index()
        self._results = df_final
