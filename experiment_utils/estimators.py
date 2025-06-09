""" "This module contains classes for performing causal inference using various estimators."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

from .entbal import entbal, ess
from .utils import get_logger, log_and_raise_error


class Estimators:
    """
    A class for performing causal inference using various estimators.
    """

    def __init__(
        self,
        treatment_col: str,
        instrument_col: str | None = None,
        target_ipw_effect: str = "ATT",
        target_weights: dict[str, str] = None,
        alpha: float = 0.05,
        min_ps_score: float = 0.05,
        max_ps_score: float = 0.95,
        polynomial_ipw: bool = False,
    ) -> None:
        self._logger = get_logger("Estimators")
        self._treatment_col = treatment_col
        self._instrument_col = instrument_col
        self._target_ipw_effect = target_ipw_effect
        self._target_weights = target_weights
        self._alpha = alpha
        self._max_ps_score = max_ps_score
        self._min_ps_score = min_ps_score
        self._polynomial_ipw = (polynomial_ipw,)

    def __create_formula(
        self, outcome_variable: str, covariates: list[str] | None, model_type: str = "regression"
    ) -> str:  # noqa: E501
        """
        Create the formula for the regression model.

        Parameters:
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.
        model_type (str): The type of regression model to perform.

        Returns:
        str: The formula for the regression model.
        """

        formula_dict = {
            "regression": f"{outcome_variable} ~ 1 + {self._treatment_col}",
            "iv": f"{outcome_variable} ~ 1 + [{self._treatment_col} ~ {self._instrument_col}]",
        }
        if covariates:
            standardized_covariates = [f"z_{covariate}" for covariate in covariates]
            formula = formula_dict[model_type] + " + " + " + ".join(standardized_covariates)
        else:
            formula = formula_dict[model_type]
        return formula

    def linear_regression(
        self, data: pd.DataFrame, outcome_variable: str, covariates: list[str] | None = None
    ) -> dict[str, str | int | float]:  # noqa: E501
        """
        Perform linear regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.

        Returns:
        Dict: A dictionary containing the results of the regression, including:
            - "outcome" (str): The name of the outcome variable.
            - "treated_units" (int): The number of treated units in the data.
            - "control_units" (int): The number of control units in the data.
            - "control_value" (float): The intercept of the regression model.
            - "treatment_value" (float): The predicted value for the treatment group.
            - "absolute_effect" (float): The coefficient of the treatment variable.
            - "relative_effect" (float): The relative effect of the treatment.
            - "standard_error" (float): The standard error of the treatment coefficient.
            - "pvalue" (float): The p-value of the treatment coefficient.
            - "stat_significance" (int): Indicator of statistical significance (1 if p-value < alpha, else 0).
        """

        formula = self.__create_formula(outcome_variable=outcome_variable, covariates=covariates)
        model = smf.ols(formula, data=data)
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        return {
            "outcome": outcome_variable,
            "treated_units": data[self._treatment_col].sum(),
            "control_units": data[self._treatment_col].count() - data[self._treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
        }

    def weighted_least_squares(
        self, data: pd.DataFrame, outcome_variable: str, weight_column: str, covariates: list[str] | None = None
    ) -> dict[str, str | int | float]:
        """
        Perform weighted least squares regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        weight_column (str): The name of the column containing the weights for the regression.
        covariates (List[str]): The list of covariates to include in the regression model.

        Returns:
        Dict: A dictionary containing the results of the regression, including:
            - "outcome" (str): The name of the outcome variable.
            - "treated_units" (int): The number of treated units in the data.
            - "control_units" (int): The number of control units in the data.
            - "control_value" (float): The intercept of the regression model.
            - "treatment_value" (float): The predicted value for the treatment group.
            - "absolute_effect" (float): The coefficient of the treatment variable.
            - "relative_effect" (float): The relative effect of the treatment.
            - "standard_error" (float): The standard error of the treatment coefficient.
            - "pvalue" (float): The p-value of the treatment coefficient.
            - "stat_significance" (int): Indicator of statistical significance (1 if p-value < alpha, else 0).
        """
        formula = self.__create_formula(outcome_variable=outcome_variable, covariates=covariates)
        model = smf.wls(
            formula,
            data=data,
            weights=data[weight_column],
        )
        results = model.fit(cov_type="HC3")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        weights = data[weight_column].values
        treat = data[self._treatment_col].values
        n_control = np.sum(treat == 0)
        n_treatment = np.sum(treat == 1)
        ess_control = round(ess(weights[treat == 0]), 2) if n_control > 0 else np.nan
        ess_treatment = round(ess(weights[treat == 1]), 2) if n_treatment > 0 else np.nan
        control_sample_reduction = 100 * (1 - ess_control / n_control) if n_control > 0 else np.nan
        treatment_sample_reduction = 100 * (1 - ess_treatment / n_treatment) if n_treatment > 0 else np.nan

        return {
            "outcome": outcome_variable,
            "treated_units": data[self._treatment_col].sum().astype(int),
            "control_units": (data[self._treatment_col].count() - data[self._treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "ess_control": ess_control,
            "ess_treatment": ess_treatment,
            "control_sample_reduction": control_sample_reduction,
            "treatment_sample_reduction": treatment_sample_reduction,
        }

    def iv_regression(
        self, data: pd.DataFrame, outcome_variable: str, covariates: list[str] | None = None
    ) -> dict[str, str | int | float]:  # noqa: E501
        """ "
        Perform instrumental variable regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.

        Returns:
        Dict: A dictionary containing the results of the regression, including:
            - "outcome" (str): The name of the outcome variable.
            - "treated_units" (int): The number of treated units in the data.
            - "control_units" (int): The number of control units in the data.
            - "control_value" (float): The intercept of the regression model.
            - "treatment_value" (float): The predicted value for the treatment group.
            - "absolute_effect" (float): The coefficient of the treatment variable.
            - "relative_effect" (float): The relative effect of the treatment.
            - "standard_error" (float): The standard error of the treatment coefficient.
            - "pvalue" (float): The p-value of the treatment coefficient.
            - "stat_significance" (int): Indicator of statistical significance (1 if p-value < alpha, else 0).
        """
        if not self._instrument_col:
            log_and_raise_error(self._logger, "Instrument column must be specified for IV adjustment")

        formula = self.__create_formula(outcome_variable=outcome_variable, model_type="iv", covariates=covariates)
        model = IV2SLS.from_formula(formula, data)
        results = model.fit(cov_type="robust")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept
        standard_error = results.std_errors[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        return {
            "outcome": outcome_variable,
            "treated_units": data[self._treatment_col].sum().astype(int),
            "control_units": (data[self._treatment_col].count() - data[self._treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
        }

    def ipw_logistic(
        self, data: pd.DataFrame, covariates: list[str], penalty: str = "l2", C: float = 1.0, max_iter: int = 5000
    ) -> pd.DataFrame:  # noqa: E501
        """
        Estimate the Inverse Probability Weights (IPW) using logistic regression with regularization.

        Parameters
        ----------
        data : pd.DataFrame
            Data to estimate the IPW from
        covariates : List[str]
            List of covariates to include in the estimation
        penalty : str, optional
            Regularization penalty to use in the logistic regression model, by default 'l2'
        C : float, optional
            Inverse of regularization strength, by default 1.0
        max_iter : int, optional


        Returns
        -------
        pd.DataFrame
            Data with the estimated IPW
        """

        logistic_model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter)

        if self._polynomial_ipw:
            poly = PolynomialFeatures()
            X = poly.fit_transform(data[covariates])
            feature_names = poly.get_feature_names_out(covariates)
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = data[covariates]

        y = data[self._treatment_col]

        self._logger.info(
            "Estimating Inverse Probability Weights using logistic regression with covariates: %s",
            ", ".join(covariates),
        )
        logistic_model.fit(X, y)

        if not logistic_model.n_iter_[0] < logistic_model.max_iter:
            self._logger.warning(
                "Logistic regression model did not converge. Consider increasing the number of iterations or adjusting other parameters."  # noqa: E501
            )  # noqa: E501

        data["propensity_score"] = logistic_model.predict_proba(X)[:, 1]
        data["propensity_score"] = np.minimum(self._max_ps_score, data["propensity_score"])
        data["propensity_score"] = np.maximum(self._min_ps_score, data["propensity_score"])

        data = self.__calculate_stabilized_weights(data)

        self._logger.info("Finished estimating Inverse Probability Weights.")
        return data

    def entropy_balancing(self, data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        """
        Perform entropy balancing to create weights.

        This method generates weights for the control group units such that the covariate moments
        in the re-weighted control group match those of the treatment group. Treated units
        receive a weight of 1.

        Parameters
        ----------
        data : pd.DataFrame
            The input data containing the treatment status and covariates.
        covariates : list[str]
            A list of covariate names to use for balancing.

        Returns
        -------
        pd.DataFrame
            The original DataFrame with an added 'eb_weights' column.
        """
        if not covariates:
            log_and_raise_error(self._logger, "Covariates must be specified for entropy balancing.")

        treatment_indicator = data[self._treatment_col]
        covariate_data = data[covariates]

        self._logger.info("Performing entropy balancing with the following covariates: %s", ", ".join(covariates))
        eb = entbal()
        eb.fit(covariate_data, treatment_indicator, estimand=self._target_ipw_effect)
        data[self._target_weights[self._target_ipw_effect]] = eb.W

        self._logger.info("Finished!")
        return data

    def ipw_xgboost(self, data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        """
        Estimate the Inverse Probability Weights (IPW) using XGBoost.

        Parameters:
        data (pd.DataFrame): Data to estimate the IPW from.
        covariates (List[str]): List of covariates to include in the estimation.

        Returns:
        pd.DataFrame: Data with the estimated IPW.
        """

        X = data[covariates]
        y = data[self._treatment_col]

        xgb_model = XGBClassifier(eval_metric="logloss")
        xgb_model.fit(X, y)

        data["propensity_score"] = xgb_model.predict_proba(X)[:, 1]
        data["propensity_score"] = np.minimum(self._max_ps_score, data["propensity_score"])
        data["propensity_score"] = np.maximum(self._min_ps_score, data["propensity_score"])
        data = self.__calculate_stabilized_weights(data)

        return data

    def __calculate_stabilized_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the stabilized weights for IPW.

        Parameters
        ----------
        data : pd.DataFrame
            Data with the estimated propensity scores

        Returns
        -------
        pd.DataFrame
            Data with the calculated stabilized weights
        """
        num_units = data.shape[0]
        p_treatment = sum(data[self._treatment_col]) / num_units

        data["ips_stabilized_weight"] = data[self._treatment_col] / data["propensity_score"] * p_treatment + (
            1 - data[self._treatment_col]
        ) / (1 - data["propensity_score"]) * (1 - p_treatment)

        data["tips_stabilized_weight"] = data[self._treatment_col] * p_treatment + (
            1 - data[self._treatment_col]
        ) * data["propensity_score"] / (1 - data["propensity_score"]) * (1 - p_treatment)

        data["cips_stabilized_weight"] = data[self._treatment_col] * (1 - data["propensity_score"]) / data[
            "propensity_score"
        ] * p_treatment + (1 - data[self._treatment_col]) * (1 - p_treatment)

        return data
