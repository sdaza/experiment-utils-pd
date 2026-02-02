""" "This module contains classes for performing causal inference using various estimators."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBClassifier

from .entbal import EntropyBalance
from .utils import get_logger, log_and_raise_error


class Estimators:
    """
    A class for performing causal inference using various estimators.

    Supports adjustment methods: 'balance' (with balance_method: 'ps-logistic', 'ps-xgboost', 'entropy') and 'IV'.
    """

    def __init__(
        self,
        treatment_col: str,
        instrument_col: str | None = None,
        target_effect: str = "ATT",
        target_weights: dict = None,
        alpha: float = 0.05,
        min_ps_score: float = 0.05,
        max_ps_score: float = 0.95,
        polynomial_ipw: bool = False,
    ) -> None:
        self._logger = get_logger("Estimators")
        self._treatment_col = treatment_col
        self._instrument_col = instrument_col
        self._target_effect = target_effect
        self._target_weights = target_weights
        self._alpha = alpha
        self._max_ps_score = max_ps_score
        self._min_ps_score = min_ps_score
        self._polynomial_ipw = polynomial_ipw

    def get_balance_method(self, method_name: str):
        """
        Returns the balance method function based on the method name.
        Supported: 'ps-logistic', 'ps-xgboost', 'entropy'.
        """
        balance_methods = {
            "ps-logistic": self.ipw_logistic,
            "ps-xgboost": self.ipw_xgboost,
            "entropy": self.entropy_balance,
        }
        if method_name not in balance_methods:
            log_and_raise_error(self._logger, f"Unknown balance_method: {method_name}")
        return balance_methods[method_name]

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

    def _compute_fieller_ci(
        self,
        coefficient: float,
        intercept: float,
        se_coef: float,
        se_intercept: float,
        cov_matrix: np.ndarray,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Compute Fieller confidence interval for ratio coefficient/intercept using Fieller's theorem.

        Fieller's theorem provides exact confidence intervals for ratios of normally distributed
        random variables. This is the correct method for computing CIs on relative effects (lift)
        in A/B tests, as it properly accounts for uncertainty in both numerator and denominator.

        The naive approach of dividing absolute effect CI bounds by the control value is
        mathematically incorrect and can severely underestimate uncertainty, especially with
        small samples.

        Parameters
        ----------
        coefficient : float
            Treatment effect (numerator of ratio)
        intercept : float
            Control mean (denominator of ratio)
        se_coef : float
            Standard error of coefficient
        se_intercept : float
            Standard error of intercept
        cov_matrix : np.ndarray
            Covariance matrix from regression results
        alpha : float
            Significance level (default 0.05 for 95% CI)

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound) or (nan, nan) if interval cannot be computed

        References
        ----------
        Fieller, E. C. (1954). "Some Problems in Interval Estimation".
        Journal of the Royal Statistical Society, Series B. 16 (2): 175-185.

        Franz, V. H. (2007). "Ratios: A short guide to confidence limits and proper use".
        https://arxiv.org/abs/0710.2024
        """
        from scipy import stats

        # Handle edge cases
        if intercept == 0:
            self._logger.warning("Control value is zero. Cannot compute relative effect CI.")
            return (np.nan, np.nan)

        # Use small value threshold to avoid numerical issues
        if abs(intercept) < 1e-10:
            self._logger.warning(
                f"Control value very close to zero ({intercept:.2e}). Relative effect CI may be unreliable."
            )
            return (np.nan, np.nan)

        # Critical value (z for large samples, can use t for small samples)
        z_crit = stats.norm.ppf(1 - alpha / 2)
        g_sq = z_crit**2

        # Extract covariance between treatment coefficient and intercept
        # The covariance matrix has treatment_col and "Intercept" as indices
        try:
            cov = cov_matrix.loc[self._treatment_col, "Intercept"]
        except (KeyError, AttributeError):
            # If cov_matrix is numpy array, try positional access
            # Typically intercept is first, treatment is second
            if isinstance(cov_matrix, np.ndarray):
                # Assume [Intercept, treatment_col, ...] order
                cov = cov_matrix[0, 1] if cov_matrix.shape[0] > 1 else 0
            else:
                self._logger.warning("Could not extract covariance from matrix. Using 0.")
                cov = 0

        # Point estimate of ratio
        theta_hat = coefficient / intercept

        # Fieller's method: Solve quadratic inequality
        # (intercept² - g²×se_intercept²)θ² - 2(intercept×coef - g²×cov)θ + (coef² - g²×se_coef²) ≤ 0

        a = intercept**2 - g_sq * se_intercept**2
        b = -(2 * intercept * coefficient - 2 * g_sq * cov)
        c = coefficient**2 - g_sq * se_coef**2

        # Check if we have a valid quadratic (a ≈ 0 means denominator uncertainty dominates)
        if abs(a) < 1e-10:
            # Degenerate case: very large uncertainty in denominator
            # This can happen when g²×se_intercept² ≈ intercept²
            # In this case, the interval may be unbounded or the entire real line
            self._logger.warning(
                "Fieller CI is degenerate (denominator uncertainty too large). "
                "Consider increasing sample size or using absolute effects."
            )
            return (np.nan, np.nan)

        # Solve quadratic equation: aθ² + bθ + c = 0
        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            # No real solutions - this shouldn't happen for valid CIs
            # but can occur with numerical issues
            self._logger.warning("Fieller CI has no real solution (negative discriminant).")
            return (np.nan, np.nan)

        sqrt_disc = np.sqrt(discriminant)

        if a > 0:
            # Standard case: parabola opens upward, solutions give CI bounds
            theta_lower = (-b - sqrt_disc) / (2 * a)
            theta_upper = (-b + sqrt_disc) / (2 * a)
        else:
            # a < 0: parabola opens downward
            # The inequality is satisfied outside the roots
            # This means the CI is (−∞, theta1] ∪ [theta2, ∞)
            # This happens when denominator is very uncertain
            theta1 = (-b + sqrt_disc) / (2 * a)
            theta2 = (-b - sqrt_disc) / (2 * a)
            self._logger.warning(
                f"Fieller CI is unbounded (two disjoint intervals). "
                f"Point estimate {theta_hat:.4f} but CI cannot be properly bounded. "
                f"This indicates very high uncertainty in the control value."
            )
            # Return the finite interval containing the point estimate
            if theta_hat < theta1:
                return (np.nan, theta1)
            else:
                return (theta2, np.nan)

        return (theta_lower, theta_upper)

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
            - "treatment_units" (int): The number of units in the treatment group.
            - "control_units" (int): The number of units in the control group.
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
        relative_effect = coefficient / intercept if intercept != 0 else 0
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect
        se_intercept = results.bse["Intercept"]
        cov_matrix = results.cov_params()
        rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
            coefficient, intercept, standard_error, se_intercept, cov_matrix, self._alpha
        )

        return {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum(),
            "control_units": data[self._treatment_col].count() - data[self._treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
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
            - "treatment_units" (int): The number of units in the treatment group.
            - "control_units" (int): The number of units in the control group.
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
        relative_effect = coefficient / abs(intercept) if intercept != 0 else 0
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect
        se_intercept = results.bse["Intercept"]
        cov_matrix = results.cov_params()
        rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
            coefficient, abs(intercept), standard_error, se_intercept, cov_matrix, self._alpha
        )

        return {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum().astype(int),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
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
            - "treatment_units" (int): The number of units in the treatment group.
            - "control_units" (int): The number of units in the control group.
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
        relative_effect = coefficient / intercept if intercept != 0 else 0
        standard_error = results.std_errors[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect
        se_intercept = results.std_errors["Intercept"]
        cov_matrix = results.cov
        rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
            coefficient, intercept, standard_error, se_intercept, cov_matrix, self._alpha
        )

        return {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum().astype(int),
            "control_units": (data[self._treatment_col].count() - data[self._treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
        }

    def ipw_logistic(
        self, data: pd.DataFrame, covariates: list[str], penalty: str = "l2", C: float = 1.0, max_iter: int = 5000
    ) -> pd.DataFrame:
        """
        Balance method: Estimate weights using logistic regression (ps-logistic).

        Parameters
        ----------
        data : pd.DataFrame
            Data to estimate the weights from
        covariates : List[str]
            List of covariates to include in the estimation
        penalty : str, optional
            Regularization penalty to use in the logistic regression model, by default 'l2'
        C : float, optional
            Inverse of regularization strength, by default 1.0
        max_iter : int, optional
            Maximum number of iterations, by default 5000

        Returns
        -------
        pd.DataFrame
            Data with the estimated weights
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
        logistic_model.fit(X, y)

        if not logistic_model.n_iter_[0] < logistic_model.max_iter:
            self._logger.warning(
                "Logistic regression model did not converge. Consider increasing the number of iterations or adjusting other parameters."  # noqa: E501
            )  # noqa: E501

        data["propensity_score"] = logistic_model.predict_proba(X)[:, 1]
        data["propensity_score"] = np.minimum(self._max_ps_score, data["propensity_score"])
        data["propensity_score"] = np.maximum(self._min_ps_score, data["propensity_score"])

        data = self.__calculate_stabilized_weights(data)
        return data

    def ipw_xgboost(self, data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        """
        Balance method: Estimate weights using XGBoost (ps-xgboost).

        Parameters
        ----------
        data : pd.DataFrame
            Data to estimate the weights from
        covariates : List[str]
            List of covariates to include in the estimation

        Returns
        -------
        pd.DataFrame
            Data with the estimated weights
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

    def entropy_balance(self, data: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
        """
        Balance method: Perform entropy balancing to create weights ('entropy').

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
            The original DataFrame with an added weights column.
        """
        if not covariates:
            log_and_raise_error(self._logger, "Covariates must be specified for entropy balancing.")

        treatment_indicator = data[self._treatment_col]
        covariate_data = data[covariates]

        eb = EntropyBalance()
        weights = eb.fit(covariate_data, treatment_indicator, estimand=self._target_effect)
        data[self._target_weights[self._target_effect]] = weights

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
