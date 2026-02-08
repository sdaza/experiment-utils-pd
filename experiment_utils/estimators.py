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
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        store_model: bool = False,
        compute_relative_ci: bool = True,
        **kwargs,
    ) -> dict[str, str | int | float]:  # noqa: E501
        """
        Perform linear regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.
        cluster_col (str, optional): Column for clustered standard errors (not supported for OLS yet).
        store_model (bool, optional): Whether to store the fitted model object.
        **kwargs: Additional arguments (ignored for compatibility).

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

        # TODO: Add clustered SE support for OLS
        if cluster_col:
            self._logger.warning(
                "Clustered standard errors not yet implemented for OLS. "
                "Using heteroskedasticity-robust (HC3) standard errors instead."
            )

        results = model.fit(cov_type="HC3")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept if intercept != 0 else 0
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect (skip during bootstrap)
        if compute_relative_ci:
            se_intercept = results.bse["Intercept"]
            cov_matrix = results.cov_params()

            # Extract covariance between treatment coefficient and intercept for future use
            try:
                cov_coef_intercept = cov_matrix.loc[self._treatment_col, "Intercept"]
            except (KeyError, AttributeError):
                cov_coef_intercept = 0.0

            rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
                coefficient, intercept, standard_error, se_intercept, cov_matrix, self._alpha
            )
        else:
            # During bootstrap, skip Fieller CI computation (will use percentiles later)
            rel_effect_lower = np.nan
            rel_effect_upper = np.nan
            se_intercept = np.nan
            cov_coef_intercept = np.nan

        output = {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum(),
            "control_units": data[self._treatment_col].count() - data[self._treatment_col].sum(),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
            "se_intercept": se_intercept,
            "cov_coef_intercept": cov_coef_intercept,
            "model_type": "ols",
            "effect_type": "mean_difference",
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def weighted_least_squares(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        weight_column: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        store_model: bool = False,
        compute_relative_ci: bool = True,
        **kwargs,
    ) -> dict[str, str | int | float]:
        """
        Perform weighted least squares regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        weight_column (str): The name of the column containing the weights for the regression.
        covariates (List[str]): The list of covariates to include in the regression model.
        cluster_col (str, optional): Column for clustered standard errors (not supported for WLS yet).
        store_model (bool, optional): Whether to store the fitted model object.
        **kwargs: Additional arguments (ignored for compatibility).

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

        # TODO: Add clustered SE support for WLS
        if cluster_col:
            self._logger.warning(
                "Clustered standard errors not yet implemented for WLS. "
                "Using heteroskedasticity-robust (HC3) standard errors instead."
            )

        results = model.fit(cov_type="HC3")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / abs(intercept) if intercept != 0 else 0
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect (skip during bootstrap)
        if compute_relative_ci:
            se_intercept = results.bse["Intercept"]
            cov_matrix = results.cov_params()

            # Extract covariance between treatment coefficient and intercept for future use
            try:
                cov_coef_intercept = cov_matrix.loc[self._treatment_col, "Intercept"]
            except (KeyError, AttributeError):
                cov_coef_intercept = 0.0

            rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
                coefficient, abs(intercept), standard_error, se_intercept, cov_matrix, self._alpha
            )
        else:
            # During bootstrap, skip Fieller CI computation (will use percentiles later)
            rel_effect_lower = np.nan
            rel_effect_upper = np.nan
            se_intercept = np.nan
            cov_coef_intercept = np.nan

        output = {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum().astype(int),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
            "se_intercept": se_intercept,
            "cov_coef_intercept": cov_coef_intercept,
            "model_type": "ols",
            "effect_type": "mean_difference",
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def iv_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        store_model: bool = False,
        compute_relative_ci: bool = True,
        **kwargs,
    ) -> dict[str, str | int | float]:  # noqa: E501
        """ "
        Perform instrumental variable regression on the given data.

        Parameters:
        data (pd.DataFrame): The input data containing the variables for the regression.
        outcome_variable (str): The name of the outcome variable to be predicted.
        covariates (List[str]): The list of covariates to include in the regression model.
        cluster_col (str, optional): Column for clustered standard errors (not supported for IV yet).
        store_model (bool, optional): Whether to store the fitted model object.
        **kwargs: Additional arguments (ignored for compatibility).

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

        # TODO: Add clustered SE support for IV
        if cluster_col:
            self._logger.warning(
                "Clustered standard errors not yet implemented for IV regression. "
                "Using heteroskedasticity-robust standard errors instead."
            )

        results = model.fit(cov_type="robust")

        coefficient = results.params[self._treatment_col]
        intercept = results.params["Intercept"]
        relative_effect = coefficient / intercept if intercept != 0 else 0
        standard_error = results.std_errors[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Compute Fieller CI for relative effect (skip during bootstrap)
        if compute_relative_ci:
            se_intercept = results.std_errors["Intercept"]
            cov_matrix = results.cov

            # Extract covariance between treatment coefficient and intercept for future use
            try:
                if isinstance(cov_matrix, np.ndarray):
                    # For IV regression, cov is numpy array - assume [Intercept, treatment_col, ...] order
                    cov_coef_intercept = cov_matrix[0, 1] if cov_matrix.shape[0] > 1 else 0.0
                else:
                    cov_coef_intercept = cov_matrix.loc[self._treatment_col, "Intercept"]
            except (KeyError, AttributeError, IndexError):
                cov_coef_intercept = 0.0

            rel_effect_lower, rel_effect_upper = self._compute_fieller_ci(
                coefficient, intercept, standard_error, se_intercept, cov_matrix, self._alpha
            )
        else:
            # During bootstrap, skip Fieller CI computation (will use percentiles later)
            rel_effect_lower = np.nan
            rel_effect_upper = np.nan
            se_intercept = np.nan
            cov_coef_intercept = np.nan

        output = {
            "outcome": outcome_variable,
            "treatment_units": data[self._treatment_col].sum().astype(int),
            "control_units": (data[self._treatment_col].count() - data[self._treatment_col].sum()).astype(int),
            "control_value": intercept,
            "treatment_value": intercept + coefficient,
            "absolute_effect": coefficient,
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "relative_effect": relative_effect,
            "rel_effect_lower": rel_effect_lower,
            "rel_effect_upper": rel_effect_upper,
            "se_intercept": se_intercept,
            "cov_coef_intercept": cov_coef_intercept,
            "model_type": "ols",
            "effect_type": "mean_difference",
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def logistic_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform logistic regression with optional marginal effects and clustered standard errors.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Binary outcome variable (0/1)
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Control marginal effects computation (default: "overall")
            - "overall": Average Marginal Effect (AME)
            - "mean": Marginal Effect at the Mean (MEM)
            - "median": Marginal effect at median
            - True: Same as "overall"
            - False: Return odds ratios instead
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        formula = self.__create_formula(outcome_variable, covariates)
        model = smf.logit(formula, data=data)

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, disp=False)
        else:
            results = model.fit(cov_type="HC3", disp=False)

        # Extract treatment effect on logit scale
        coefficient = results.params[self._treatment_col]
        odds_ratio = np.exp(coefficient)
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects (change in probability)
        if compute_marginal_effects:
            try:
                # Determine at value based on method
                if compute_marginal_effects == "overall":
                    at_value = "overall"  # AME - average marginal effect
                elif compute_marginal_effects == "mean":
                    at_value = "mean"  # MEM - marginal effect at mean
                elif compute_marginal_effects == "median":
                    at_value = "median"  # Marginal effect at median
                else:
                    at_value = "overall"  # Default

                marginal_effects = results.get_margeff(at=at_value, method="dydx")

                # Get all info from summary_frame (has dy/dx, std err, z, P>|z|, etc.)
                me_summary = marginal_effects.summary_frame()

                if self._treatment_col in me_summary.index:
                    me_treatment = me_summary.loc[self._treatment_col, "dy/dx"]
                    me_se = me_summary.loc[self._treatment_col, "Std. Err."]
                    me_pvalue = me_summary.loc[self._treatment_col, "Pr(>|z|)"]
                else:
                    raise KeyError(f"Treatment column '{self._treatment_col}' not found in marginal effects")

                # Predicted probabilities for interpretation
                # Control: treatment = 0
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                prob_control = results.predict(data_control).mean()

                # Treatment: treatment = 1
                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                prob_treatment = results.predict(data_treatment).mean()

                control_value = prob_control
                treatment_value = prob_treatment
                absolute_effect = me_treatment  # Change in probability (percentage points)
                relative_effect = absolute_effect / prob_control if prob_control > 0 else 0
                final_se = me_se
                final_pvalue = me_pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log-odds scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            # Fall back to log-odds scale
            control_value = results.params["Intercept"]
            treatment_value = control_value + coefficient
            absolute_effect = coefficient
            relative_effect = odds_ratio - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "odds_ratio": odds_ratio,
            "log_odds": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "logistic",
            "effect_type": "probability_change" if compute_marginal_effects else "log_odds",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def weighted_logistic_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        weight_column: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform weighted logistic regression with IPW weights.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Binary outcome variable (0/1)
        weight_column : str
            Column containing IPW weights
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Control marginal effects computation (default: "overall")
            - Any truthy value: Compute marginal effects
            - False: Return odds ratios
            Note: Weighted models use manual computation (equivalent to "overall")
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        import statsmodels.api as sm

        formula = self.__create_formula(outcome_variable, covariates)
        model = smf.glm(formula, data=data, family=sm.families.Binomial(), freq_weights=data[weight_column])

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        else:
            results = model.fit(cov_type="HC3")

        # Extract treatment effect
        coefficient = results.params[self._treatment_col]
        odds_ratio = np.exp(coefficient)
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects
        if compute_marginal_effects:
            try:
                # For weighted GLM, manually compute marginal effects
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                prob_control = results.predict(data_control).mean()

                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                prob_treatment = results.predict(data_treatment).mean()

                me_treatment = prob_treatment - prob_control
                control_value = prob_control
                treatment_value = prob_treatment
                absolute_effect = me_treatment
                relative_effect = absolute_effect / prob_control if prob_control > 0 else 0

                # Use delta method for SE approximation
                final_se = standard_error * prob_control * (1 - prob_control)
                final_pvalue = pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log-odds scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            control_value = results.params["Intercept"]
            treatment_value = control_value + coefficient
            absolute_effect = coefficient
            relative_effect = odds_ratio - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "odds_ratio": odds_ratio,
            "log_odds": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "logistic",
            "effect_type": "probability_change" if compute_marginal_effects else "log_odds",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def poisson_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform Poisson regression for count outcomes with optional marginal effects.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Count outcome variable (non-negative integers)
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Control marginal effects computation (default: "overall")
            - "overall": Average Marginal Effect (AME)
            - "mean": Marginal Effect at the Mean (MEM)
            - "median": Marginal effect at median
            - True: Same as "overall"
            - False: Return rate ratios
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        formula = self.__create_formula(outcome_variable, covariates)
        model = smf.poisson(formula, data=data)

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, disp=False)
        else:
            results = model.fit(cov_type="HC3", disp=False)

        # Extract treatment effect on log scale
        coefficient = results.params[self._treatment_col]
        irr = np.exp(coefficient)  # Incidence rate ratio
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects (change in expected count)
        if compute_marginal_effects:
            try:
                # Determine at value based on method
                if compute_marginal_effects == "overall":
                    at_value = "overall"  # AME
                elif compute_marginal_effects == "mean":
                    at_value = "mean"  # MEM
                elif compute_marginal_effects == "median":
                    at_value = "median"
                else:
                    at_value = "overall"  # Default

                marginal_effects = results.get_margeff(at=at_value, method="dydx")

                # Get all info from summary_frame
                me_summary = marginal_effects.summary_frame()

                if self._treatment_col in me_summary.index:
                    me_treatment = me_summary.loc[self._treatment_col, "dy/dx"]
                    me_se = me_summary.loc[self._treatment_col, "Std. Err."]
                    me_pvalue = me_summary.loc[self._treatment_col, "Pr(>|z|)"]
                else:
                    raise KeyError(f"Treatment column '{self._treatment_col}' not found in marginal effects")

                # Predicted counts
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                count_control = results.predict(data_control).mean()

                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                count_treatment = results.predict(data_treatment).mean()

                control_value = count_control
                treatment_value = count_treatment
                absolute_effect = me_treatment  # Change in expected count
                relative_effect = absolute_effect / count_control if count_control > 0 else 0
                final_se = me_se
                final_pvalue = me_pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            # Fall back to log scale
            control_value = np.exp(results.params["Intercept"])
            treatment_value = control_value * irr
            absolute_effect = coefficient
            relative_effect = irr - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "incidence_rate_ratio": irr,
            "log_irr": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "poisson",
            "effect_type": "count_change" if compute_marginal_effects else "log_rate_ratio",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def weighted_poisson_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        weight_column: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform weighted Poisson regression with IPW weights.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Count outcome variable
        weight_column : str
            Column containing IPW weights
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Control marginal effects computation (default: "overall")
            Note: Weighted models use manual computation
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        import statsmodels.api as sm

        formula = self.__create_formula(outcome_variable, covariates)
        model = smf.glm(formula, data=data, family=sm.families.Poisson(), freq_weights=data[weight_column])

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        else:
            results = model.fit(cov_type="HC3")

        # Extract treatment effect
        coefficient = results.params[self._treatment_col]
        irr = np.exp(coefficient)
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects
        if compute_marginal_effects:
            try:
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                count_control = results.predict(data_control).mean()

                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                count_treatment = results.predict(data_treatment).mean()

                me_treatment = count_treatment - count_control
                control_value = count_control
                treatment_value = count_treatment
                absolute_effect = me_treatment
                relative_effect = absolute_effect / count_control if count_control > 0 else 0
                final_se = standard_error * count_control
                final_pvalue = pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            control_value = np.exp(results.params["Intercept"])
            treatment_value = control_value * irr
            absolute_effect = coefficient
            relative_effect = irr - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "incidence_rate_ratio": irr,
            "log_irr": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "poisson",
            "effect_type": "count_change" if compute_marginal_effects else "log_rate_ratio",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def negative_binomial_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform Negative Binomial regression for overdispersed count outcomes.

        Use this instead of Poisson when variance > mean (overdispersion).
        Common in real-world count data (orders, page views, etc.)

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Count outcome variable (non-negative integers)
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Control marginal effects computation (default: "overall")
            - "overall": Average Marginal Effect (AME)
            - "mean": Marginal Effect at the Mean (MEM)
            - "median": Marginal effect at median
            - True: Same as "overall"
            - False: Return rate ratios
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        formula = self.__create_formula(outcome_variable, covariates)
        model = smf.negativebinomial(formula, data=data)

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]}, disp=False)
        else:
            results = model.fit(cov_type="HC3", disp=False)

        # Extract treatment effect on log scale
        coefficient = results.params[self._treatment_col]
        irr = np.exp(coefficient)  # Incidence rate ratio
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects (change in expected count)
        if compute_marginal_effects:
            try:
                # Determine at value based on method
                if compute_marginal_effects == "overall":
                    at_value = "overall"  # AME
                elif compute_marginal_effects == "mean":
                    at_value = "mean"  # MEM
                elif compute_marginal_effects == "median":
                    at_value = "median"
                else:
                    at_value = "overall"  # Default

                marginal_effects = results.get_margeff(at=at_value, method="dydx")

                # Get all info from summary_frame
                me_summary = marginal_effects.summary_frame()

                if self._treatment_col in me_summary.index:
                    me_treatment = me_summary.loc[self._treatment_col, "dy/dx"]
                    me_se = me_summary.loc[self._treatment_col, "Std. Err."]
                    me_pvalue = me_summary.loc[self._treatment_col, "Pr(>|z|)"]
                else:
                    raise KeyError(f"Treatment column '{self._treatment_col}' not found in marginal effects")

                # Predicted counts
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                count_control = results.predict(data_control).mean()

                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                count_treatment = results.predict(data_treatment).mean()

                control_value = count_control
                treatment_value = count_treatment
                absolute_effect = me_treatment  # Change in expected count
                relative_effect = absolute_effect / count_control if count_control > 0 else 0
                final_se = me_se
                final_pvalue = me_pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            # Fall back to log scale
            control_value = np.exp(results.params["Intercept"])
            treatment_value = control_value * irr
            absolute_effect = coefficient
            relative_effect = irr - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "incidence_rate_ratio": irr,
            "log_irr": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "negative_binomial",
            "effect_type": "count_change" if compute_marginal_effects else "log_rate_ratio",
            "alpha_param": results.params.get("alpha", np.nan),  # Dispersion parameter
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def weighted_negative_binomial_regression(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        weight_column: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        compute_marginal_effects: str | bool = "overall",
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform weighted Negative Binomial regression with IPW weights.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Count outcome variable
        weight_column : str
            Column containing IPW weights
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        compute_marginal_effects : str | bool, optional
            Whether to compute marginal effects (default: True)
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        import statsmodels.api as sm

        formula = self.__create_formula(outcome_variable, covariates)

        # For weighted NB, use GLM with NegativeBinomial family
        model = smf.glm(formula, data=data, family=sm.families.NegativeBinomial(), freq_weights=data[weight_column])

        if cluster_col:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": data[cluster_col]})
        else:
            results = model.fit(cov_type="HC3")

        # Extract treatment effect
        coefficient = results.params[self._treatment_col]
        irr = np.exp(coefficient)
        standard_error = results.bse[self._treatment_col]
        pvalue = results.pvalues[self._treatment_col]

        # Convert True to "overall" for backward compatibility
        if compute_marginal_effects is True:
            compute_marginal_effects = "overall"

        # Compute marginal effects
        if compute_marginal_effects:
            try:
                data_control = data.copy()
                data_control[self._treatment_col] = 0
                count_control = results.predict(data_control).mean()

                data_treatment = data.copy()
                data_treatment[self._treatment_col] = 1
                count_treatment = results.predict(data_treatment).mean()

                me_treatment = count_treatment - count_control
                control_value = count_control
                treatment_value = count_treatment
                absolute_effect = me_treatment
                relative_effect = absolute_effect / count_control if count_control > 0 else 0
                final_se = standard_error * count_control
                final_pvalue = pvalue
            except Exception as e:
                self._logger.warning(f"Could not compute marginal effects: {e}. Using log scale.")
                compute_marginal_effects = False

        if not compute_marginal_effects:
            control_value = np.exp(results.params["Intercept"])
            treatment_value = control_value * irr
            absolute_effect = coefficient
            relative_effect = irr - 1
            final_se = standard_error
            final_pvalue = pvalue

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "control_value": control_value,
            "treatment_value": treatment_value,
            "absolute_effect": absolute_effect,
            "relative_effect": relative_effect,
            "standard_error": final_se,
            "pvalue": final_pvalue,
            "stat_significance": 1 if final_pvalue < self._alpha else 0,
            "incidence_rate_ratio": irr,
            "log_irr": coefficient,
            "marginal_effect": absolute_effect if compute_marginal_effects else None,
            "model_type": "negative_binomial",
            "effect_type": "count_change" if compute_marginal_effects else "log_rate_ratio",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = results

        return output

    def cox_proportional_hazards(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        event_col: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform Cox proportional hazards regression for survival analysis.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Duration/time column
        event_col : str
            Event indicator column (1=event, 0=censored)
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError as e:
            import sys

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            error_msg = (
                f"lifelines package is required for Cox regression but could not be imported.\n"
                f"Python version: {python_version}\n"
                f"Error: {e}\n\n"
                f"Installation instructions:\n"
                f"  - Python 3.11+: pip install 'lifelines>=0.29.0'\n"
                f"  - Python 3.10: pip install 'lifelines>=0.27.0,<0.30.0'\n"
                f"  - Or install with package extras: pip install experiment-utils-pd[survival]"
            )
            log_and_raise_error(self._logger, error_msg)

        cph = CoxPHFitter()

        # Prepare data for lifelines
        if covariates:
            formula_cols = [self._treatment_col] + [f"z_{c}" for c in covariates]
        else:
            formula_cols = [self._treatment_col]

        # Select columns for modeling
        model_data = data[[outcome_variable, event_col, self._treatment_col]].copy()
        if covariates:
            for cov in covariates:
                model_data[f"z_{cov}"] = data[f"z_{cov}"]

        # Fit model
        try:
            cph.fit(
                model_data,
                duration_col=outcome_variable,
                event_col=event_col,
                formula=" + ".join(formula_cols),
                cluster_col=cluster_col if cluster_col else None,
                robust=True if cluster_col else False,
            )
        except Exception as e:
            log_and_raise_error(self._logger, f"Cox regression failed: {e}")

        # Extract treatment effect (log hazard ratio)
        coefficient = cph.params_[self._treatment_col]
        hr = np.exp(coefficient)  # Hazard ratio
        standard_error = cph.standard_errors_[self._treatment_col]
        pvalue = cph.summary.loc[self._treatment_col, "p"]

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "absolute_effect": coefficient,  # Log hazard ratio
            "relative_effect": hr,  # Hazard ratio itself
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "hazard_ratio": hr,
            "log_hazard_ratio": coefficient,
            "marginal_effect": None,
            "model_type": "cox",
            "effect_type": "hazard_ratio",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = cph

        return output

    def weighted_cox_proportional_hazards(
        self,
        data: pd.DataFrame,
        outcome_variable: str,
        event_col: str,
        weight_column: str,
        covariates: list[str] | None = None,
        cluster_col: str | None = None,
        store_model: bool = False,
        compute_relative_ci: bool = True,
    ) -> dict[str, str | int | float]:
        """
        Perform Cox PH regression with IPW weights for balance adjustment.

        Parameters
        ----------
        data : pd.DataFrame
            Input data
        outcome_variable : str
            Duration/time column
        event_col : str
            Event indicator column (1=event, 0=censored)
        weight_column : str
            Column containing IPW weights
        covariates : list[str], optional
            Covariates to include
        cluster_col : str, optional
            Column for clustered standard errors
        store_model : bool, optional
            Whether to store the fitted model object (default: False)

        Returns
        -------
        dict
            Results dictionary with treatment effects, p-values, etc.
        """
        try:
            from lifelines import CoxPHFitter
        except ImportError as e:
            import sys

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            error_msg = (
                f"lifelines package is required for Cox regression but could not be imported.\n"
                f"Python version: {python_version}\n"
                f"Error: {e}\n\n"
                f"Installation instructions:\n"
                f"  - Python 3.11+: pip install 'lifelines>=0.29.0'\n"
                f"  - Python 3.10: pip install 'lifelines>=0.27.0,<0.30.0'\n"
                f"  - Or install with package extras: pip install experiment-utils-pd[survival]"
            )
            log_and_raise_error(self._logger, error_msg)

        cph = CoxPHFitter()

        # Prepare data
        if covariates:
            formula_cols = [self._treatment_col] + [f"z_{c}" for c in covariates]
        else:
            formula_cols = [self._treatment_col]

        model_data = data[[outcome_variable, event_col, self._treatment_col, weight_column]].copy()
        if covariates:
            for cov in covariates:
                model_data[f"z_{cov}"] = data[f"z_{cov}"]

        # Fit model with weights and clustering
        try:
            cph.fit(
                model_data,
                duration_col=outcome_variable,
                event_col=event_col,
                formula=" + ".join(formula_cols),
                weights_col=weight_column,  # IPW weights
                cluster_col=cluster_col if cluster_col else None,
                robust=True,  # Recommended with sampling weights
            )
        except Exception as e:
            log_and_raise_error(self._logger, f"Weighted Cox regression failed: {e}")

        # Extract treatment effect
        coefficient = cph.params_[self._treatment_col]
        hr = np.exp(coefficient)
        standard_error = cph.standard_errors_[self._treatment_col]
        pvalue = cph.summary.loc[self._treatment_col, "p"]

        output = {
            "outcome": outcome_variable,
            "treatment_units": int(data[self._treatment_col].sum()),
            "control_units": int(data[self._treatment_col].count() - data[self._treatment_col].sum()),
            "absolute_effect": coefficient,  # Log hazard ratio
            "relative_effect": hr,  # Hazard ratio itself
            "standard_error": standard_error,
            "pvalue": pvalue,
            "stat_significance": 1 if pvalue < self._alpha else 0,
            "hazard_ratio": hr,
            "log_hazard_ratio": coefficient,
            "marginal_effect": None,
            "model_type": "cox",
            "effect_type": "hazard_ratio",
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "se_intercept": np.nan,
            "cov_coef_intercept": np.nan,
        }

        if store_model:
            output["fitted_model"] = cph

        return output

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
