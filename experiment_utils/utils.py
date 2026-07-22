"""
Collection of helper methods. These should be fully generic and make no
assumptions about the format of input data.
"""

import logging
import warnings
from contextlib import contextmanager
from functools import cache
from math import sqrt

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist


@contextmanager
def suppress_fit_warnings():
    """Silence benign, noisy warnings emitted while fitting a model.

    Two families are suppressed:

    - numpy's ``"encountered in matmul"`` RuntimeWarnings. A (quasi-)separated
      logistic fit or a near-singular fixed-effects design makes the solver
      produce huge coefficients, so the ``X @ beta`` step overflows and numpy
      emits divide-by-zero/overflow/invalid warnings from inside matmul (e.g.
      sklearn's ``_linear_loss`` or pyfixest's residual computation).
    - pyfixest's ``"singleton fixed effect(s) dropped"`` UserWarning. Singleton
      FE groups (one observation) are absorbed by their own dummy and contribute
      nothing to identification, so pyfixest drops them; the effective sample
      size is reported in the estimator output instead.

    The fitted results are still returned and validated/NaN-guarded downstream,
    so wrap only the fit/predict call to keep these out of callers' logs without
    hiding genuine warnings elsewhere.
    """
    with warnings.catch_warnings(), np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul")
        warnings.filterwarnings("ignore", category=UserWarning, message=".*singleton fixed effect.*")
        yield


def turn_off_package_logger(package: str) -> None:
    """ "
    Turn off logging for a specific package.

    :param package: The name of the package to turn off logging for.
    """
    logger = logging.getLogger(package)
    logger.setLevel(logging.ERROR)
    logger.handlers = [logging.NullHandler()]


def get_logger(name: str) -> logging.Logger:
    """ "
    Get a logger with the specified name.

    :param name: The name of the logger.
    :return: The logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[::-1]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def log_and_raise_error(logger: logging.Logger, message: str, exception_type: type[Exception] = ValueError) -> None:
    """ "
    Logs an error message and raises an exception of the specified type.

    :param message: The error message to log and raise.
    :param exception_type: The type of exception to raise (default is ValueError).
    """

    logger.error(message)
    raise exception_type(message)


def _check_assignment_balance(df, assignment_col, covariates, variant1, variant2, threshold=0.1):
    """
    Calculate standardized mean differences (SMD) between two assignment groups.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing assignments and covariates
    assignment_col : str
        Name of column containing assignment labels
    covariates : list[str]
        List of covariate column names to check balance for
    variant1 : str
        Name of first variant/group
    variant2 : str
        Name of second variant/group
    threshold : float
        SMD threshold for balance flag (default 0.1)

    Returns
    -------
    pd.DataFrame
        DataFrame with balance metrics (covariate, mean_var1, mean_var2, smd, balance_flag)
    """

    # Filter to the two groups being compared
    group1 = df[df[assignment_col] == variant1]
    group2 = df[df[assignment_col] == variant2]

    smd_results = []

    for cov in covariates:
        # Handle categorical covariates by one-hot encoding
        if pd.api.types.is_object_dtype(df[cov]) or isinstance(df[cov].dtype, pd.CategoricalDtype):
            # One-hot encode and calculate SMD for each dummy variable
            dummies = pd.get_dummies(df[cov], prefix=cov)
            for dummy_col in dummies.columns:
                dummy_group1 = dummies.loc[group1.index, dummy_col]
                dummy_group2 = dummies.loc[group2.index, dummy_col]

                mean1 = dummy_group1.mean()
                mean2 = dummy_group2.mean()

                var1 = dummy_group1.var()
                var2 = dummy_group2.var()

                pooled_std = np.sqrt((var1 + var2) / 2)
                smd = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

                balance_flag = 1 if abs(smd) <= threshold else 0

                smd_results.append(
                    {
                        "covariate": dummy_col,
                        f"mean_{variant1}": mean1,
                        f"mean_{variant2}": mean2,
                        "smd": smd,
                        "balance_flag": balance_flag,
                    }
                )
        else:
            # Numeric covariate
            mean1 = group1[cov].mean()
            mean2 = group2[cov].mean()

            var1 = group1[cov].var()
            var2 = group2[cov].var()

            pooled_std = np.sqrt((var1 + var2) / 2)
            smd = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

            balance_flag = 1 if abs(smd) <= threshold else 0

            smd_results.append(
                {
                    "covariate": cov,
                    f"mean_{variant1}": mean1,
                    f"mean_{variant2}": mean2,
                    "smd": smd,
                    "balance_flag": balance_flag,
                }
            )

    return pd.DataFrame(smd_results)


def generate_comparison_pairs(
    data: pd.DataFrame,
    treatment_col: str,
    treatment_comparisons: list[tuple] | None = None,
    logger: logging.Logger | None = None,
) -> list[tuple]:
    """
    Generate comparison pairs for treatment analysis.

    This is a general-purpose function used by both check_covariate_balance
    and ExperimentAnalyzer to determine which treatment groups to compare.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing treatment column
    treatment_col : str
        Name of treatment column
    treatment_comparisons : list[tuple], optional
        List of (treatment, control) tuples to compare. If None, auto-generate.
    logger : logging.Logger, optional
        Logger for warnings. If None, creates one.

    Returns
    -------
    list[tuple]
        List of (treatment_val, control_val) tuples to compare

    Examples
    --------
    >>> # Binary treatment - auto-generates (1, 0)
    >>> pairs = generate_comparison_pairs(df, "treatment")

    >>> # Explicit comparisons for categorical treatment
    >>> pairs = generate_comparison_pairs(
    ...     df, "treatment",
    ...     treatment_comparisons=[("variant_1", "control"), ("variant_2", "control")]
    ... )
    """
    import itertools

    if logger is None:
        logger = get_logger("ComparisonPairs")

    treatment_values = set(data[treatment_col].unique())

    if treatment_comparisons is not None:
        # Validate provided comparisons
        valid_pairs = []
        for treatment_val, control_val in treatment_comparisons:
            if treatment_val in treatment_values and control_val in treatment_values:
                valid_pairs.append((treatment_val, control_val))
            else:
                logger.warning(
                    f"Skipping comparison ({treatment_val}, {control_val}) - one or both groups not found in data. "
                    f"Available values: {sorted(treatment_values)}"
                )
        return valid_pairs

    # Auto-generate comparison pairs
    if len(treatment_values) < 2:
        logger.warning(
            f"Cannot generate comparison pairs - only {len(treatment_values)} treatment group(s) found. "
            f"Available values: {sorted(treatment_values)}"
        )
        return []

    # Special case: binary treatment {0, 1}
    if treatment_values == {0, 1}:
        return [(1, 0)]

    # General case: all pairwise combinations (higher value as treatment)
    sorted_values = sorted(list(treatment_values))
    pairs = list(itertools.combinations(sorted_values, 2))
    return [(b, a) for a, b in pairs]


def detect_categorical_covariates(
    data: pd.DataFrame,
    covariates: list[str],
    categorical_max_unique: int = 10,
) -> dict[str, list]:
    """
    Detect categorical covariates and return their unique categories.

    This is a general-purpose function used by both check_covariate_balance
    and ExperimentAnalyzer for consistent categorical variable detection.

    Detection rules:
    - Object or category dtype columns are always treated as categorical
    - Numeric columns (int or float) with 2 to categorical_max_unique unique values
      are treated as categorical, EXCEPT natural binary {0, 1} columns

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing covariates
    covariates : list[str]
        List of covariate names to check
    categorical_max_unique : int, optional
        Maximum unique values to consider a numeric variable categorical (default 10)

    Returns
    -------
    dict[str, list]
        Dictionary mapping covariate names to their sorted list of unique categories

    Examples
    --------
    >>> categorical_info = detect_categorical_covariates(
    ...     df, ["region", "tier", "age"], categorical_max_unique=10
    ... )
    >>> # Returns: {"region": ["North", "South", ...], "tier": [1, 2, 3]}
    """
    categorical_info = {}

    for cov in covariates:
        if cov not in data.columns:
            continue

        is_object = pd.api.types.is_object_dtype(data[cov]) or isinstance(data[cov].dtype, pd.CategoricalDtype)
        n_unique = data[cov].nunique()
        is_natural_binary = n_unique == 2 and set(data[cov].dropna().unique()) <= {0, 1}
        is_low_cardinality_numeric = (
            pd.api.types.is_numeric_dtype(data[cov])
            and 2 <= n_unique <= categorical_max_unique
            and not is_natural_binary
        )

        if is_object or is_low_cardinality_numeric:
            categories = sorted(data[cov].dropna().unique())
            categorical_info[cov] = categories

    return categorical_info


def _compute_balance_for_pair(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    categorical_covariates: dict[str, list] | None,
    min_binary_count: int,
    threshold: float,
    treatment_value: int | str,
    control_value: int | str,
    categorical_max_unique: int,
    weights_col: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Compute balance metrics for a single treatment vs control comparison.

    This is the core balance checking logic extracted for reuse.
    Data should already be filtered to only include the two groups being compared.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame filtered to contain only treatment and control groups
    treatment_col : str
        Name of treatment column
    covariates : list[str]
        List of covariate names to check
    categorical_covariates : dict[str, list] | None
        Dict mapping covariate names to their categories
    min_binary_count : int
        Minimum count for binary covariates
    threshold : float
        SMD threshold for balance flag
    treatment_value : int | str
        Value indicating treatment group in this comparison
    control_value : int | str
        Value indicating control group in this comparison
    categorical_max_unique : int
        Maximum unique values to consider a variable categorical
    weights_col : str
        Name of weights column
    logger : logging.Logger
        Logger for warnings

    Returns
    -------
    pd.DataFrame
        Balance metrics for this comparison
    """
    import re

    # Make a copy to avoid modifying original data
    data = data.copy()

    # Recode treatment to binary
    data[treatment_col] = (data[treatment_col] == treatment_value).astype(int)

    # Add weights column if not present
    if weights_col not in data.columns:
        data[weights_col] = 1

    # Identify categorical covariates if not provided
    if categorical_covariates is None:
        categorical_covariates = detect_categorical_covariates(
            data=data,
            covariates=covariates,
            categorical_max_unique=categorical_max_unique,
        )

    # Separate numeric/binary from categorical
    categorical_cov_names = set(categorical_covariates.keys())
    numeric_binary_covariates = [c for c in covariates if c not in categorical_cov_names and c in data.columns]

    # Identify numeric vs binary among non-categorical
    numeric_covariates = []
    binary_covariates = []

    for cov in numeric_binary_covariates:
        unique_vals = data[cov].dropna().unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
            binary_covariates.append(cov)
        else:
            numeric_covariates.append(cov)

    # Impute missing values
    for cov in numeric_covariates:
        if data[cov].isna().any():
            data[cov] = data[cov].fillna(data[cov].mean())

    for cov in binary_covariates:
        if data[cov].isna().any():
            if not data[cov].mode().empty:
                data[cov] = data[cov].fillna(data[cov].mode()[0])

    # Filter covariates based on variance and frequency
    filtered_numeric = [c for c in numeric_covariates if data[c].std(ddof=0) != 0]
    filtered_binary = [c for c in binary_covariates if data[c].sum() >= min_binary_count]
    filtered_binary = [c for c in filtered_binary if data[c].std(ddof=0) != 0]

    final_numeric_binary = filtered_numeric + filtered_binary

    # Log removed covariates
    removed_numeric = set(numeric_covariates) - set(filtered_numeric)
    removed_binary_freq = [c for c in binary_covariates if data[c].sum() < min_binary_count]
    removed_binary_var = [c for c in binary_covariates if c not in removed_binary_freq and data[c].std(ddof=0) == 0]

    if removed_numeric or removed_binary_freq or removed_binary_var:
        logger.warning("Removed covariates:")
        if removed_numeric:
            logger.warning(f"  - Zero variance (numeric): {sorted(removed_numeric)}")
        if removed_binary_freq:
            logger.warning(f"  - Low frequency (< {min_binary_count}): {sorted(removed_binary_freq)}")
        if removed_binary_var:
            logger.warning(f"  - Zero variance (binary): {sorted(removed_binary_var)}")

    # Handle categorical covariates - create dummies
    balance_covariates = []

    if categorical_covariates:

        def _clean_category_name(cat):
            """Convert category to lowercase and replace spaces/special chars with underscores"""
            cat_str = str(cat).lower()
            cat_str = re.sub(r"[^\w]+", "_", cat_str)
            cat_str = cat_str.strip("_")
            return cat_str

        # Create dummy variables for categorical covariates (include all categories)
        for covariate, categories in categorical_covariates.items():
            if covariate not in data.columns:
                continue

            for cat in categories:
                cat_clean = _clean_category_name(cat)
                dummy_col = f"{covariate}_{cat_clean}"
                data[dummy_col] = (data[covariate] == cat).astype(int)

                # Apply same filtering as binary covariates
                # Also check that dummy appears in both treatment and control groups
                if data[dummy_col].sum() >= min_binary_count and data[dummy_col].std(ddof=0) != 0:
                    treated_sum = data[data[treatment_col] == 1][dummy_col].sum()
                    control_sum = data[data[treatment_col] == 0][dummy_col].sum()
                    if treated_sum > 0 and control_sum > 0:
                        balance_covariates.append(dummy_col)

    # Add filtered numeric/binary covariates
    balance_covariates.extend(final_numeric_binary)

    if not balance_covariates:
        logger.warning("No valid covariates remaining after filtering")
        return pd.DataFrame()

    # Standardize covariates
    for cov in balance_covariates:
        mean_val = data[cov].mean()
        std_val = data[cov].std()
        if std_val != 0:
            data[f"z_{cov}"] = (data[cov] - mean_val) / std_val
        else:
            data[f"z_{cov}"] = 0

    # Calculate SMD for balance covariates (using original scale for means, standardized for SMD)
    treated = data[data[treatment_col] == 1]
    control = data[data[treatment_col] == 0]

    smd_results = []
    for cov in balance_covariates:
        z_cov = f"z_{cov}"

        # Calculate weighted means on original scale
        mean_treated = np.average(treated[cov], weights=treated[weights_col])
        mean_control = np.average(control[cov], weights=control[weights_col])

        # Calculate weighted variance on standardized scale
        var_treated = np.average((treated[z_cov] - treated[z_cov].mean()) ** 2, weights=treated[weights_col])
        var_control = np.average((control[z_cov] - control[z_cov].mean()) ** 2, weights=control[weights_col])

        pooled_std = np.sqrt((var_treated + var_control) / 2)

        # SMD using standardized means
        z_mean_treated = np.average(treated[z_cov], weights=treated[weights_col])
        z_mean_control = np.average(control[z_cov], weights=control[weights_col])
        smd = (z_mean_treated - z_mean_control) / pooled_std if pooled_std != 0 else 0

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

    return pd.DataFrame(smd_results)


def check_covariate_balance(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    categorical_covariates: dict[str, list] | None = None,
    min_binary_count: int = 5,
    threshold: float = 0.1,
    treatment_comparisons: list[tuple] | None = None,
    categorical_max_unique: int = 10,
    weights_col: str = "weights",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Check covariate balance between treatment groups with full preprocessing.

    This function performs comprehensive balance checking including:
    - Categorical variable identification and dummy creation
    - Covariate filtering (zero variance, minimum counts)
    - Standardization
    - SMD calculation with balance flags
    - Support for multiple treatment comparisons

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing treatment and covariates
    treatment_col : str
        Name of treatment column
    covariates : list[str]
        List of covariate names to check
    categorical_covariates : dict[str, list], optional
        Dict mapping covariate names to their categories. If None, will auto-detect.
    min_binary_count : int, optional
        Minimum count for binary covariates (default 5)
    threshold : float, optional
        SMD threshold for balance flag (default 0.1)
    treatment_comparisons : list[tuple], optional
        List of (treatment_value, control_value) tuples specifying which groups to compare.
        If None, automatically generates all pairwise comparisons (or [(1, 0)] for binary).
        Example: [("variant_1", "control"), ("variant_2", "control")]
    categorical_max_unique : int, optional
        Maximum unique values to consider a variable categorical (default 10)
    weights_col : str, optional
        Name of weights column (default "weights")
    logger : logging.Logger, optional
        Logger for warnings. If None, creates one.

    Returns
    -------
    pd.DataFrame
        Balance metrics with columns:
        - covariate: str - covariate name
        - mean_treated: float - weighted mean for treatment group
        - mean_control: float - weighted mean for control group
        - smd: float - standardized mean difference
        - balance_flag: int - 1 if balanced, 0 if imbalanced
        - treatment_group: int | str - treatment group identifier
        - control_group: int | str - control group identifier

    Examples
    --------
    >>> # Explicit comparisons
    >>> balance = check_covariate_balance(
    ...     data=df,
    ...     treatment_col='assignment',
    ...     covariates=['age', 'income', 'region'],
    ...     treatment_comparisons=[("variant_1", "control"), ("variant_2", "control")],
    ...     threshold=0.1
    ... )

    >>> # Auto-generate all pairwise comparisons
    >>> balance = check_covariate_balance(
    ...     data=df,
    ...     treatment_col='treatment',
    ...     covariates=['age', 'income', 'region'],
    ...     threshold=0.1
    ... )
    """
    if logger is None:
        logger = get_logger("BalanceChecker")

    # Make a copy to avoid modifying original data
    data = data.copy()

    # Generate comparison pairs
    comparison_pairs = generate_comparison_pairs(data, treatment_col, treatment_comparisons, logger)

    if not comparison_pairs:
        logger.warning("No valid comparison pairs found")
        # Return empty DataFrame with proper column structure
        return pd.DataFrame(
            columns=[
                "covariate",
                "mean_treated",
                "mean_control",
                "smd",
                "balance_flag",
                "treatment_group",
                "control_group",
            ]
        )

    # Iterate through comparison pairs and compute balance for each
    balance_results = []

    for treatment_val, control_val in comparison_pairs:
        logger.info(f"Checking balance: {treatment_val} vs {control_val}")

        # Filter data to only include these two groups
        comparison_data = data[data[treatment_col].isin([treatment_val, control_val])].copy()

        if comparison_data.empty:
            logger.warning(f"No data for comparison {treatment_val} vs {control_val}. Skipping.")
            continue

        # Check if we have both treatment and control groups
        n_treatment = (comparison_data[treatment_col] == treatment_val).sum()
        n_control = (comparison_data[treatment_col] == control_val).sum()

        if n_treatment == 0:
            logger.warning(f"No treatment units ({treatment_val}) found. Skipping.")
            continue

        if n_control == 0:
            logger.warning(f"No control units ({control_val}) found. Skipping.")
            continue

        # Compute balance for this pair
        balance_df = _compute_balance_for_pair(
            data=comparison_data,
            treatment_col=treatment_col,
            covariates=covariates,
            categorical_covariates=categorical_covariates,
            min_binary_count=min_binary_count,
            threshold=threshold,
            treatment_value=treatment_val,
            control_value=control_val,
            categorical_max_unique=categorical_max_unique,
            weights_col=weights_col,
            logger=logger,
        )

        if not balance_df.empty:
            # Add tracking and count columns
            balance_df["n_treated"] = n_treatment
            balance_df["n_control"] = n_control
            balance_df["treatment_group"] = treatment_val
            balance_df["control_group"] = control_val

            # Reorder columns consistently
            balance_df = balance_df[
                [
                    "covariate",
                    "n_treated",
                    "n_control",
                    "mean_treated",
                    "mean_control",
                    "smd",
                    "balance_flag",
                    "treatment_group",
                    "control_group",
                ]
            ]

            # Log balance summary
            balance_mean = balance_df["balance_flag"].mean()
            logger.info(f"Balance: {balance_mean:.2%}")

            balance_results.append(balance_df)

    # Concatenate all results
    if not balance_results:
        logger.warning("No balance results computed for any comparison")
        return pd.DataFrame(
            columns=[
                "covariate",
                "n_treated",
                "n_control",
                "mean_treated",
                "mean_control",
                "smd",
                "balance_flag",
                "treatment_group",
                "control_group",
            ]
        )

    return pd.concat(balance_results, ignore_index=True)


def _check_stratification_quality(
    df: pd.DataFrame,
    stratify_cols: list[str],
    n_variants: int,
    min_stratum_pct: float,
    min_stratum_n: int,
    logger: logging.Logger,
) -> None:
    """
    Warn when any category of a stratification variable has too few observations.

    A category is flagged when it falls below *both* a minimum percentage of the
    total sample and a minimum absolute count.  Either condition alone may be
    acceptable (e.g. a very large dataset with 3 % in a category is fine), so
    the threshold used is ``max(min_stratum_n, round(n_total * min_stratum_pct))``.
    """
    n_total = len(df)
    threshold = max(min_stratum_n, round(n_total * min_stratum_pct))

    for col in stratify_cols:
        if col == "_dummy_group":
            continue

        display_name = f"{col[5:]} (binned)" if col.startswith("_bin_") else col

        value_counts = df[col].value_counts()
        low_count = value_counts[value_counts < threshold]

        if not low_count.empty:
            details = ", ".join(f"{k} (n={v}, {v / n_total:.1%})" for k, v in low_count.items())
            logger.warning(
                f"Stratification variable '{display_name}' has categories with low prevalence "
                f"(threshold: n>={threshold} or >={min_stratum_pct:.0%}): {details}. "
                "Consider not blocking on this variable."
            )


def balanced_random_assignment(
    df,
    seed=42,
    allocation_ratio=0.5,
    variants=None,
    stratification_covariates=None,
    balance_covariates=None,
    comparison=None,
    smd_threshold=0.1,
    n_bins=5,
    min_stratum_pct=0.05,
    min_stratum_n=10,
    logger=None,
):
    """
    Randomly assign units to variants with forced balance according to allocation ratios.
    Optionally stratify by covariates to ensure balance within strata.
    Always reports variant distribution and, when covariates are provided,
    covariate balance after assignment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the units to assign
    seed : int, optional
        Random seed for reproducibility (default is 42)
    allocation_ratio : float or dict, optional
        If float: proportion allocated to the first variant (remaining goes to the
        second). Works for both the default binary case (test/control) and any
        two-element ``variants`` list (e.g. ``variants=[1, 0]``, ``allocation_ratio=0.667``
        gives 66.7 % to label ``1`` and 33.3 % to label ``0``).
        If dict: mapping of variant names to their allocation ratios (must sum to 1.0).
        Required when ``variants`` has more than 2 elements.
        Default is 0.5 (50/50 split).
    variants : list, optional
        List of variant names. If provided, allocation_ratio should be a dict or
        units will be split equally among variants. If None, uses ['control', 'test']
    stratification_covariates : list, optional
        List of column names to use for block randomization (stratification).
        Continuous covariates are automatically binned into quantiles.
        Warns when any category has low prevalence or too few observations.
    balance_covariates : list, optional
        List of column names to check balance for after assignment. If None, falls
        back to stratification_covariates. Use this to check balance on covariates
        that were not (or could not be) used for stratification.
    comparison : list[tuple], optional
        List of (variant1, variant2) tuples specifying which pairs to compare for
        balance. If None, performs all pairwise comparisons.
    smd_threshold : float, optional
        Threshold for standardized mean difference to flag imbalance (default 0.1).
        Covariates with |SMD| < threshold are considered balanced.
    n_bins : int, optional
        Number of quantile bins for continuous covariates in stratification (default 5).
    min_stratum_pct : float, optional
        Minimum prevalence (as a fraction of total) for a stratification category
        before a warning is raised (default 0.05).
    min_stratum_n : int, optional
        Minimum absolute count for a stratification category before a warning is
        raised (default 10).
    logger : logging.Logger, optional
        Logger for warnings/info. If None, creates one.

    Returns
    -------
    pd.Series
        Series with variant assignments indexed by df.index

    Examples
    --------
    # Binary assignment (test/control) without stratification
    assignment = balanced_random_assignment(df, allocation_ratio=0.5)

    # Stratified assignment with categorical covariates
    assignment = balanced_random_assignment(
        df,
        allocation_ratio=0.5,
        stratification_covariates=['region', 'segment']
    )

    # Stratified assignment with continuous covariates (auto-binned)
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        stratification_covariates=['age', 'previous_purchases']
    )

    # No stratification, but check balance on covariates
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        balance_covariates=['age', 'income', 'region']
    )

    # Stratify by region, check balance on a broader set of covariates
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        stratification_covariates=['region'],
        balance_covariates=['age', 'income', 'region']
    )
    """

    if logger is None:
        logger = get_logger("RandomAssignment")

    np.random.seed(seed)

    df_work = df.copy()
    use_stratification = stratification_covariates is not None

    if use_stratification:
        # Auto-bin continuous covariates for stratification
        stratify_cols = []
        for cov in stratification_covariates:
            is_numeric = pd.api.types.is_numeric_dtype(df_work[cov])
            n_unique = df_work[cov].nunique()
            if is_numeric and n_unique > n_bins * 2:
                bin_col = f"_bin_{cov}"
                df_work[bin_col] = pd.qcut(df_work[cov], q=n_bins, duplicates="drop")
                stratify_cols.append(bin_col)
            else:
                stratify_cols.append(cov)

        # Check stratification variable quality
        n_variants_count = len(variants) if variants is not None else 2
        _check_stratification_quality(
            df=df_work,
            stratify_cols=stratify_cols,
            n_variants=n_variants_count,
            min_stratum_pct=min_stratum_pct,
            min_stratum_n=min_stratum_n,
            logger=logger,
        )
    else:
        df_work["_dummy_group"] = 1
        stratify_cols = ["_dummy_group"]

    assignments = []
    for _, group_df in df_work.groupby(stratify_cols, observed=True):
        n = len(group_df)
        if variants is None:
            if isinstance(allocation_ratio, dict):
                raise ValueError("When variants is None, allocation_ratio must be a float")
            # Use round() instead of int() to avoid systematic under-allocation when
            # many small strata are used: int() always floors, so every stratum gives
            # slightly fewer treated units than the target ratio. round() distributes
            # the rounding error evenly (up/down) so errors cancel across strata.
            n_test = round(n * allocation_ratio)
            n_control = n - n_test
            assignment = ["test"] * n_test + ["control"] * n_control
        else:
            if isinstance(allocation_ratio, dict):
                total = sum(allocation_ratio.values())
                if not np.isclose(total, 1.0):
                    raise ValueError(f"Allocation ratios must sum to 1.0, got {total}")

                if set(variants) != set(allocation_ratio.keys()):
                    raise ValueError("Variants list must match allocation_ratio keys")

                assignment = []
                allocated = 0
                for _, variant in enumerate(variants[:-1]):
                    n_variant = round(n * allocation_ratio[variant])
                    assignment.extend([variant] * n_variant)
                    allocated += n_variant
                assignment.extend([variants[-1]] * (n - allocated))
            else:
                # Float allocation_ratio with explicit variants list.
                if len(variants) == 2:
                    # Ratio applies to the first variant; remainder goes to the second.
                    # e.g. variants=[1, 0], allocation_ratio=0.667 → 66.7% label-1, 33.3% label-0
                    n_first = round(n * allocation_ratio)
                    assignment = [variants[0]] * n_first + [variants[1]] * (n - n_first)
                else:
                    # For 3+ variants a float ratio is ambiguous; fall back to equal
                    # allocation.  Use a dict to specify custom per-variant ratios.
                    n_per_variant = n // len(variants)
                    remainder = n % len(variants)
                    assignment = []
                    for i, variant in enumerate(variants):
                        n_variant = n_per_variant + (1 if i < remainder else 0)
                        assignment.extend([variant] * n_variant)

        np.random.shuffle(assignment)
        assignments.append(pd.Series(assignment, index=group_df.index))

    final_assignments = pd.concat(assignments).sort_index()

    # Always print variant distribution
    counts = final_assignments.value_counts()
    n_total = len(final_assignments)
    print("\n" + "=" * 50)
    print("Variant Distribution")
    print("-" * 50)
    for variant, count in counts.items():
        pct = count / n_total * 100
        print(f"  {str(variant):<20} {count:>6,}   ({pct:.1f}%)")
    print("-" * 50)
    print(f"  {'Total':<20} {n_total:>6,}   (100.0%)")
    print("=" * 50)

    # Determine which covariates to check balance for
    covs_to_check = balance_covariates or (stratification_covariates if use_stratification else None)

    if covs_to_check is not None:
        temp_df = df.copy()
        temp_df["_assignment"] = final_assignments

        unique_variants = final_assignments.unique()

        # Generate comparison pairs
        if comparison is None:
            from itertools import combinations

            comparison_pairs = list(combinations(unique_variants, 2))
        else:
            comparison_pairs = comparison

        print("\n" + "=" * 70)
        print("Balance Check After Assignment")
        print("=" * 70)

        for var1, var2 in comparison_pairs:
            n_var1 = (temp_df["_assignment"] == var1).sum()
            n_var2 = (temp_df["_assignment"] == var2).sum()

            balance_df = _check_assignment_balance(temp_df, "_assignment", covs_to_check, var1, var2, smd_threshold)

            print(f"\nComparison: {var1} (n={n_var1:,}) vs {var2} (n={n_var2:,})")
            print("-" * 70)
            display_df = balance_df[["covariate"]].copy()
            display_df[f"n_{var1}"] = n_var1
            display_df[f"n_{var2}"] = n_var2
            display_df[f"mean_{var1}"] = balance_df[f"mean_{var1}"]
            display_df[f"mean_{var2}"] = balance_df[f"mean_{var2}"]
            display_df["smd"] = balance_df["smd"]
            display_df["balanced"] = balance_df["balance_flag"].map({1: "✓", 0: "✗"})

            print(display_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else x))

            n_balanced = balance_df["balance_flag"].sum()
            n_total_covs = len(balance_df)
            mean_abs_smd = balance_df["smd"].abs().mean()
            max_abs_smd = balance_df["smd"].abs().max()

            print(f"\nSummary: {n_balanced}/{n_total_covs} covariates balanced (|SMD| < {smd_threshold})")
            print(f"Mean |SMD|: {mean_abs_smd:.4f}")
            print(f"Max |SMD|: {max_abs_smd:.4f}")

            if n_balanced < n_total_covs:
                imbalanced = balance_df[balance_df["balance_flag"] == 0]["covariate"].tolist()
                print(f"⚠️  Imbalanced covariates: {', '.join(imbalanced)}")

        print("\n" + "=" * 70)

    return final_assignments


def false_positive_risk(alpha: float, power: float, prior_success_rate: float) -> float:
    """
    False Positive Risk (FPR) via Bayes Rule — Kohavi & Chen (2024).

    FPR = P(H0 is true | statistically significant result)
        = (alpha * pi) / (alpha * pi + power * (1 - pi))
    where pi = 1 - prior_success_rate.

    Parameters
    ----------
    alpha : float
        Significance level (e.g. 0.05).
    power : float
        Statistical power (1 - beta), e.g. 0.80.
    prior_success_rate : float
        Estimated proportion of experiments with a true positive effect,
        derived from historical win rates (must be in (0, 1)).

    Returns
    -------
    float
        Probability that a statistically significant result is a false positive.

    Examples
    --------
    >>> # alpha=0.10, power=0.80, prior_success_rate=0.12 -> ~47.8% FPR
    >>> false_positive_risk(alpha=0.10, power=0.80, prior_success_rate=0.12)
    0.478...
    """
    if not 0 < prior_success_rate < 1:
        raise ValueError("prior_success_rate must be strictly between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 < power <= 1:
        raise ValueError("power must be in (0, 1]")
    pi = 1.0 - prior_success_rate
    return (alpha * pi) / (alpha * pi + power * (1.0 - pi))


def estimate_true_success_rate(win_rate: float, alpha: float, power: float) -> float:
    """
    Estimate the true success rate from the observed win rate — Kohavi & Chen (2024, §4.3).

    Inverts the conditional probability formula:
        P(SS) = pi * (alpha + beta - 1) + 1 - beta
    Solving for pi (= 1 - true_success_rate):
        pi = (power - win_rate) / (power - alpha)

    Result is clamped to [0, 1] — a win_rate below alpha implies all wins are
    false positives (true success rate → 0).

    Parameters
    ----------
    win_rate : float
        Observed proportion of statistically significant positive results.
    alpha : float
        Significance level used in the experiments (e.g. 0.05).
    power : float
        Statistical power (1 - beta), e.g. 0.80.

    Returns
    -------
    float
        Estimated proportion of experiments with a genuine positive effect.

    Examples
    --------
    >>> # win_rate=0.12, alpha=0.10, power=0.80 -> ~2.9% true success rate
    >>> estimate_true_success_rate(win_rate=0.12, alpha=0.10, power=0.80)
    0.028...
    """
    if power == alpha:
        raise ValueError("power and alpha cannot be equal (division by zero)")
    pi = (power - win_rate) / (power - alpha)
    pi = float(np.clip(pi, 0.0, 1.0))
    return 1.0 - pi


def winners_curse_estimate(
    effect: float,
    standard_error: float,
    alpha: float = 0.05,
    ci: float = 0.95,
    alternative: str = "two-sided",
) -> dict:
    """
    Winner's-curse correction for a single estimate selected by significance.

    Models ``effect ~ N(beta, standard_error**2)`` truncated to the selection
    region implied by ``alternative``:

    - ``"two-sided"`` (default): ``|effect| >= z* * SE`` with
      ``z* = Phi^{-1}(1 - alpha/2)``.
    - ``"greater"``: ``effect >= z* * SE`` with ``z* = Phi^{-1}(1 - alpha)``
      (one-sided launch / positive win; Kessler-style).
    - ``"less"``: ``effect <= -z* * SE`` with the same one-sided ``z*``.

    Returns the median-unbiased estimate of ``beta`` (the value whose
    conditional CDF at ``effect`` equals 0.5) and a selection-adjusted
    equal-tailed confidence interval. Operates on whatever scale
    ``effect``/``standard_error`` are supplied on (for GLMs that is the
    log/coefficient scale).

    The function assumes the effect lies in the selection region. If this
    precondition is violated a ``RuntimeWarning`` is emitted and the
    computation proceeds; it does not raise so downstream pipelines that pass
    already-screened rows are not disrupted when the screening threshold
    differs slightly from ``alpha``.

    Parameters
    ----------
    effect : float
        The observed (significant) point estimate.
    standard_error : float
        Its standard error; must be > 0.
    alpha : float
        Significance level that defined selection (default 0.05). Interpreted
        as two-sided for ``alternative="two-sided"`` and one-sided otherwise.
    ci : float
        Confidence level for the adjusted interval (default 0.95).
    alternative : {"two-sided", "greater", "less"}
        Selection region (default ``"two-sided"``).

    Returns
    -------
    dict
        ``corrected`` (median-unbiased estimate), ``ci_lower``, ``ci_upper``
        (selection-adjusted interval), ``observed_z`` (= effect/standard_error),
        ``shrinkage`` (= corrected/effect), ``alternative``.
    """
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    if not (np.isfinite(standard_error) and standard_error > 0):
        raise ValueError("standard_error must be positive and finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    s = float(standard_error)
    b = float(effect)
    if alternative == "two-sided":
        c = norm.ppf(1.0 - alpha / 2.0) * s
        selected = abs(b) >= c
    else:
        c = norm.ppf(1.0 - alpha) * s
        selected = b >= c if alternative == "greater" else b <= -c

    if not selected:
        warnings.warn(
            "winners_curse_estimate: effect is outside the selection region; "
            "the correction assumes the estimate was selected by significance.",
            RuntimeWarning,
            stacklevel=2,
        )

    observed_z = b / s

    def cond_cdf(beta: float) -> float:
        if alternative == "two-sided":
            # S = (-inf, -c] U [c, inf)
            p_sel = norm.cdf((-c - beta) / s) + norm.sf((c - beta) / s)
            if b >= c:
                num = norm.cdf((-c - beta) / s) + norm.cdf((b - beta) / s) - norm.cdf((c - beta) / s)
            elif b <= -c:
                num = norm.cdf((b - beta) / s)
            else:  # excluded gap (-c, c): only the lower selected tail (-inf, -c] is <= b
                num = norm.cdf((-c - beta) / s)
            return num / p_sel
        if alternative == "greater":
            # S = [c, inf). Use survival ratio to avoid 0/0 when beta << c.
            if b < c:
                return 0.0
            z_c = (c - beta) / s
            z_b = (b - beta) / s
            den = norm.sf(z_c)
            if den == 0.0:
                return 1.0
            return float(1.0 - np.clip(norm.sf(z_b) / den, 0.0, 1.0))
        # alternative == "less": S = (-inf, -c]. Use CDF ratio for stability.
        if b > -c:
            return 1.0
        z_uc = (-c - beta) / s  # upper endpoint of selection region
        z_b = (b - beta) / s
        den = norm.cdf(z_uc)
        if den == 0.0:
            return 0.0
        return float(np.clip(norm.cdf(z_b) / den, 0.0, 1.0))

    def invert(target: float) -> float:
        # cond_cdf is monotone DECREASING in beta (1 at -inf, 0 at +inf).
        f = lambda beta: cond_cdf(beta) - target  # noqa: E731
        # One-sided near-threshold observations need a wide left bracket:
        # the median-unbiased beta can be many SEs below the truncation point.
        lo, hi = b - 2.0 * s, b + 2.0 * s
        steps = 0
        while f(lo) < 0 and steps < 200:
            lo -= 4.0 * s
            steps += 1
        steps = 0
        while f(hi) > 0 and steps < 200:
            hi += 4.0 * s
            steps += 1
        try:
            return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-12, maxiter=200))
        except ValueError:
            warnings.warn(
                "winners_curse_estimate: root-finding failed to bracket; returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")

    gamma = 1.0 - ci
    corrected = invert(0.5)
    ci_lower = invert(1.0 - gamma / 2.0)  # decreasing CDF: high target -> small beta
    ci_upper = invert(gamma / 2.0)
    shrinkage = corrected / b if b != 0 else float("nan")

    return {
        "corrected": corrected,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "observed_z": observed_z,
        "shrinkage": shrinkage,
        "alternative": alternative,
    }


def _paule_mandel_tau2(y: np.ndarray, v: np.ndarray, prior_mean: float = 0.0) -> float:
    """
    Method-of-moments (Paule-Mandel style) between-estimate variance with the
    prior mean FIXED at ``prior_mean``. Solves, for tau2 >= 0,
    ``sum (y_i - prior_mean)^2 / (v_i + tau2) = k``. Monotone decreasing, so
    a single root via brentq; returns 0 when the estimates are no more
    dispersed than their standard errors imply.
    """
    k = y.size
    resid2 = (y - prior_mean) ** 2

    def g(tau2: float) -> float:
        return float(np.sum(resid2 / (v + tau2)) - k)

    if g(0.0) <= 0:
        return 0.0
    hi = 100.0 * float(np.max(v))
    steps = 0
    while g(hi) > 0 and steps < 80:
        hi *= 2.0
        steps += 1
    return float(brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


def empirical_bayes_shrinkage(
    effects,
    standard_errors,
    prior_mean: float = 0.0,
    ci: float = 0.95,
    tau2: float | None = None,
) -> dict:
    """
    Empirical-Bayes (normal-prior) shrinkage of a family of estimates.

    Assumes ``beta_i ~ N(prior_mean, tau2)`` with ``effect_i | beta_i ~
    N(beta_i, se_i**2)``, estimates ``tau2`` by method of moments
    (:func:`_paule_mandel_tau2`), and returns posterior means and credible
    intervals. High-variance "winners" shrink most. All inputs must be on the
    same scale (e.g. all log-odds, or all mean differences).

    Alternatively, pass a fixed ``tau2`` learned from historical experiments
    (e.g. ``empirical_bayes_shrinkage(past_effects, past_ses)["tau2"]``) to
    shrink any number of new estimates — including a single one — with that
    external prior instead of re-learning it (van Zwet, Schwab & Senn 2021;
    Azevedo et al. 2020).

    Parameters
    ----------
    effects, standard_errors : array-like
        Estimates and their standard errors, on a common scale.
    prior_mean : float
        Mean of the normal prior (default 0.0).
    ci : float
        Credible-interval level (default 0.95).
    tau2 : float, optional
        Fixed prior variance. When given, it is used as-is (no estimation)
        and a single estimate is allowed; when None (default), ``tau2`` is
        learned from the supplied estimates, which requires at least 3.

    Returns
    -------
    dict
        ``shrunk`` (posterior means), ``shrinkage_factor`` (= tau2/(tau2+se^2)),
        ``posterior_sd``, ``ci_lower``, ``ci_upper`` (np.ndarray aligned with
        inputs), plus scalar ``tau2`` and ``prior_mean``.
    """
    y = np.asarray(effects, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("effects and standard_errors must be 1-D arrays of equal length")
    if tau2 is None:
        if y.size < 3:
            raise ValueError(
                "empirical Bayes requires at least 3 estimates to learn a prior; "
                "pass a fixed tau2 (e.g. learned from historical experiments) to shrink fewer"
            )
    else:
        if not (np.isfinite(tau2) and tau2 >= 0):
            raise ValueError("tau2 must be finite and >= 0")
        if y.size < 1:
            raise ValueError("effects must contain at least one estimate")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))) or np.any(s <= 0):
        raise ValueError("all standard_errors must be positive and finite, and effects finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    v = s**2
    if tau2 is None:
        tau2 = _paule_mandel_tau2(y, v, prior_mean=prior_mean)
    shrinkage_factor = tau2 / (tau2 + v)
    shrunk = prior_mean + shrinkage_factor * (y - prior_mean)
    posterior_sd = np.sqrt(tau2 * v / (tau2 + v))
    z = norm.ppf(1.0 - (1.0 - ci) / 2.0)
    return {
        "shrunk": shrunk,
        "shrinkage_factor": shrinkage_factor,
        "posterior_sd": posterior_sd,
        "ci_lower": shrunk - z * posterior_sd,
        "ci_upper": shrunk + z * posterior_sd,
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
    }


def _validate_effects_ses(effects, standard_errors) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(effects, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("effects and standard_errors must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))) or np.any(s <= 0):
        raise ValueError("all standard_errors must be positive and finite, and effects finite")
    return y, s


def t_prior_shrinkage(
    effects,
    standard_errors,
    scale: float,
    df: float,
    prior_mean: float = 0.0,
    ci: float = 0.95,
) -> dict:
    """
    Bayesian shrinkage under a fat-tailed Student-t prior.

    Assumes ``beta_i ~ t_df(prior_mean, scale)`` with ``effect_i | beta_i ~
    N(beta_i, se_i**2)`` and returns the posterior mean, sd, and equal-tailed
    credible interval per estimate (computed by numerical integration; the
    t-prior posterior has no closed form). Unlike normal-prior shrinkage the
    posterior mean is nonlinear in the estimate: moderate effects shrink
    hard while very large ones pass through mostly untouched, which suits
    experiment archives with occasional genuine big winners (Azevedo et al.
    2020, "A/B Testing with Fat Tails").

    Prior parameters are external by design — learn them once from historical
    experiments with :func:`fit_t_prior` — so a single new estimate is enough.

    Parameters
    ----------
    effects, standard_errors : array-like
        Estimates and their standard errors, on a common scale.
    scale : float
        Scale of the t prior; must be > 0.
    df : float
        Degrees of freedom of the t prior; must be > 0 (typically 3-10; the
        prior variance is only finite for df > 2).
    prior_mean : float
        Location of the prior (default 0.0).
    ci : float
        Credible-interval level (default 0.95).

    Returns
    -------
    dict
        ``shrunk`` (posterior means), ``shrinkage_factor``
        (= (shrunk - prior_mean)/(effect - prior_mean), NaN at the prior mean),
        ``posterior_sd``, ``ci_lower``, ``ci_upper`` (np.ndarray aligned with
        inputs), plus scalars ``scale``, ``df``, ``prior_mean`` and ``tau2``
        (implied prior variance, ``inf`` when df <= 2).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError("scale must be positive and finite")
    if not (np.isfinite(df) and df > 0):
        raise ValueError("df must be positive and finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    gamma = 1.0 - ci
    shrunk = np.empty_like(y)
    posterior_sd = np.empty_like(y)
    ci_lower = np.empty_like(y)
    ci_upper = np.empty_like(y)
    for i, (b, se) in enumerate(zip(y, s, strict=True)):
        lo = min(b - 8.0 * se, prior_mean - 8.0 * scale)
        hi = max(b + 8.0 * se, prior_mean + 8.0 * scale)
        grid = np.linspace(lo, hi, 4001)
        log_post = norm.logpdf(b, loc=grid, scale=se) + t_dist.logpdf(grid, df, loc=prior_mean, scale=scale)
        dens = np.exp(log_post - log_post.max())
        dens /= np.trapz(dens, grid)
        mean = np.trapz(grid * dens, grid)
        var = np.trapz((grid - mean) ** 2 * dens, grid)
        cdf = cumulative_trapezoid(dens, grid, initial=0.0)
        cdf /= cdf[-1]
        shrunk[i] = mean
        posterior_sd[i] = np.sqrt(max(var, 0.0))
        ci_lower[i] = np.interp(gamma / 2.0, cdf, grid)
        ci_upper[i] = np.interp(1.0 - gamma / 2.0, cdf, grid)

    centered = y - prior_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        shrinkage_factor = np.where(centered != 0, (shrunk - prior_mean) / centered, np.nan)
    tau2 = scale**2 * df / (df - 2.0) if df > 2 else float("inf")
    return {
        "shrunk": shrunk,
        "shrinkage_factor": shrinkage_factor,
        "posterior_sd": posterior_sd,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "scale": float(scale),
        "df": float(df),
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
    }


def fit_t_prior(
    effects,
    standard_errors,
    prior_mean: float = 0.0,
    df: float | None = None,
) -> dict:
    """
    Fit a Student-t prior to a historical archive of experiment estimates.

    Maximizes the marginal likelihood of ``effect_i ~ N(beta_i, se_i**2)``
    with ``beta_i ~ t_df(prior_mean, scale)`` (integrated via Gauss-Hermite
    quadrature). Learn the prior once from past experiments, then shrink each
    new result with :func:`t_prior_shrinkage` (or pass the returned dict as
    ``prior=`` to ``ExperimentAnalyzer.winners_curse_summary``).

    Parameters
    ----------
    effects, standard_errors : array-like
        Historical estimates and their standard errors, on a common scale.
        At least 3 are required; reliably estimating ``df`` needs a large
        archive (dozens of experiments) — with a small one, fix ``df``
        (e.g. ``df=4``) so only the scale is learned.
    prior_mean : float
        Location of the prior (default 0.0).
    df : float, optional
        Fix the degrees of freedom and fit only the scale. When None
        (default) both are fitted, with ``df`` constrained to > 2 so the
        prior variance is finite.

    Returns
    -------
    dict
        ``scale``, ``df``, ``tau2`` (implied prior variance), ``prior_mean``,
        ``loglik``, ``n``.
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 3:
        raise ValueError("fit_t_prior requires at least 3 estimates")
    if df is not None and not (np.isfinite(df) and df > 0):
        raise ValueError("df must be positive and finite")

    gh_x, gh_w = np.polynomial.hermite.hermgauss(64)
    nodes = y[:, None] + np.sqrt(2.0) * s[:, None] * gh_x[None, :]
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

    def nll(log_scale: float, dfv: float) -> float:
        marginal = inv_sqrt_pi * (t_dist.pdf(nodes, dfv, loc=prior_mean, scale=np.exp(log_scale)) * gh_w).sum(axis=1)
        return float(-np.sum(np.log(np.maximum(marginal, 1e-300))))

    tau2_0 = _paule_mandel_tau2(y, s**2, prior_mean=prior_mean)
    scale0 = np.sqrt(tau2_0) if tau2_0 > 0 else 0.5 * float(np.median(s))
    if df is not None:
        res = minimize(lambda u: nll(u[0], df), x0=[np.log(scale0)], method="Nelder-Mead")
        scale_hat, df_hat = float(np.exp(res.x[0])), float(df)
    else:
        # df = 2 + exp(u) keeps the fitted prior variance finite
        res = minimize(lambda u: nll(u[0], 2.0 + np.exp(u[1])), x0=[np.log(scale0), np.log(3.0)], method="Nelder-Mead")
        scale_hat, df_hat = float(np.exp(res.x[0])), float(2.0 + np.exp(res.x[1]))

    tau2 = scale_hat**2 * df_hat / (df_hat - 2.0) if df_hat > 2 else float("inf")
    return {
        "scale": scale_hat,
        "df": df_hat,
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
        "loglik": float(-res.fun),
        "n": int(y.size),
    }


def fit_t_prior_with_estimated_mean(
    effects,
    standard_errors,
    *,
    df: float = 4.0,
    mean_ci: float = 0.95,
) -> dict:
    """
    Fit a Student-t prior's location and scale by profile likelihood.

    :func:`fit_t_prior` treats ``prior_mean`` as fixed. This helper profiles
    over that argument, then obtains a likelihood-ratio interval for the
    fitted location. With ``df > 1`` the Student-t location is also its mean,
    so the interval answers whether the archive's average underlying effect
    is distinguishable from zero.

    Learn the prior once from historical experiments, then shrink new results
    with :func:`t_prior_shrinkage` (pass ``prior_mean=fit["prior_mean"]``) or
    pass the returned dict as ``prior=`` to
    ``ExperimentAnalyzer.winners_curse_summary``.

    Parameters
    ----------
    effects, standard_errors : array-like
        Historical estimates and their standard errors, on a common scale.
        At least 3 are required.
    df : float
        Fixed degrees of freedom of the t prior; must be > 1 so the prior
        mean exists (default 4.0).
    mean_ci : float
        Nominal level of the profile likelihood-ratio interval for the prior
        mean (default 0.95).

    Returns
    -------
    dict
        Everything returned by :func:`fit_t_prior` at the fitted mean, plus
        ``prior_mean_ci_lower``, ``prior_mean_ci_upper``,
        ``prior_mean_ci_level``, and ``prior_mean_method``
        (``"profile_likelihood"``).
    """
    y, se = _validate_effects_ses(effects, standard_errors)
    if y.size < 3:
        raise ValueError("at least 3 estimates are required")
    if not 0 < mean_ci < 1:
        raise ValueError("mean_ci must be strictly between 0 and 1")
    if not np.isfinite(df) or df <= 1:
        raise ValueError("df must be finite and greater than 1 so the prior mean exists")

    @cache
    def conditional_fit(prior_mean: float) -> dict:
        return fit_t_prior(y, se, prior_mean=float(prior_mean), df=df)

    def objective(prior_mean: float) -> float:
        return -conditional_fit(float(prior_mean))["loglik"]

    # Robust bounds: min/max effects are unstable when the archive has
    # enormous ratios from near-zero baselines.
    center = float(np.median(y))
    q05, q95 = np.quantile(y, [0.05, 0.95])
    span = max(float(q95 - q05), 10.0 * float(np.median(se)), 0.01)
    for _ in range(4):
        lower, upper = center - span, center + span
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"prior-mean fit failed: {result.message}")
        edge_tolerance = max(span * 1e-3, 1e-8)
        if lower + edge_tolerance < result.x < upper - edge_tolerance:
            break
        span *= 4.0
    else:
        raise RuntimeError("prior-mean fit remained on the search boundary")

    prior_mean = float(result.x)
    prior = conditional_fit(prior_mean).copy()
    max_loglik = float(prior["loglik"])
    target_loglik = max_loglik - 0.5 * float(chi2.ppf(mean_ci, df=1))

    def profile_distance(candidate: float) -> float:
        return float(conditional_fit(float(candidate))["loglik"] - target_loglik)

    initial_step = max(
        float(prior["scale"]) / sqrt(y.size),
        float(np.median(se)) / sqrt(y.size),
        1e-6,
    )

    def find_endpoint(direction: float) -> float:
        inner = prior_mean
        step = initial_step
        for _ in range(60):
            outer = prior_mean + direction * step
            if profile_distance(outer) <= 0:
                lo, hi = sorted((inner, outer))
                return float(brentq(profile_distance, lo, hi, xtol=1e-8))
            inner = outer
            step *= 1.8
        raise RuntimeError("could not bracket the prior-mean likelihood interval")

    prior.update(
        {
            "prior_mean": prior_mean,
            "prior_mean_ci_lower": find_endpoint(-1.0),
            "prior_mean_ci_upper": find_endpoint(1.0),
            "prior_mean_ci_level": float(mean_ci),
            "prior_mean_method": "profile_likelihood",
        }
    )
    return prior


def joint_metric_shrinkage(
    primary_effects,
    primary_ses,
    guardrail_effects,
    guardrail_ses,
    *,
    rho: float,
    prior_sd_primary: float,
    prior_sd_guard: float | None = None,
    prior_mean_primary: float = 0.0,
    prior_mean_guard: float = 0.0,
    ci: float = 0.95,
) -> dict:
    """
    Bivariate normal–normal shrinkage of primary and guardrail effects.

    Models true effects ``(delta, gamma)`` as jointly normal with correlation
    ``rho`` and known prior SDs, observed with independent sampling errors.
    Returns posterior means and equal-tailed CIs. ``rho`` is caller-supplied
    (not estimated). When ``rho=0`` the primary posterior matches univariate
    normal EB with the same prior SD.

    Parameters
    ----------
    primary_effects, primary_ses, guardrail_effects, guardrail_ses : array-like
        Aligned 1-D arrays of equal length.
    rho : float
        Corr(true primary, true guardrail); must be in (-1, 1).
    prior_sd_primary : float
        Prior SD of the primary true effect; must be > 0.
    prior_sd_guard : float, optional
        Prior SD of the guardrail; defaults to ``prior_sd_primary``.
    prior_mean_primary, prior_mean_guard : float
        Prior means (default 0).
    ci : float
        Credible-interval level (default 0.95).

    Returns
    -------
    dict
        ``primary_shrunk``, ``guard_shrunk``, ``primary_posterior_sd``,
        ``guard_posterior_sd``, ``primary_ci_lower``, ``primary_ci_upper``,
        ``guard_ci_lower``, ``guard_ci_upper``, plus scalar prior params.
    """
    x, sx = _validate_effects_ses(primary_effects, primary_ses)
    g, sg = _validate_effects_ses(guardrail_effects, guardrail_ses)
    if x.shape != g.shape:
        raise ValueError("primary and guardrail arrays must have the same length")
    if x.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not (np.isfinite(rho) and abs(rho) < 1.0):
        raise ValueError("rho must be finite and strictly inside (-1, 1)")
    if not (np.isfinite(prior_sd_primary) and prior_sd_primary > 0):
        raise ValueError("prior_sd_primary must be positive and finite")
    if prior_sd_guard is None:
        prior_sd_guard = prior_sd_primary
    if not (np.isfinite(prior_sd_guard) and prior_sd_guard > 0):
        raise ValueError("prior_sd_guard must be positive and finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    tau_p2 = float(prior_sd_primary) ** 2
    tau_g2 = float(prior_sd_guard) ** 2
    cov = float(rho) * prior_sd_primary * prior_sd_guard
    sigma = np.array([[tau_p2, cov], [cov, tau_g2]], dtype=float)
    mu = np.array([prior_mean_primary, prior_mean_guard], dtype=float)
    z = norm.ppf(1.0 - (1.0 - ci) / 2.0)

    primary_shrunk = np.empty_like(x)
    guard_shrunk = np.empty_like(x)
    primary_sd = np.empty_like(x)
    guard_sd = np.empty_like(x)
    for i in range(x.size):
        v = np.diag([sx[i] ** 2, sg[i] ** 2])
        post_cov = np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(v))
        a = sigma @ np.linalg.inv(sigma + v)
        post_mean = mu + a @ (np.array([x[i], g[i]]) - mu)
        primary_shrunk[i] = post_mean[0]
        guard_shrunk[i] = post_mean[1]
        primary_sd[i] = np.sqrt(max(post_cov[0, 0], 0.0))
        guard_sd[i] = np.sqrt(max(post_cov[1, 1], 0.0))

    return {
        "primary_shrunk": primary_shrunk,
        "guard_shrunk": guard_shrunk,
        "primary_posterior_sd": primary_sd,
        "guard_posterior_sd": guard_sd,
        "primary_ci_lower": primary_shrunk - z * primary_sd,
        "primary_ci_upper": primary_shrunk + z * primary_sd,
        "guard_ci_lower": guard_shrunk - z * guard_sd,
        "guard_ci_upper": guard_shrunk + z * guard_sd,
        "rho": float(rho),
        "prior_sd_primary": float(prior_sd_primary),
        "prior_sd_guard": float(prior_sd_guard),
        "prior_mean_primary": float(prior_mean_primary),
        "prior_mean_guard": float(prior_mean_guard),
    }


def cumulative_impact(
    effects,
    standard_errors,
    *,
    shipped=None,
    prior=None,
    tau2: float | None = None,
    prior_mean: float = 0.0,
    aggregation: str = "sum",
    coverage=None,
    ci: float = 0.95,
    min_shipped: int = 1,
) -> dict:
    """
    Noise-adjusted cumulative impact of shipped experiments (Kessler / Datadog).

    Shrinks every estimate with a normal or Student-t archive prior, then
    aggregates **only** the ``shipped`` subset. Pass the real launch rule in
    ``shipped`` (including guardrails); do not confuse significance on the
    primary with shipping.

    Aggregation:

    - ``"sum"``: ``sum(w_i * theta_hat_i)`` over shipped (absolute / additive).
    - ``"product"``: ``prod(1 + w_i * theta_hat_i) - 1`` (relative lifts).

    ``w_i`` is coverage (share of eligible users) when ``coverage`` is given,
    else 1. The CI uses the plug-in sum of posterior variances (fixed-prior
    Kessler formula); when the prior is learned from the same data the
    interval is slightly anti-conservative.

    Parameters
    ----------
    effects, standard_errors : array-like
        Experiment lifts and SEs on a common scale.
    shipped : array-like of bool, optional
        Launch mask; default ships all experiments.
    prior : dict or ``"map"``, optional
        External prior: ``{"tau2": ...}`` (normal), ``{"scale", "df"}``
        (Student-t), or ``"map"`` to fit a Datadog-style Half-Cauchy MAP
        normal prior via :func:`fit_normal_prior_map`. Optional ``prior_mean``
        in dict priors.
    tau2 : float, optional
        Fixed normal prior variance when ``prior`` is omitted.
    prior_mean : float
        Shrinkage location when learning / fixing a normal prior (default 0).
        Ignored when ``prior`` already supplies ``prior_mean``.
    aggregation : {"sum", "product"}
        How to combine shipped posterior means.
    coverage : array-like, optional
        Per-experiment exposure weights in ``[0, 1]`` for a global view.
    ci : float
        Interval level (default 0.95).
    min_shipped : int
        Require at least this many shipped experiments (default 1).

    Returns
    -------
    dict
        ``cumulative``, ``ci_lower``, ``ci_upper``, ``n_total``, ``n_shipped``,
        ``shrunk``, ``posterior_sd``, ``shipped_mask``, ``aggregation``,
        prior metadata (``prior_family``, ``tau2`` / ``scale`` / ``df``,
        ``prior_mean``).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if aggregation not in {"sum", "product"}:
        raise ValueError("aggregation must be 'sum' or 'product'")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")
    if not isinstance(min_shipped, int) or min_shipped < 1:
        raise ValueError("min_shipped must be an integer >= 1")

    if shipped is None:
        shipped_mask = np.ones(y.size, dtype=bool)
    else:
        shipped_mask = np.asarray(shipped, dtype=bool)
        if shipped_mask.shape != y.shape:
            raise ValueError("shipped must be a 1-D boolean array aligned with effects")

    n_shipped = int(shipped_mask.sum())
    if n_shipped < min_shipped:
        raise ValueError(f"need at least {min_shipped} shipped experiments; got {n_shipped}")

    if coverage is None:
        weights = np.ones(y.size, dtype=float)
    else:
        weights = np.asarray(coverage, dtype=float)
        if weights.shape != y.shape:
            raise ValueError("coverage must be a 1-D array aligned with effects")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("coverage values must be finite and in [0, 1]")

    # Resolve prior / shrink all rows (fit on all; aggregate shipped only).
    prior_meta: dict = {"prior_mean": float(prior_mean)}
    if prior == "map":
        prior = fit_normal_prior_map(y, s)
    if prior is not None:
        if not isinstance(prior, dict):
            raise ValueError("prior must be a dict or the string 'map'")
        loc = float(prior.get("prior_mean", prior_mean))
        if "scale" in prior and "df" in prior:
            eb = t_prior_shrinkage(y, s, scale=float(prior["scale"]), df=float(prior["df"]), prior_mean=loc, ci=ci)
            prior_meta.update(
                {
                    "prior_family": "student_t",
                    "scale": float(eb["scale"]),
                    "df": float(eb["df"]),
                    "tau2": float(eb["tau2"]),
                    "prior_mean": loc,
                }
            )
        elif "tau2" in prior:
            eb = empirical_bayes_shrinkage(y, s, prior_mean=loc, ci=ci, tau2=float(prior["tau2"]))
            prior_meta.update(
                {
                    "prior_family": "normal",
                    "tau2": float(eb["tau2"]),
                    "prior_mean": loc,
                    "prior_fit": prior.get("method"),
                }
            )
        else:
            raise ValueError("prior must contain 'tau2' or both 'scale' and 'df'")
    elif tau2 is not None:
        eb = empirical_bayes_shrinkage(y, s, prior_mean=prior_mean, ci=ci, tau2=tau2)
        prior_meta.update({"prior_family": "normal", "tau2": float(eb["tau2"])})
    else:
        eb = empirical_bayes_shrinkage(y, s, prior_mean=prior_mean, ci=ci)
        prior_meta.update({"prior_family": "normal", "tau2": float(eb["tau2"])})

    shrunk = np.asarray(eb["shrunk"], dtype=float)
    post_sd = np.asarray(eb["posterior_sd"], dtype=float)
    w_l = weights[shipped_mask]
    theta_l = shrunk[shipped_mask]
    sd_l = post_sd[shipped_mask]
    contrib = w_l * theta_l

    if aggregation == "sum":
        cumulative = float(np.sum(contrib))
        se_cum = float(np.sqrt(np.sum((w_l * sd_l) ** 2)))
        z = norm.ppf(1.0 - (1.0 - ci) / 2.0)
        ci_lower = cumulative - z * se_cum
        ci_upper = cumulative + z * se_cum
    else:
        factors = 1.0 + contrib
        if np.any(~np.isfinite(factors)) or np.any(factors <= 0):
            raise ValueError("product aggregation requires 1 + coverage * shrunk > 0 for every shipped experiment")
        cumulative = float(np.prod(factors) - 1.0)
        # Delta method on log(1+Δ) = sum log(1 + w θ)
        var_log = float(np.sum((w_l * sd_l / factors) ** 2))
        se_log = np.sqrt(var_log)
        z = norm.ppf(1.0 - (1.0 - ci) / 2.0)
        log_point = float(np.sum(np.log(factors)))
        ci_lower = float(np.exp(log_point - z * se_log) - 1.0)
        ci_upper = float(np.exp(log_point + z * se_log) - 1.0)

    return {
        "cumulative": cumulative,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_total": int(y.size),
        "n_shipped": n_shipped,
        "shrunk": shrunk,
        "posterior_sd": post_sd,
        "shipped_mask": shipped_mask,
        "aggregation": aggregation,
        **prior_meta,
    }


def process_level_total_effect(
    effects,
    standard_errors,
    *,
    alpha: float = 0.05,
    alternative: str = "greater",
    n_bootstrap: int = 0,
    ci: float = 0.95,
    random_seed: int | None = None,
) -> dict:
    """
    Lee & Shen (2018) process-level correction for the expected total of winners.

    Estimand is ``E[T_A]`` where ``T_A = sum_{i in A} a_i`` and ``A`` is the
    (random) set of experiments that pass a one-sided launch threshold. The
    naive sum ``S_A`` is biased high; the plug-in debiased total subtracts a
    bias contribution from **every** experiment (selected or not):

    ``hat{T}_A = S_A - sum_i SE_i * phi((SE_i * b_i - X_i) / SE_i)``.

    This differs from :func:`winners_curse_estimate` (conditional on selection)
    and from :func:`cumulative_impact` (Bayesian shrink-then-sum). Prefer
    Bayesian cumulative impact when a prior is available; use this when the
    estimand is specifically the Airbnb process-level total.

    Parameters
    ----------
    effects, standard_errors : array-like
        Experiment lifts and SEs.
    alpha : float
        One-sided significance level for launch (default 0.05).
    alternative : {"greater", "less"}
        Launch direction (Airbnb uses ``"greater"``).
    n_bootstrap : int
        If > 0, parametric bootstrap percentile CI for ``hat{T}_A``.
    ci : float
        Bootstrap CI level (default 0.95).
    random_seed : int, optional
        Bootstrap RNG seed.

    Returns
    -------
    dict
        ``total`` (``hat{T}_A``), ``naive_total`` (``S_A``), ``conditional_total``
        (Zhong–Prentice style sum of conditional debiasings), ``bias_estimate``,
        ``n_selected``, ``selected_mask``, and optional ``ci_lower`` / ``ci_upper``.
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less' (Lee & Shen one-sided launch)")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be an integer >= 0")
    if n_bootstrap > 0 and not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    def _one_shot(x: np.ndarray) -> dict:
        z = float(norm.ppf(1.0 - alpha))
        b = z * s  # thresholds on the effect scale
        if alternative == "greater":
            selected = x > b
            # bias_i = SE * phi((SE*b_z - X)/SE) = SE * phi(z - X/SE)
            bias_terms = s * norm.pdf(z - x / s)
            # conditional: for selected only, SE * phi(z - X/SE) / (1 - Phi(z - X/SE))
            mills = norm.pdf(z - x / s) / np.maximum(norm.sf(z - x / s), 1e-300)
            cond_debias = np.where(selected, x - s * mills, 0.0)
        else:
            selected = x < -b
            bias_terms = s * norm.pdf(-z - x / s)
            mills = norm.pdf(-z - x / s) / np.maximum(norm.cdf(-z - x / s), 1e-300)
            cond_debias = np.where(selected, x + s * mills, 0.0)

        naive = float(np.sum(x[selected])) if selected.any() else 0.0
        bias_est = float(np.sum(bias_terms))
        # For "less", the bias identity is E[I(X<-b)(X-a)] = -SE*phi(...);
        # plug-in subtracts the signed bias so we add bias_terms when less.
        if alternative == "greater":
            total = naive - bias_est
        else:
            total = naive + bias_est
        return {
            "total": float(total),
            "naive_total": naive,
            "conditional_total": float(np.sum(cond_debias)),
            "bias_estimate": bias_est if alternative == "greater" else -bias_est,
            "n_selected": int(selected.sum()),
            "selected_mask": selected,
        }

    out = _one_shot(y)
    out["alternative"] = alternative
    out["alpha"] = float(alpha)

    if n_bootstrap > 0:
        rng = np.random.default_rng(random_seed)
        boots = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            x_b = rng.normal(y, s)
            boots[i] = _one_shot(x_b)["total"]
        gamma = 1.0 - ci
        out["ci_lower"] = float(np.quantile(boots, gamma / 2.0))
        out["ci_upper"] = float(np.quantile(boots, 1.0 - gamma / 2.0))
        out["n_bootstrap"] = n_bootstrap
    return out


def estimate_guardrail_rho(
    primary_effects,
    primary_ses,
    guardrail_effects,
    guardrail_ses,
    *,
    prior_mean_primary: float = 0.0,
    prior_mean_guard: float = 0.0,
) -> dict:
    """
    Method-of-moments estimate of Corr(true primary, true guardrail).

    Under independent sampling noise, ``Cov(X, G) = Cov(delta, gamma)``.
    Prior SDs are Paule–Mandel ``tau`` estimates; ``rho`` is clipped to
    ``(-0.999, 0.999)``. Requires at least 5 paired experiments.

    Returns
    -------
    dict
        ``rho``, ``tau_primary``, ``tau_guard``, ``cov``, ``n``.
    """
    x, sx = _validate_effects_ses(primary_effects, primary_ses)
    g, sg = _validate_effects_ses(guardrail_effects, guardrail_ses)
    if x.shape != g.shape:
        raise ValueError("primary and guardrail arrays must have the same length")
    if x.size < 5:
        raise ValueError("estimate_guardrail_rho requires at least 5 paired experiments")

    tau_p2 = _paule_mandel_tau2(x, sx**2, prior_mean=prior_mean_primary)
    tau_g2 = _paule_mandel_tau2(g, sg**2, prior_mean=prior_mean_guard)
    cov = float(np.mean((x - prior_mean_primary) * (g - prior_mean_guard)))
    tau_p = float(np.sqrt(tau_p2))
    tau_g = float(np.sqrt(tau_g2))
    if tau_p == 0.0 or tau_g == 0.0:
        warnings.warn(
            "estimate_guardrail_rho: one prior SD is 0; returning rho=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        rho = 0.0
    else:
        rho = float(np.clip(cov / (tau_p * tau_g), -0.999, 0.999))
    return {
        "rho": rho,
        "tau_primary": tau_p,
        "tau_guard": tau_g,
        "cov": cov,
        "n": int(x.size),
    }


def fit_normal_prior_map(
    effects,
    standard_errors,
    *,
    mu_prior_sd: float = 1.0,
    tau_prior_scale: float = 0.25,
) -> dict:
    """
    Datadog-style MAP for a hierarchical normal prior ``N(mu, tau^2)``.

    Rescales lifts to mean 0 / SD 1, places ``mu ~ N(0, mu_prior_sd^2)`` and
    ``tau ~ HalfCauchy(0, tau_prior_scale)`` on the rescaled scale, maximizes
    the marginal posterior (Nelder–Mead over ``(mu, log tau)``), then maps
    ``(mu, tau)`` back to the original scale. Returns a dict usable as
    ``prior=`` in :func:`cumulative_impact` / :func:`empirical_bayes_shrinkage`.

    Requires at least 5 estimates (same practical floor as Datadog Cumulative Impact).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 5:
        raise ValueError("fit_normal_prior_map requires at least 5 estimates")
    if not (np.isfinite(mu_prior_sd) and mu_prior_sd > 0):
        raise ValueError("mu_prior_sd must be positive and finite")
    if not (np.isfinite(tau_prior_scale) and tau_prior_scale > 0):
        raise ValueError("tau_prior_scale must be positive and finite")

    y_bar = float(np.mean(y))
    s_y = float(np.std(y, ddof=1))
    if s_y <= 0:
        # All effects identical: no heterogeneity to learn
        return {
            "prior_mean": y_bar,
            "tau2": 0.0,
            "prior_family": "normal",
            "method": "half_cauchy_map",
            "n": int(y.size),
        }

    y_t = (y - y_bar) / s_y
    s_t = s / s_y

    def neg_log_post(u: np.ndarray) -> float:
        mu, eta = float(u[0]), float(u[1])
        tau = np.exp(eta)
        marg_var = tau**2 + s_t**2
        nll = 0.5 * float(np.sum(np.log(2.0 * np.pi * marg_var) + (y_t - mu) ** 2 / marg_var))
        # mu ~ N(0, mu_prior_sd^2)
        nll += 0.5 * (mu / mu_prior_sd) ** 2 + np.log(mu_prior_sd * np.sqrt(2.0 * np.pi))
        # Half-Cauchy(0, scale): log dens = log(2/pi) - log(scale) - log(1+(tau/scale)^2)
        nll -= np.log(2.0 / np.pi) - np.log(tau_prior_scale) - np.log(1.0 + (tau / tau_prior_scale) ** 2)
        # Jacobian tau = exp(eta): subtract eta from nll <=> add eta to log posterior
        nll -= eta
        return float(nll)

    tau2_mom = _paule_mandel_tau2(y_t, s_t**2, prior_mean=0.0)
    eta0 = np.log(np.sqrt(tau2_mom)) if tau2_mom > 0 else np.log(0.1)
    res = minimize(neg_log_post, x0=np.array([0.0, eta0]), method="Nelder-Mead")
    if not res.success:
        warnings.warn(
            f"fit_normal_prior_map: optimizer reported failure ({res.message}); using last iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
    mu_t, eta = float(res.x[0]), float(res.x[1])
    tau_t = float(np.exp(eta))
    mu = mu_t * s_y + y_bar
    tau = tau_t * s_y
    return {
        "prior_mean": float(mu),
        "tau2": float(tau**2),
        "prior_family": "normal",
        "method": "half_cauchy_map",
        "n": int(y.size),
        "log_posterior": float(-res.fun),
    }
