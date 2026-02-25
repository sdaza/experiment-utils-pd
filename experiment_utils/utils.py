"""
Collection of helper methods. These should be fully generic and make no
assumptions about the format of input data.
"""

import logging

import numpy as np
import pandas as pd


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
            # Add tracking columns
            balance_df["treatment_group"] = treatment_val
            balance_df["control_group"] = control_val

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
                "mean_treated",
                "mean_control",
                "smd",
                "balance_flag",
                "treatment_group",
                "control_group",
            ]
        )

    return pd.concat(balance_results, ignore_index=True)


def balanced_random_assignment(
    df,
    seed=42,
    allocation_ratio=0.5,
    variants=None,
    balance_covariates=None,
    check_balance=True,
    check_balance_covariates=None,
    comparison=None,
    smd_threshold=0.1,
    n_bins=5,
):
    """
    Randomly assign units to variants with forced balance according to allocation ratios.
    Optionally stratify by covariates to ensure balance within strata.
    Optionally check and report covariate balance after assignment.

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
    balance_covariates : list, optional
        List of column names to use for stratification (block randomization).
        Continuous covariates are automatically binned into quantiles for stratification.
    check_balance : bool, optional
        Whether to check and print balance diagnostics after assignment (default True).
    check_balance_covariates : list, optional
        List of column names to check balance for after assignment. If None, falls back
        to balance_covariates. Use this to check balance on covariates that were not
        used for stratification.
    comparison : list[tuple], optional
        List of (variant1, variant2) tuples specifying which pairs to compare for balance.
        If None, performs all pairwise comparisons.
    smd_threshold : float, optional
        Threshold for standardized mean difference to flag imbalance (default 0.1).
        Covariates with |SMD| < threshold are considered balanced.
    n_bins : int, optional
        Number of quantile bins for continuous covariates in stratification (default 5).

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
        balance_covariates=['region', 'segment']
    )

    # Stratified assignment with continuous covariates (auto-binned)
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        balance_covariates=['age', 'previous_purchases']
    )

    # No stratification, but check balance on covariates
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        check_balance_covariates=['age', 'income', 'region']
    )

    # Stratify by region, but check balance on additional covariates
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'treatment'],
        balance_covariates=['region'],
        check_balance_covariates=['age', 'income', 'region']
    )
    """

    np.random.seed(seed)

    df_work = df.copy()
    use_stratification = balance_covariates is not None

    if use_stratification:
        # Auto-bin continuous covariates for stratification
        stratify_cols = []
        for cov in balance_covariates:
            is_numeric = pd.api.types.is_numeric_dtype(df_work[cov])
            n_unique = df_work[cov].nunique()
            if is_numeric and n_unique > n_bins * 2:
                bin_col = f"_bin_{cov}"
                df_work[bin_col] = pd.qcut(df_work[cov], q=n_bins, duplicates="drop")
                stratify_cols.append(bin_col)
            else:
                stratify_cols.append(cov)
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

    # Determine which covariates to check balance for
    covs_to_check = check_balance_covariates or (balance_covariates if use_stratification else None)

    if check_balance and covs_to_check is not None:
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
            balance_df = _check_assignment_balance(temp_df, "_assignment", covs_to_check, var1, var2, smd_threshold)

            print(f"\nComparison: {var1} vs {var2}")
            print("-" * 70)
            display_df = balance_df.copy()
            display_df.columns = ["covariate", f"mean_{var1}", f"mean_{var2}", "smd", "balanced"]
            display_df["balanced"] = display_df["balanced"].map({1: "✓", 0: "✗"})

            print(display_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else x))

            n_balanced = balance_df["balance_flag"].sum()
            n_total = len(balance_df)
            mean_abs_smd = balance_df["smd"].abs().mean()
            max_abs_smd = balance_df["smd"].abs().max()

            print(f"\nSummary: {n_balanced}/{n_total} covariates balanced (|SMD| < {smd_threshold})")
            print(f"Mean |SMD|: {mean_abs_smd:.4f}")
            print(f"Max |SMD|: {max_abs_smd:.4f}")

            if n_balanced < n_total:
                imbalanced = balance_df[balance_df["balance_flag"] == 0]["covariate"].tolist()
                print(f"⚠️  Imbalanced covariates: {', '.join(imbalanced)}")

        print("\n" + "=" * 70)

    return final_assignments
