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
        if df[cov].dtype == "object" or df[cov].dtype.name == "category":
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


def check_covariate_balance(
    data: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    categorical_covariates: dict[str, list] | None = None,
    min_binary_count: int = 5,
    threshold: float = 0.1,
    treatment_value: int | str = 1,
    control_value: int | str = 0,
    categorical_max_unique: int = 10,
    weights_col: str = "weights",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Check covariate balance between treatment and control groups with full preprocessing.

    This function performs comprehensive balance checking including:
    - Categorical variable identification and dummy creation
    - Covariate filtering (zero variance, minimum counts)
    - Standardization
    - SMD calculation with balance flags

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
    treatment_value : int | str, optional
        Value indicating treatment group (default 1)
    control_value : int | str, optional
        Value indicating control group (default 0)
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
        - mean_treated: float - weighted mean for treatment
        - mean_control: float - weighted mean for control
        - smd: float - standardized mean difference
        - balance_flag: int - 1 if balanced, 0 if imbalanced

    Examples
    --------
    >>> balance = check_covariate_balance(
    ...     data=df,
    ...     treatment_col='treatment',
    ...     covariates=['age', 'income', 'region'],
    ...     threshold=0.1
    ... )
    """
    import re

    if logger is None:
        logger = get_logger("BalanceChecker")

    # Make a copy to avoid modifying original data
    data = data.copy()

    # Filter to treatment and control groups
    data = data[data[treatment_col].isin([treatment_value, control_value])].copy()

    if data.empty:
        logger.warning("No data found for specified treatment and control values")
        return pd.DataFrame()

    # Check if we have both treatment and control groups
    n_treatment = (data[treatment_col] == treatment_value).sum()
    n_control = (data[treatment_col] == control_value).sum()

    if n_treatment == 0:
        logger.warning(f"No treatment units ({treatment_value}) found. Cannot check balance.")
        return pd.DataFrame()

    if n_control == 0:
        logger.warning(f"No control units ({control_value}) found. Cannot check balance.")
        return pd.DataFrame()

    # Recode treatment to binary
    data[treatment_col] = (data[treatment_col] == treatment_value).astype(int)

    # Add weights column if not present
    if weights_col not in data.columns:
        data[weights_col] = 1

    # Identify categorical covariates if not provided
    if categorical_covariates is None:
        categorical_covariates = {}
        for cov in covariates:
            if cov not in data.columns:
                continue

            is_object = pd.api.types.is_object_dtype(data[cov]) or isinstance(data[cov].dtype, pd.CategoricalDtype)
            is_low_cardinality_numeric = (
                pd.api.types.is_numeric_dtype(data[cov]) and 3 <= data[cov].nunique() <= categorical_max_unique
            )

            if is_object or is_low_cardinality_numeric:
                categories = sorted(data[cov].dropna().unique())
                categorical_covariates[cov] = categories

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


def balanced_random_assignment(
    df,
    seed=42,
    allocation_ratio=0.5,
    variants=None,
    balance_covariates=None,
    check_balance=True,
    comparison=None,
    smd_threshold=0.1,
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
        If float: proportion allocated to 'test' (remaining goes to 'control')
        If dict: mapping of variant names to their allocation ratios (must sum to 1.0)
        Default is 0.5 (50/50 split between test and control)
    variants : list, optional
        List of variant names. If provided, allocation_ratio should be a dict or
        units will be split equally among variants. If None, uses ['control', 'test']
    balance_covariates : list, optional
        List of column names to use for stratification. If provided, balanced assignment
        will be performed within each stratum defined by these columns.
    check_balance : bool, optional
        Whether to check and print balance diagnostics after assignment (default True).
        Only applies when balance_covariates is provided.
    comparison : list[tuple], optional
        List of (variant1, variant2) tuples specifying which pairs to compare for balance.
        If None, performs all pairwise comparisons. Example: [('test', 'control'), ('variant_a', 'control')]
    smd_threshold : float, optional
        Threshold for standardized mean difference to flag imbalance (default 0.1).
        Covariates with |SMD| < threshold are considered balanced.

    Returns
    -------
    pd.Series
        Series with variant assignments indexed by df.index

    Examples
    --------
    # Binary assignment (test/control) without stratification
    assignment = balanced_random_assignment(df, allocation_ratio=0.5)

    # Binary assignment with stratification and balance checking
    assignment = balanced_random_assignment(
        df,
        allocation_ratio=0.5,
        balance_covariates=['age', 'prior_purchase']
    )

    # Multiple variants with custom comparisons
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'variant_a', 'variant_b'],
        balance_covariates=['age', 'region'],
        comparison=[('variant_a', 'control'), ('variant_b', 'control')]
    )

    # Disable balance checking
    assignment = balanced_random_assignment(
        df,
        balance_covariates=['age'],
        check_balance=False
    )
    """

    np.random.seed(seed)

    if balance_covariates is None:
        df = df.copy()
        df["_dummy_group"] = 1
        balance_covariates = ["_dummy_group"]

    assignments = []
    for _, group_df in df.groupby(balance_covariates):
        n = len(group_df)
        if variants is None:
            if isinstance(allocation_ratio, dict):
                raise ValueError("When variants is None, allocation_ratio must be a float")
            n_test = int(n * allocation_ratio)
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
                    n_variant = int(n * allocation_ratio[variant])
                    assignment.extend([variant] * n_variant)
                    allocated += n_variant
                assignment.extend([variants[-1]] * (n - allocated))
            else:
                n_per_variant = n // len(variants)
                remainder = n % len(variants)

                assignment = []
                for i, variant in enumerate(variants):
                    n_variant = n_per_variant + (1 if i < remainder else 0)
                    assignment.extend([variant] * n_variant)

        np.random.shuffle(assignment)
        assignments.append(pd.Series(assignment, index=group_df.index))

    final_assignments = pd.concat(assignments).sort_index()

    if check_balance and balance_covariates is not None and balance_covariates != ["_dummy_group"]:
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
            balance_df = _check_assignment_balance(
                temp_df, "_assignment", balance_covariates, var1, var2, smd_threshold
            )

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
