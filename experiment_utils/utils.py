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

    # If no balance_covariates, create a dummy grouping column
    if balance_covariates is None:
        df = df.copy()
        df["_dummy_group"] = 1
        balance_covariates = ["_dummy_group"]

    # Perform assignment within each stratum
    assignments = []
    for _, group_df in df.groupby(balance_covariates):
        n = len(group_df)
        # Handle different input formats
        if variants is None:
            # Binary case: test/control
            if isinstance(allocation_ratio, dict):
                raise ValueError("When variants is None, allocation_ratio must be a float")
            n_test = int(n * allocation_ratio)
            n_control = n - n_test
            assignment = ["test"] * n_test + ["control"] * n_control
        else:
            # Multiple variants case
            if isinstance(allocation_ratio, dict):
                # Validate that ratios sum to 1.0
                total = sum(allocation_ratio.values())
                if not np.isclose(total, 1.0):
                    raise ValueError(f"Allocation ratios must sum to 1.0, got {total}")

                # Validate that all variants have ratios
                if set(variants) != set(allocation_ratio.keys()):
                    raise ValueError("Variants list must match allocation_ratio keys")

                # Allocate based on ratios
                assignment = []
                allocated = 0
                for _, variant in enumerate(variants[:-1]):
                    n_variant = int(n * allocation_ratio[variant])
                    assignment.extend([variant] * n_variant)
                    allocated += n_variant
                # Assign remaining to last variant to ensure exact total
                assignment.extend([variants[-1]] * (n - allocated))
            else:
                # Equal allocation among all variants
                n_per_variant = n // len(variants)
                remainder = n % len(variants)

                assignment = []
                for i, variant in enumerate(variants):
                    # Distribute remainder across first variants
                    n_variant = n_per_variant + (1 if i < remainder else 0)
                    assignment.extend([variant] * n_variant)

        np.random.shuffle(assignment)
        assignments.append(pd.Series(assignment, index=group_df.index))

    # Combine all assignments
    final_assignments = pd.concat(assignments).sort_index()

    # Check balance if requested and covariates were provided (not dummy group)
    if check_balance and balance_covariates is not None and balance_covariates != ["_dummy_group"]:
        # Create temp dataframe with assignments and covariates
        temp_df = df.copy()
        temp_df["_assignment"] = final_assignments

        # Determine which variants to compare
        unique_variants = final_assignments.unique()

        # Generate comparison pairs
        if comparison is None:
            # All pairwise comparisons
            from itertools import combinations

            comparison_pairs = list(combinations(unique_variants, 2))
        else:
            # Use provided comparison pairs
            comparison_pairs = comparison

        # Print header
        print("\n" + "=" * 70)
        print("Balance Check After Assignment")
        print("=" * 70)

        # Check balance for each comparison
        for var1, var2 in comparison_pairs:
            balance_df = _check_assignment_balance(
                temp_df, "_assignment", balance_covariates, var1, var2, smd_threshold
            )

            # Print results
            print(f"\nComparison: {var1} vs {var2}")
            print("-" * 70)

            # Format column names for display
            display_df = balance_df.copy()
            display_df.columns = ["covariate", f"mean_{var1}", f"mean_{var2}", "smd", "balanced"]
            display_df["balanced"] = display_df["balanced"].map({1: "✓", 0: "✗"})

            # Print table
            print(display_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else x))

            # Summary statistics
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
