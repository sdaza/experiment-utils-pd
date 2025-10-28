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


def balanced_random_assignment(df, seed=42, allocation_ratio=0.5, variants=None, balance_covariates=None):
    """
    Randomly assign units to variants with forced balance according to allocation ratios.
    Optionally stratify by covariates to ensure balance within strata.

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

    Returns
    -------
    pd.Series
        Series with variant assignments indexed by df.index

    Examples
    --------
    # Binary assignment (test/control) without stratification
    assignment = balanced_random_assignment(df, allocation_ratio=0.5)

    # Binary assignment with stratification by age_group and gender
    assignment = balanced_random_assignment(df, allocation_ratio=0.5, balance_covariates=['age_group', 'gender'])

    # Multiple variants with equal allocation
    assignment = balanced_random_assignment(df, variants=['control', 'variant_a', 'variant_b'])

    # Multiple variants with custom allocation and stratification
    assignment = balanced_random_assignment(
        df,
        variants=['control', 'variant_a', 'variant_b'],
        allocation_ratio={'control': 0.5, 'variant_a': 0.3, 'variant_b': 0.2},
        balance_covariates=['region']
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
    return pd.concat(assignments).sort_index()
