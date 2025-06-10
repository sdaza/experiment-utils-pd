"""
Entropy Balancing Implementation
This module implements an optimized version of entropy balancing for causal inference studies.
"""

import concurrent.futures

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def design_matrix_optimized(X_np, n_moments=2):
    """
    Creates a design matrix from the original covariate matrix X using
    vectorized NumPy operations.
    """
    # X_np is expected to be a NumPy array
    moment_matrices = []
    for p in range(1, n_moments + 1):
        Xp = X_np**p
        mean_Xp = np.mean(Xp, axis=0)
        std_Xp = np.std(Xp, axis=0)

        # Avoid division by zero for columns with constant value
        std_Xp[std_Xp == 0] = 1.0

        Xp_std = (Xp - mean_Xp) / std_Xp
        moment_matrices.append(Xp_std)

    return np.hstack(moment_matrices)


def eb_loss(pars, XD, bw, tars):
    """Loss function for entropy balancing."""
    XDP = -XD.dot(pars)
    # Numerical stability trick
    max_XDP = np.max(XDP)
    log_sum_exp = max_XDP + np.log(np.dot(bw, np.exp(XDP - max_XDP)))
    return log_sum_exp + np.dot(tars, pars)


class EntropyBalance:
    """
    Optimized Entropy Balancing class focused on performance.

    This class calculates balancing weights for causal inference studies
    by directly optimizing the dual problem of entropy balancing.
    It is designed to be fast for large datasets by leveraging NumPy.
    """

    def __init__(self):
        self.W = None

    def fit(self, X, TA, estimand="ATE", n_moments=2, base_weights=None):
        """
        Fits the entropy balancing weights.

        Args:
            X (pd.DataFrame or np.array): Covariate matrix.
            TA (pd.Series or np.array): Treatment assignment vector.
            estimand (str): The causal estimand ('ATE', 'ATT', 'ATC').
            n_moments (int): Number of moments to balance.
            base_weights (np.array, optional): Prior weights. Defaults to uniform weights.

        Returns:
            np.array: The final balancing weights.
        """

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        TA_np = TA.values if isinstance(TA, pd.Series) else TA
        XD = design_matrix_optimized(X_np, n_moments)

        if base_weights is None:
            base_weights = np.ones(X_np.shape[0])

        estimand_upper = estimand.upper()
        if estimand_upper == "ATE":
            targets = np.mean(XD, axis=0)
        elif estimand_upper == "ATT":
            targets = np.mean(XD[TA_np == 1], axis=0)
        elif estimand_upper == "ATC":
            targets = np.mean(XD[TA_np == 0], axis=0)
        else:
            raise ValueError("Invalid Estimand. Choose from 'ATE', 'ATT', 'ATC'.")

        # Use boolean masks for fast indexing
        control_mask = TA_np == 0
        treated_mask = TA_np == 1
        initial_params = np.zeros(XD.shape[1])

        optimizer_args = {"method": "L-BFGS-B", "options": {"maxiter": 200, "ftol": 1e-7, "disp": False}}

        def _eb_weights(pars, XD, bw):
            XDP = -XD.dot(pars)
            max_XDP = np.max(XDP)
            Q = bw * np.exp(XDP - max_XDP)
            return Q / np.sum(Q)

        self.W = base_weights.copy() if not np.shares_memory(base_weights, X_np) else np.array(base_weights)

        # Parallelize if both minimizations are needed (ATE)
        if estimand_upper == "ATE":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                fut0 = executor.submit(
                    minimize,
                    eb_loss,
                    initial_params,
                    args=(XD[control_mask], base_weights[control_mask], targets),
                    **optimizer_args,
                )
                fut1 = executor.submit(
                    minimize,
                    eb_loss,
                    initial_params,
                    args=(XD[treated_mask], base_weights[treated_mask], targets),
                    **optimizer_args,
                )
                res0 = fut0.result()
                res1 = fut1.result()
            self.W[control_mask] = _eb_weights(res0.x, XD[control_mask], base_weights[control_mask])
            self.W[treated_mask] = _eb_weights(res1.x, XD[treated_mask], base_weights[treated_mask])
        else:
            if estimand_upper == "ATT":
                res0 = minimize(
                    eb_loss,
                    initial_params,
                    args=(XD[control_mask], base_weights[control_mask], targets),
                    **optimizer_args,
                )
                self.W[control_mask] = _eb_weights(res0.x, XD[control_mask], base_weights[control_mask])
            if estimand_upper == "ATC":
                res1 = minimize(
                    eb_loss,
                    initial_params,
                    args=(XD[treated_mask], base_weights[treated_mask], targets),
                    **optimizer_args,
                )
                self.W[treated_mask] = _eb_weights(res1.x, XD[treated_mask], base_weights[treated_mask])

        return self.W
