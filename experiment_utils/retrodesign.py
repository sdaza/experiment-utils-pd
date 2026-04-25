"""Retrodesign analysis mixin for ExperimentAnalyzer."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import log_and_raise_error


class RetrodesignMixin:
    """Mixin providing retrodesign (power, Type S, Type M) analysis via PowerSim."""

    def _calculate_retrodesign_via_powersim(
        self,
        row: pd.Series,
        true_effect: float,
        nsim: int,
        alpha: float,
        seed: int | None = None,
    ) -> dict[str, float] | None:
        """
        Calculate retrodesign using PowerSim and the observed standard error.

        The observed SE already encodes all the complexity of the study design
        (model type, covariates, binary/count/continuous outcome, etc.), so we
        back-calculate the per-group SD and run a normal-approximation simulation.
        This is accurate for all model types at the large sample sizes typical of
        online experiments.

        Parameters
        ----------
        row : pd.Series
            Result row from get_effects()
        true_effect : float
            True effect size (same scale as absolute_effect / standard_error)
        nsim : int
            Number of Monte Carlo simulations for type_s / type_m estimation
        alpha : float
            Significance level
        seed : int, optional
            Random seed

        Returns
        -------
        dict or None
        """
        from .power_sim import PowerSim

        outcome = row.get("outcome", "unknown")
        n_treatment = int(row["treatment_units"])
        n_control = int(row["control_units"])

        se = row.get("standard_error")
        if pd.isna(se) or se is None or se <= 0:
            self._logger.warning(f"No valid standard_error for outcome '{outcome}'. Cannot compute retrodesign.")
            return None

        se = float(se)

        # SE = sd * sqrt(1/n_t + 1/n_c)  →  sd = SE / sqrt(1/n_t + 1/n_c)
        se_factor = np.sqrt(1.0 / n_treatment + 1.0 / n_control)
        control_std = se / se_factor

        try:
            power_sim = PowerSim(
                metric="average",
                variants=1,
                nsim=nsim,
                alpha=alpha,
                early_stopping=True,
                early_stopping_precision=0.01,
            )
            retro_result = power_sim.simulate_retrodesign(
                true_effect=true_effect,
                sample_size=[n_control, n_treatment],
                baseline=0.0,
                standard_deviation=[control_std],
                nsim=nsim,
                random_seed=seed,
            )
            return {
                "power": retro_result["power"].iloc[0],
                "type_s_error": retro_result["type_s_error"].iloc[0],
                "type_m_error": retro_result["exaggeration_ratio"].iloc[0],
                "relative_bias": retro_result["relative_bias"].iloc[0],
                "retrodesign_method": "powersim",
            }

        except Exception as e:
            self._logger.warning(
                f"PowerSim retrodesign failed for outcome '{outcome}': {e} "
                f"(true_effect={true_effect}, se={se:.5f}, n=[{n_control}, {n_treatment}])"
            )
            return None

    def calculate_retrodesign(
        self,
        true_effect: float | dict | None = None,
        alpha: float | None = None,
        outcomes: list[str] | None = None,
        experiments: list | None = None,
        nsim: int = 5000,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        Calculate retrodesign metrics (power, Type S, Type M) via PowerSim.

        Uses the observed standard error from get_effects() to back-calculate a
        per-group standard deviation and runs a normal-approximation simulation via
        PowerSim. This approach works for all model types (OLS, logistic, Poisson,
        negative binomial, Cox) because the observed SE already encodes all model
        complexity (covariates, link functions, etc.).

        Retrodesign analysis calculates:
        - Power: probability of achieving statistical significance given true effect
        - Type S error: probability of getting the wrong sign (sign error rate)
        - Type M error: expected exaggeration ratio (magnitude error)
        - Relative bias: expected bias ratio preserving signs (Jaksic et al. 2026)

        Parameters
        ----------
        true_effect : float, dict, or None
            The hypothesized true effect size. Scale depends on model type:
            - OLS: Mean difference (e.g., 5.0 for $5 increase)
            - Logistic (marginal effects): Probability change (e.g., 0.05 for 5pp)
            - Cox: log(HR) (e.g., -0.223 for HR=0.8)
            - Poisson / NegBin (marginal): Count change (e.g., 2.0 for +2 events)

            Can be:
            - A single float to apply to all results
            - A dict mapping outcome names to their true effects:
              {'revenue': 5.0, 'clicked': 0.05, 'time_to_churn': -0.223}
            - A dict mapping (treatment_group, control_group) tuples to true effects
            - A dict mapping (outcome, treatment_group, control_group) tuples
            - None to use the observed effect as the assumed true effect (conservative)
        alpha : float, optional
            Significance level. If None, uses self._alpha
        outcomes : list of str, optional
            Filter to specific outcomes. If None, uses all outcomes
        experiments : list, optional
            Filter to specific experiments. If None, uses all experiments
        nsim : int, optional
            Number of Monte Carlo draws for type_s / type_m estimation (default: 5000).
            Use 1000–2000 for quick estimates, 5000+ for final results.
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Original results with added columns:
            - true_effect: The assumed true effect size
            - power: Probability of significance given true effect
            - type_s_error: Probability of wrong sign when significant
            - type_m_error: Expected exaggeration ratio (absolute values)
            - relative_bias: Expected bias ratio preserving signs
            - trimmed_abs_effect: Bias-corrected effect estimate
              (absolute_effect / relative_bias). NaN when relative_bias < 1
              or type_s_error ≥ 0.10.
            - retrodesign_method: Always "powersim"

        Examples
        --------
        >>> analyzer.get_effects()
        >>> retro = analyzer.calculate_retrodesign(nsim=2000)
        >>> retro = analyzer.calculate_retrodesign(true_effect={'revenue': 5.0})
        """
        if self._results is None:
            log_and_raise_error(self._logger, "Run get_effects() first.")

        df = self._results.copy()
        alpha_val = alpha if alpha is not None else self._alpha

        mask = pd.Series([True] * len(df))
        if outcomes is not None:
            mask &= df["outcome"].isin(outcomes)
        if experiments is not None:
            group_cols = self._experiment_identifier or ["experiment"]
            for col, vals in zip(group_cols, zip(*experiments, strict=False), strict=False):
                mask &= df[col].isin(vals)

        df_filtered = df[mask].copy()

        # filter to only statistically significant effects
        if "pvalue_mcp" in df_filtered.columns and "stat_significance_mcp" in df_filtered.columns:
            self._logger.info("Using MPC-adjusted significance for retrodesign filtering")
            sig_mask = df_filtered["stat_significance_mcp"] == 1
        else:
            sig_mask = df_filtered["stat_significance"] == 1

        n_total = len(df_filtered)
        n_significant = sig_mask.sum()
        n_filtered = n_total - n_significant

        if n_filtered > 0:
            self._logger.info(
                f"Retrodesign: Filtering to {n_significant}/{n_total} statistically significant effects "
                f"({n_filtered} non-significant effects excluded)"
            )

        df_filtered = df_filtered[sig_mask].copy()

        if df_filtered.empty:
            self._logger.warning(
                "No statistically significant effects found. Retrodesign requires significant effects. "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        # drop existing retrodesign columns if they exist (from previous calls)
        retro_cols = ["true_effect", "power", "type_s_error", "type_m_error", "relative_bias", "trimmed_abs_effect"]
        existing_retro_cols = [col for col in retro_cols if col in df_filtered.columns]
        if existing_retro_cols:
            df_filtered = df_filtered.drop(columns=existing_retro_cols)

        # determine true effect for each row
        if true_effect is None:
            self._logger.info(
                "No true_effect specified. Using observed effects as assumed true effects (conservative approach)."
            )
            df_filtered["true_effect"] = df_filtered["absolute_effect"]
        elif isinstance(true_effect, dict):
            sample_key = next(iter(true_effect.keys()))

            if isinstance(sample_key, tuple):
                if len(sample_key) == 2:
                    self._logger.info(
                        f"Using true_effect mapping by (treatment_group, control_group): {len(true_effect)} comparison(s)"  # noqa: E501
                    )
                    df_filtered["true_effect"] = df_filtered.apply(
                        lambda row: true_effect.get(
                            (row["treatment_group"], row["control_group"]),
                            row["absolute_effect"],
                        ),
                        axis=1,
                    )
                elif len(sample_key) == 3:
                    self._logger.info(
                        f"Using true_effect mapping by (outcome, treatment_group, control_group): {len(true_effect)} combination(s)"  # noqa: E501
                    )
                    df_filtered["true_effect"] = df_filtered.apply(
                        lambda row: true_effect.get(
                            (row["outcome"], row["treatment_group"], row["control_group"]),
                            row["absolute_effect"],
                        ),
                        axis=1,
                    )
                else:
                    log_and_raise_error(
                        self._logger,
                        f"Dict keys must be tuples of length 2 (treatment_group, control_group) "
                        f"or 3 (outcome, treatment_group, control_group), got length {len(sample_key)}",
                    )
            else:
                self._logger.info(f"Using true_effect mapping by outcome: {true_effect}")
                df_filtered["true_effect"] = df_filtered["outcome"].map(true_effect)
                missing = df_filtered["true_effect"].isna()
                if missing.any():
                    skipped = df_filtered.loc[missing, "outcome"].unique().tolist()
                    self._logger.info(
                        f"Retrodesign: skipping {missing.sum()} row(s) for outcomes not in true_effect dict: {skipped}"
                    )
                df_filtered = df_filtered.dropna(subset=["true_effect"])
        else:
            self._logger.info(f"Using single true_effect value for all comparisons: {true_effect}")
            df_filtered["true_effect"] = true_effect

        if seed is not None:
            np.random.seed(seed)

        import time

        start_time = time.time()

        results = []
        for _idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered), desc="Retrodesign", unit="outcome"):
            te = row["true_effect"]

            if pd.isna(te):
                results.append(
                    {
                        "power": np.nan,
                        "type_s_error": np.nan,
                        "type_m_error": np.nan,
                        "relative_bias": np.nan,
                        "retrodesign_method": np.nan,
                    }
                )
                continue

            retro_result = self._calculate_retrodesign_via_powersim(
                row=row,
                true_effect=te,
                nsim=nsim,
                alpha=alpha_val,
                seed=seed,
            )
            if retro_result is None:
                results.append(
                    {
                        "power": np.nan,
                        "type_s_error": np.nan,
                        "type_m_error": np.nan,
                        "relative_bias": np.nan,
                        "retrodesign_method": np.nan,
                    }
                )
                continue

            results.append(retro_result)

        elapsed_time = time.time() - start_time
        self._logger.debug(
            f"Retrodesign completed in {elapsed_time:.2f}s ({elapsed_time / max(len(df_filtered), 1):.2f}s per row)"
        )

        retro_df = pd.DataFrame(results, index=df_filtered.index)
        df_filtered = pd.concat([df_filtered, retro_df], axis=1)

        # trimmed_abs_effect: observed effect deflated by relative_bias to approximate the true effect.
        # relative_bias = E[estimated / true] among significant estimates, so dividing the
        # observed effect by it yields a sign-preserving, bias-corrected estimate.
        # Set to NaN when type_s_error >= 0.10 (high sign error rates contaminate relative_bias)
        # or when relative_bias < 1 (correction would inflate rather than deflate the observed effect).
        TYPE_S_TRIM_THRESHOLD = 0.10
        rb = df_filtered["relative_bias"]
        type_s = df_filtered["type_s_error"]

        unreliable_mask = (type_s >= TYPE_S_TRIM_THRESHOLD) | (rb < 1)

        if (type_s >= TYPE_S_TRIM_THRESHOLD).any():
            flagged_outcomes = df_filtered.loc[type_s >= TYPE_S_TRIM_THRESHOLD, "outcome"].tolist()
            self._logger.warning(
                f"trimmed_abs_effect set to NaN for {(type_s >= TYPE_S_TRIM_THRESHOLD).sum()} row(s) where "
                f"type_s_error >= {TYPE_S_TRIM_THRESHOLD} (outcomes: {flagged_outcomes}). "
                "High sign-error rates make the bias correction unreliable."
            )

        df_filtered["trimmed_abs_effect"] = df_filtered["absolute_effect"].where(
            rb.isna() | (rb == 0) | unreliable_mask,
            df_filtered["absolute_effect"] / rb,
        )
        df_filtered.loc[unreliable_mask, "trimmed_abs_effect"] = np.nan

        if df_filtered["trimmed_abs_effect"].isna().all():
            df_filtered = df_filtered.drop(columns=["trimmed_abs_effect"])

        # Drop internal columns not meant for display
        internal_cols = ["se_intercept", "cov_coef_intercept", "control_std", "alpha_param"]
        df_filtered = df_filtered.drop(columns=[c for c in internal_cols if c in df_filtered.columns])

        return df_filtered
