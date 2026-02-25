"""Retrodesign analysis mixin for ExperimentAnalyzer."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .utils import log_and_raise_error


class RetrodesignMixin:
    """Mixin providing retrodesign (power, Type S, Type M) analysis."""

    def _can_use_analytical(
        self,
        row: pd.Series,
        method: str,
    ) -> bool:
        """
        Determine if we can use analytical/PowerSim approach for this result.

        Parameters
        ----------
        row : pd.Series
            Result row from get_effects()
        method : str
            User-specified method ("auto", "analytical", or "simulation")

        Returns
        -------
        bool
            True if analytical/PowerSim approach should be used
        """
        if method == "simulation":
            return False
        if method == "analytical":
            return True

        model_type = row.get("model_type", "ols")
        effect_type = row.get("effect_type", "mean_difference")
        complex_models = {"cox", "poisson", "negative_binomial"}

        if model_type == "logistic" and effect_type == "log_odds":
            return False
        if model_type in complex_models:
            return False

        has_fitted_models = self._store_fitted_models and len(self._fitted_models) > 0
        has_covariates = self._regression_covariates and len(self._regression_covariates) > 0

        if has_fitted_models and has_covariates:
            return False

        return True

    def _calculate_retrodesign_via_powersim(
        self,
        row: pd.Series,
        true_effect: float,
        nsim: int,
        alpha: float,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Calculate retrodesign using PowerSim for fast approximation.

        Parameters
        ----------
        row : pd.Series
            Result row from get_effects()
        true_effect : float
            True effect size
        nsim : int
            Number of simulations
        alpha : float
            Significance level
        seed : int, optional
            Random seed

        Returns
        -------
        dict
            Dictionary with power, type_s_error, type_m_error, relative_bias
        """
        from .power_sim import PowerSim

        n_treatment = int(row["treatment_units"])
        n_control = int(row["control_units"])
        model_type = row.get("model_type", "ols")
        effect_type = row.get("effect_type", "mean_difference")

        control_value = row.get("control_value")
        if pd.isna(control_value) or control_value is None:
            control_value = row.get("control_mean", row.get("control_rate", 0))

        if model_type == "logistic":
            control_value = float(control_value) if not pd.isna(control_value) else 0.1
        elif model_type in ["poisson", "negative_binomial"]:
            control_value = float(control_value) if not pd.isna(control_value) else 1.0
        else:
            control_value = float(control_value) if not pd.isna(control_value) else 0.0

        # Resolve control_std before metric selection so it can inform the heuristic
        control_std = row.get("control_std")
        if pd.isna(control_std) or control_std is None or control_std <= 0:
            control_std = None  # will be set to metric-aware default after metric selection

        if model_type == "logistic" or (model_type == "ols" and effect_type == "probability_change"):
            metric = "proportion"
            baseline = control_value
        elif model_type in ["poisson", "negative_binomial"]:
            metric = "count"
            baseline = control_value
            if baseline <= 0:
                baseline = 1.0
        elif model_type == "ols" and 0 <= control_value <= 1 and effect_type == "mean_difference":
            # Check if variance is consistent with binary/proportion data.
            # For a true proportion, SD should be close to sqrt(p*(1-p)).
            # If actual SD is much larger, this is likely count or continuous data
            # whose mean happens to fall in [0, 1].
            expected_binomial_sd = np.sqrt(control_value * (1 - control_value))
            if control_std is not None and control_std > 0 and control_std > expected_binomial_sd * 1.5:
                metric = "average"
            else:
                metric = "proportion"
            baseline = control_value
        else:
            metric = "average"
            baseline = control_value

        # Set metric-aware fallback for control_std if it was missing/invalid
        if control_std is None or control_std <= 0:
            if metric == "proportion":
                control_std = np.sqrt(control_value * (1 - control_value))
                if control_std < 0.01:
                    control_std = 0.01
            elif metric == "count":
                # Poisson SD as minimum; actual SD may be higher with overdispersion
                control_std = np.sqrt(max(control_value, 0.1))
            else:
                control_std = 1.0

        try:
            power_sim = PowerSim(
                metric=metric,
                variants=1,
                nsim=nsim,
                alpha=alpha,
                early_stopping=True,
                early_stopping_precision=0.01,
            )

            if metric == "proportion":
                baseline = np.clip(baseline, 0.001, 0.999)
                retro_result = power_sim.simulate_retrodesign(
                    true_effect=true_effect,
                    sample_size=[n_control, n_treatment],
                    baseline=baseline,
                    nsim=nsim,
                )
            elif metric == "count":
                baseline = max(0.1, baseline)
                retro_result = power_sim.simulate_retrodesign(
                    true_effect=true_effect,
                    sample_size=[n_control, n_treatment],
                    baseline=baseline,
                    nsim=nsim,
                )
            else:
                retro_result = power_sim.simulate_retrodesign(
                    true_effect=true_effect,
                    sample_size=[n_control, n_treatment],
                    baseline=baseline,
                    standard_deviation=[control_std],
                    nsim=nsim,
                )

            return {
                "power": retro_result["power"].iloc[0],
                "type_s_error": retro_result["type_s_error"].iloc[0],
                "type_m_error": retro_result["exaggeration_ratio"].iloc[0],
                "relative_bias": retro_result["relative_bias"].iloc[0],
                "retrodesign_method": "powersim",
            }

        except Exception as e:
            import traceback

            self._logger.debug(
                f"PowerSim retrodesign failed for outcome '{row.get('outcome', 'unknown')}': {e}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Parameters: metric={metric}, baseline={baseline}, true_effect={true_effect}, "
                f"sample_sizes=[{n_control}, {n_treatment}]"
            )
            self._logger.warning(
                f"PowerSim retrodesign failed for outcome '{row.get('outcome', 'unknown')}': {e}. "
                f"Falling back to full simulation. "
                f"(Parameters: metric={metric}, baseline={baseline}, true_effect={true_effect}, "
                f"sample_sizes=[{n_control}, {n_treatment}])"
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
        method: str = "auto",
        use_power_sim: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate retrodesign metrics using simulation (works for all model types).

        Uses simulation-based approach that:
        - Generates synthetic data matching your study design
        - Fits the actual statistical models (OLS, Cox, logistic, etc.)
        - Calculates empirical power, Type S, and Type M errors
        - Works correctly for all effect scales (log odds, log hazard ratios, etc.)

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
            - Logistic (log odds): log(OR) (e.g., 0.182 for OR=1.2)
            - Cox: log(HR) (e.g., -0.223 for HR=0.8)
            - Poisson (marginal): Count change (e.g., 2.0 for +2 events)
            - Poisson (log scale): log(IRR) (e.g., 0.182 for IRR=1.2)

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
            Number of simulations (default: 5000). More simulations = more accurate
            but slower. Use 1000-2000 for quick estimates, 5000+ for final results.
        seed : int, optional
            Random seed for reproducibility
        method : str, optional
            Calculation method. Options:
            - "auto" (default): Automatically choose between analytical and simulation.
            - "analytical": Force use of fast analytical/PowerSim approach.
            - "simulation": Force use of full simulation with fitted models.
        use_power_sim : bool, optional
            Whether to use PowerSim for fast retrodesign when applicable (default: True).

        Returns
        -------
        pd.DataFrame
            Original results with added columns:
            - true_effect: The assumed true effect size
            - power: Empirical probability of significance given true effect
            - type_s_error: Probability of wrong sign when significant
            - type_m_error: Expected exaggeration ratio using absolute values
            - relative_bias: Expected bias ratio preserving signs
            - trimmed_effect: Bias-corrected effect estimate (absolute_effect / relative_bias).
              Deflates the observed effect by the sign-preserving exaggeration factor to
              approximate the true effect. NaN when relative_bias is unavailable or zero.
            - retrodesign_method: Method used ("powersim" or "simulation")

        Examples
        --------
        >>> analyzer.get_effects()
        >>> retro = analyzer.calculate_retrodesign(nsim=2000)
        >>> retro = analyzer.calculate_retrodesign(true_effect=0.05, method="auto")
        >>> retro = analyzer.calculate_retrodesign(true_effect=0.05, method="simulation")
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
        retro_cols = ["true_effect", "power", "type_s_error", "type_m_error", "relative_bias", "trimmed_effect"]
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
                df_filtered.loc[:, "true_effect"] = df_filtered["true_effect"].fillna(df_filtered["absolute_effect"])
        else:
            self._logger.info(f"Using single true_effect value for all comparisons: {true_effect}")
            df_filtered["true_effect"] = true_effect

        if seed is not None:
            np.random.seed(seed)

        if method not in ["auto", "analytical", "simulation"]:
            log_and_raise_error(self._logger, f"method must be 'auto', 'analytical', or 'simulation', got '{method}'")

        method_counts = {"powersim": 0, "simulation": 0}
        for _idx, row in df_filtered.iterrows():
            if use_power_sim and self._can_use_analytical(row, method):
                method_counts["powersim"] += 1
            else:
                method_counts["simulation"] += 1

        import time

        start_time = time.time()

        if method_counts["powersim"] > 0 and method_counts["simulation"] > 0:
            self._logger.debug(
                f"Retrodesign: {method_counts['powersim']} via PowerSim, "
                f"{method_counts['simulation']} via simulation (nsim={nsim})"
            )
        elif method_counts["simulation"] > 0:
            use_fitted_models = self._store_fitted_models and len(self._fitted_models) > 0
            if not use_fitted_models:
                has_covariates = self._regression_covariates and len(self._regression_covariates) > 0
                has_adjustment = self._adjustment is not None
                has_clustering = self._cluster_col is not None

                if has_covariates or has_adjustment or has_clustering:
                    if not self._store_fitted_models:
                        self._logger.warning(
                            "Retrodesign: using simplified models (no covariates). "
                            "Enable store_fitted_models=True for accurate results with covariates."
                        )
                    else:
                        self._logger.warning(
                            "Retrodesign: using simplified models. "
                            "Fitted models not available - ensure get_effects() was called first."
                        )

        results = []
        for _idx, row in df_filtered.iterrows():
            te = row["true_effect"]
            model_type = row.get("model_type", "ols")
            effect_type = row.get("effect_type", "mean_difference")
            outcome = row.get("outcome", "unknown")

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

            use_analytical = use_power_sim and self._can_use_analytical(row, method)

            if use_analytical:
                retro_result = self._calculate_retrodesign_via_powersim(
                    row=row,
                    true_effect=te,
                    nsim=nsim,
                    alpha=alpha_val,
                    seed=seed,
                )
                if retro_result is None:
                    self._logger.debug(f"PowerSim failed for outcome '{outcome}', falling back to full simulation")
                    use_analytical = False

            if not use_analytical:
                use_fitted_models = self._store_fitted_models and len(self._fitted_models) > 0
                fitted_model = None

                if use_fitted_models:
                    experiment_tuple = (
                        tuple(row[col] for col in self._experiment_identifier) if self._experiment_identifier else None
                    )
                    comparison_tuple = (row["treatment_group"], row["control_group"])

                    if (
                        experiment_tuple in self._fitted_models
                        and comparison_tuple in self._fitted_models[experiment_tuple]
                        and outcome in self._fitted_models[experiment_tuple][comparison_tuple]
                    ):
                        fitted_obj = self._fitted_models[experiment_tuple][comparison_tuple][outcome]
                        if isinstance(fitted_obj, dict):
                            row_model_type = row.get("model_type", "ols")
                            fitted_model = fitted_obj.get(row_model_type)
                            if fitted_model is None:
                                self._logger.warning(
                                    f"No fitted model found for model_type '{row_model_type}' in outcome '{outcome}'. "
                                    f"Available models: {list(fitted_obj.keys())}"
                                )
                        else:
                            fitted_model = fitted_obj

                retro_result = self._simulate_retrodesign_single(
                    row=row,
                    true_effect=te,
                    nsim=nsim,
                    alpha=alpha_val,
                    model_type=model_type,
                    effect_type=effect_type,
                    fitted_model=fitted_model,
                )
                if "retrodesign_method" not in retro_result:
                    retro_result["retrodesign_method"] = "simulation"

            results.append(retro_result)

        elapsed_time = time.time() - start_time
        self._logger.debug(
            f"Retrodesign completed: {elapsed_time:.2f}s total, {elapsed_time / len(df_filtered):.2f}s per outcome"
        )

        retro_df = pd.DataFrame(results, index=df_filtered.index)
        df_filtered = pd.concat([df_filtered, retro_df], axis=1)

        # trimmed_effect: observed effect deflated by relative_bias to approximate the true effect.
        # relative_bias = E[estimated / true] among significant estimates, so dividing the
        # observed effect by it yields a sign-preserving, bias-corrected estimate.
        rb = df_filtered["relative_bias"]
        df_filtered["trimmed_effect"] = df_filtered["absolute_effect"].where(
            rb.isna() | (rb == 0), df_filtered["absolute_effect"] / rb
        )

        # Drop internal columns not meant for display
        internal_cols = ["se_intercept", "cov_coef_intercept", "control_std"]
        df_filtered = df_filtered.drop(columns=[c for c in internal_cols if c in df_filtered.columns])

        return df_filtered

    def _simulate_retrodesign_single(
        self,
        row: pd.Series,
        true_effect: float,
        nsim: int,
        alpha: float,
        model_type: str,
        effect_type: str,
        fitted_model=None,
    ) -> dict[str, float]:
        """
        Simulate retrodesign for a single result using the same model that produced it.

        Parameters
        ----------
        row : pd.Series
            Result row containing sample sizes, baseline values, etc.
        true_effect : float
            True effect size (on same scale as absolute_effect)
        nsim : int
            Number of simulations
        alpha : float
            Significance level
        model_type : str
            Model type ("ols", "logistic", "cox", "poisson", etc.)
        effect_type : str
            Effect type ("mean_difference", "log_odds", "hazard_ratio", etc.)
        fitted_model : optional
            Fitted model object for realistic simulation

        Returns
        -------
        dict
            Dictionary with power, type_s_error, type_m_error, relative_bias
        """
        n_treatment = int(row["treatment_units"])
        n_control = int(row["control_units"])
        control_value = row.get("control_value", 0)
        outcome = row.get("outcome", "unknown")

        if model_type == "logistic" and effect_type == "probability_change":
            if control_value < 0.05:
                self._logger.warning(
                    f"Low baseline rate ({control_value:.3f}) for outcome '{outcome}'. "
                    f"Logistic simulation may have convergence issues. "
                    f"Recommendation: Use method='analytical' for faster/more stable results, "
                    f"or consider Poisson model for rare events."
                )

        significant_effects = []
        n_significant = 0
        n_wrong_sign = 0
        n_successful = 0

        for _ in range(nsim):
            if fitted_model is not None:
                sim_data = self._generate_sim_data_from_model(
                    fitted_model=fitted_model,
                    true_effect=true_effect,
                    n_treatment=n_treatment,
                    n_control=n_control,
                    model_type=model_type,
                    effect_type=effect_type,
                )
            else:
                sim_data = self._generate_sim_data(
                    n_treatment=n_treatment,
                    n_control=n_control,
                    true_effect=true_effect,
                    control_value=control_value,
                    model_type=model_type,
                    effect_type=effect_type,
                    row=row,
                    fitted_model=fitted_model,
                )

            if sim_data is None:
                continue

            try:
                if fitted_model is not None:
                    sim_result = self._fit_sim_model_from_spec(
                        data=sim_data,
                        fitted_model=fitted_model,
                        model_type=model_type,
                        effect_type=effect_type,
                    )
                else:
                    sim_result = self._fit_sim_model(
                        data=sim_data,
                        model_type=model_type,
                        effect_type=effect_type,
                    )

                if sim_result is None:
                    continue

                n_successful += 1

                if sim_result["pvalue"] < alpha:
                    n_significant += 1
                    significant_effects.append(sim_result["absolute_effect"])
                    if true_effect != 0:
                        if np.sign(sim_result["absolute_effect"]) != np.sign(true_effect):
                            n_wrong_sign += 1

            except Exception:
                continue

        failure_rate = 1 - (n_successful / nsim) if nsim > 0 else 0
        if failure_rate > 0.3:
            outcome = row.get("outcome", "unknown")

            if model_type == "logistic":
                message = (
                    f"Retrodesign: {failure_rate:.0%} of simulations failed for outcome '{outcome}'. "
                    f"This often happens with very low baseline rates (<5%) in logistic models. "
                    f"Consider: (1) method='analytical', (2) increasing sample size, "
                    f"or (3) Poisson model for rare events."
                )
            elif model_type == "negative_binomial":
                message = (
                    f"Retrodesign: {failure_rate:.0%} of simulations failed for outcome '{outcome}'. "
                    f"Common causes: extreme dispersion, low baseline rate, small sample size. "
                    f"Consider Poisson if variance approximately equals mean."
                )
            elif model_type == "poisson":
                message = (
                    f"Retrodesign: {failure_rate:.0%} of simulations failed for outcome '{outcome}'. "
                    f"Consider: (1) increasing sample size, (2) negative_binomial model."
                )
            elif model_type == "cox":
                message = (
                    f"Retrodesign: {failure_rate:.0%} of simulations failed for outcome '{outcome}'. "
                    f"Common causes: insufficient events, complex covariate structures. "
                    f"Consider simplifying model."
                )
            else:
                message = (
                    f"Retrodesign: {failure_rate:.0%} of simulations failed for outcome '{outcome}' "
                    f"(model: {model_type}). Consider increasing sample size."
                )
            self._logger.warning(message)

        if n_successful > 0:
            power = n_significant / n_successful
        else:
            power = np.nan

        if n_significant > 0:
            type_s_error = n_wrong_sign / n_significant
            if true_effect != 0 and len(significant_effects) > 0:
                type_m_error = np.mean([abs(eff / true_effect) for eff in significant_effects])
                relative_bias = np.mean([eff / true_effect for eff in significant_effects])
            else:
                type_m_error = np.nan
                relative_bias = np.nan
        else:
            type_s_error = np.nan
            type_m_error = np.nan
            relative_bias = np.nan

        return {
            "power": power,
            "type_s_error": type_s_error,
            "type_m_error": type_m_error,
            "relative_bias": relative_bias,
        }

    def _get_nb_dispersion_parameter(
        self,
        row: pd.Series,
        fitted_model=None,
    ) -> float:
        """
        Get dispersion parameter (alpha) for negative binomial simulation.

        Parameters
        ----------
        row : pd.Series
            Result row containing outcome statistics
        fitted_model : model object, optional
            Fitted negative binomial model (preferred source)

        Returns
        -------
        float
            Dispersion parameter (alpha)
        """
        if fitted_model is not None:
            if hasattr(fitted_model, "params") and "alpha" in fitted_model.params:
                alpha = fitted_model.params["alpha"]
                if alpha > 0 and not np.isnan(alpha):
                    self._logger.debug(
                        f"Using dispersion alpha={alpha:.4f} from fitted model for "
                        f"outcome '{row.get('outcome', 'unknown')}'"
                    )
                    return float(alpha)

        if "alpha_param" in row.index and not pd.isna(row["alpha_param"]):
            alpha = row["alpha_param"]
            if alpha > 0:
                self._logger.debug(
                    f"Using stored dispersion alpha={alpha:.4f} from results for "
                    f"outcome '{row.get('outcome', 'unknown')}'"
                )
                return float(alpha)

        control_value = row.get("control_value", None)
        control_std = row.get("control_std", None)

        if control_value is not None and control_std is not None and control_value > 0:
            variance = control_std**2
            alpha = (variance - control_value) / (control_value**2)
            if alpha > 0.01:
                self._logger.warning(
                    f"Estimating dispersion alpha={alpha:.4f} from variance for "
                    f"outcome '{row.get('outcome', 'unknown')}'. "
                    f"Consider using store_fitted_models=True for more accurate retrodesign."
                )
                return float(alpha)

        self._logger.warning(
            f"No dispersion parameter available for outcome '{row.get('outcome', 'unknown')}'. "
            f"Using default alpha=1.0. For accurate retrodesign, use store_fitted_models=True."
        )
        return 1.0

    def _generate_sim_data(
        self,
        n_treatment: int,
        n_control: int,
        true_effect: float,
        control_value: float,
        model_type: str,
        effect_type: str,
        row: pd.Series | None = None,
        fitted_model=None,
    ) -> pd.DataFrame | None:
        """Generate synthetic data for simulation based on model type."""
        n_total = n_treatment + n_control
        treatment = np.concatenate([np.ones(n_treatment), np.zeros(n_control)])

        if model_type == "ols":
            sd = 1.0
            if row is not None:
                control_std = row.get("control_std")
                if control_std is not None and not pd.isna(control_std) and control_std > 0:
                    sd = float(control_std)
                else:
                    se = row.get("standard_error")
                    if se is not None and not pd.isna(se) and se > 0:
                        sd = float(se / np.sqrt(1 / n_treatment + 1 / n_control))
            outcome = np.where(
                treatment == 1,
                np.random.normal(control_value + true_effect, sd, n_total),
                np.random.normal(control_value, sd, n_total),
            )
            data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        elif model_type == "logistic":
            if effect_type == "probability_change":
                p_control = control_value
                p_treatment = p_control + true_effect
                p_treatment = np.clip(p_treatment, 0.001, 0.999)
                p_control = np.clip(p_control, 0.001, 0.999)
            else:
                logit_control = np.log(control_value / (1 - control_value)) if 0 < control_value < 1 else 0
                logit_treatment = logit_control + true_effect
                p_control = 1 / (1 + np.exp(-logit_control))
                p_treatment = 1 / (1 + np.exp(-logit_treatment))

            outcome = np.where(
                treatment == 1, np.random.binomial(1, p_treatment, n_total), np.random.binomial(1, p_control, n_total)
            )
            data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        elif model_type == "cox":
            hr = np.exp(true_effect)
            baseline_hazard = 0.01
            scale_control = 1 / baseline_hazard
            scale_treatment = 1 / (baseline_hazard * hr)

            time = np.where(
                treatment == 1,
                np.random.exponential(scale_treatment, n_total),
                np.random.exponential(scale_control, n_total),
            )
            max_time = np.percentile(time, 70)
            event = (time <= max_time).astype(int)
            time = np.minimum(time, max_time)
            data = pd.DataFrame({"treatment": treatment, "time": time, "event": event})

        elif model_type == "poisson":
            if effect_type == "count_change":
                lambda_control = control_value
                lambda_treatment = control_value + true_effect
            else:
                lambda_control = control_value
                lambda_treatment = control_value * np.exp(true_effect)

            lambda_treatment = max(0.1, lambda_treatment)
            lambda_control = max(0.1, lambda_control)

            outcome = np.where(
                treatment == 1, np.random.poisson(lambda_treatment, n_total), np.random.poisson(lambda_control, n_total)
            )
            data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        elif model_type == "negative_binomial":
            if effect_type == "count_change":
                lambda_control = control_value
                lambda_treatment = control_value + true_effect
            else:
                lambda_control = control_value
                lambda_treatment = control_value * np.exp(true_effect)

            lambda_treatment = max(0.1, lambda_treatment)
            lambda_control = max(0.1, lambda_control)

            alpha = self._get_nb_dispersion_parameter(row, fitted_model) if row is not None else 1.0
            n_param = 1 / alpha
            p_treatment = 1 / (1 + alpha * lambda_treatment)
            p_control = 1 / (1 + alpha * lambda_control)

            treatment_outcome = np.random.negative_binomial(n_param, p_treatment, n_treatment)
            control_outcome = np.random.negative_binomial(n_param, p_control, n_control)
            outcome = np.concatenate([treatment_outcome, control_outcome])
            data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        else:
            outcome = np.where(
                treatment == 1,
                np.random.normal(control_value + true_effect, 1, n_total),
                np.random.normal(control_value, 1, n_total),
            )
            data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        return data

    def _fit_sim_model(
        self,
        data: pd.DataFrame,
        model_type: str,
        effect_type: str,
    ) -> dict | None:
        """Fit model on simulated data and return results."""
        try:
            if model_type == "ols":
                model = smf.ols("outcome ~ treatment", data=data)
                results = model.fit()
                return {
                    "absolute_effect": results.params["treatment"],
                    "pvalue": results.pvalues["treatment"],
                }

            elif model_type == "logistic":
                outcome_sum = data["outcome"].sum()
                if outcome_sum == 0 or outcome_sum == len(data):
                    return None
                for treat_val in data["treatment"].unique():
                    group_data = data[data["treatment"] == treat_val]
                    group_sum = group_data["outcome"].sum()
                    if group_sum == 0 or group_sum == len(group_data):
                        return None

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    model = smf.logit("outcome ~ treatment", data=data)
                    results = model.fit(disp=False, maxiter=100, warn_convergence=False)
                    if not results.mle_retvals["converged"]:
                        return None

                if effect_type == "probability_change":
                    margeff = results.get_margeff(at="overall")
                    me_summary = margeff.summary_frame()
                    return {
                        "absolute_effect": me_summary.loc["treatment", "dy/dx"],
                        "pvalue": me_summary.loc["treatment", "Pr(>|z|)"],
                    }
                else:
                    return {
                        "absolute_effect": results.params["treatment"],
                        "pvalue": results.pvalues["treatment"],
                    }

            elif model_type == "cox":
                from lifelines import CoxPHFitter

                cph = CoxPHFitter()
                cph.fit(data, duration_col="time", event_col="event", formula="treatment")
                return {
                    "absolute_effect": cph.params_["treatment"],
                    "pvalue": cph.summary.loc["treatment", "p"],
                }

            elif model_type in ["poisson", "negative_binomial"]:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    if model_type == "poisson":
                        model = smf.poisson("outcome ~ treatment", data=data)
                    else:
                        model = smf.negativebinomial("outcome ~ treatment", data=data)
                    results = model.fit(disp=False, maxiter=100, warn_convergence=False)
                    if not results.mle_retvals["converged"]:
                        return None

                if effect_type == "count_change":
                    margeff = results.get_margeff(at="overall")
                    me_summary = margeff.summary_frame()
                    return {
                        "absolute_effect": me_summary.loc["treatment", "dy/dx"],
                        "pvalue": me_summary.loc["treatment", "Pr(>|z|)"],
                    }
                else:
                    return {
                        "absolute_effect": results.params["treatment"],
                        "pvalue": results.pvalues["treatment"],
                    }

            else:
                return None

        except Exception:
            return None

    def _generate_sim_data_from_model(
        self,
        fitted_model,
        true_effect: float,
        n_treatment: int,
        n_control: int,
        model_type: str,
        effect_type: str,
    ) -> pd.DataFrame | None:
        """Generate realistic simulation data using fitted model parameters."""
        try:
            n_total = n_treatment + n_control
            original_data = self._data.copy()

            sampled_indices = np.random.choice(len(original_data), size=n_total, replace=True)
            sim_data = original_data.iloc[sampled_indices].reset_index(drop=True)
            sim_data[self._treatment_col] = np.concatenate([np.ones(n_treatment), np.zeros(n_control)])

            if model_type == "ols":
                params = fitted_model.params.copy()
                params[self._treatment_col] = true_effect

                if hasattr(self, "_final_covariates") and self._final_covariates:
                    for cov in self._final_covariates:
                        if f"z_{cov}" in sim_data.columns:
                            continue
                        if cov in sim_data.columns:
                            mu = sim_data[cov].mean()
                            sd = sim_data[cov].std() + 1e-10
                            sim_data[f"z_{cov}"] = (sim_data[cov] - mu) / sd

                linear_pred = params["Intercept"]
                linear_pred += params[self._treatment_col] * sim_data[self._treatment_col]
                for param_name in params.index:
                    if param_name not in ["Intercept", self._treatment_col]:
                        if param_name in sim_data.columns:
                            linear_pred += params[param_name] * sim_data[param_name]

                residual_std = np.sqrt(fitted_model.scale)
                sim_data["outcome"] = linear_pred + np.random.normal(0, residual_std, n_total)

            elif model_type == "logistic":
                params = fitted_model.params.copy()
                params[self._treatment_col] = true_effect if effect_type == "log_odds" else params[self._treatment_col]

                if hasattr(self, "_final_covariates") and self._final_covariates:
                    for cov in self._final_covariates:
                        if f"z_{cov}" not in sim_data.columns and cov in sim_data.columns:
                            mu = sim_data[cov].mean()
                            sd = sim_data[cov].std() + 1e-10
                            sim_data[f"z_{cov}"] = (sim_data[cov] - mu) / sd

                logit_pred = params["Intercept"]

                if effect_type == "probability_change":
                    baseline_prob = 1 / (1 + np.exp(-params["Intercept"]))
                    logit_pred += np.log(
                        (baseline_prob + true_effect * sim_data[self._treatment_col])
                        / (1 - baseline_prob - true_effect * sim_data[self._treatment_col] + 1e-10)
                    )
                else:
                    logit_pred += params[self._treatment_col] * sim_data[self._treatment_col]

                for param_name in params.index:
                    if param_name not in ["Intercept", self._treatment_col]:
                        if param_name in sim_data.columns:
                            logit_pred += params[param_name] * sim_data[param_name]

                prob = 1 / (1 + np.exp(-logit_pred))
                prob = np.clip(prob, 0.001, 0.999)
                sim_data["outcome"] = np.random.binomial(1, prob)

                treatment_outcomes = sim_data[sim_data[self._treatment_col] == 1]["outcome"]
                control_outcomes = sim_data[sim_data[self._treatment_col] == 0]["outcome"]

                if (
                    treatment_outcomes.sum() == 0
                    or treatment_outcomes.sum() == len(treatment_outcomes)
                    or control_outcomes.sum() == 0
                    or control_outcomes.sum() == len(control_outcomes)
                ):
                    return None

            elif model_type == "cox":
                params = fitted_model.params_.copy()
                params[self._treatment_col] = true_effect

                if hasattr(self, "_final_covariates") and self._final_covariates:
                    for cov in self._final_covariates:
                        if f"z_{cov}" not in sim_data.columns and cov in sim_data.columns:
                            mu = sim_data[cov].mean()
                            sd = sim_data[cov].std() + 1e-10
                            sim_data[f"z_{cov}"] = (sim_data[cov] - mu) / sd

                log_hazard = params[self._treatment_col] * sim_data[self._treatment_col]
                for param_name in params.index:
                    if param_name != self._treatment_col:
                        if param_name in sim_data.columns:
                            log_hazard += params[param_name] * sim_data[param_name]

                baseline_hazard = 0.01
                hazard = baseline_hazard * np.exp(log_hazard)
                sim_data["time"] = np.random.exponential(1 / (hazard + 1e-10))

                censoring_quantile = 0.7
                max_time = np.percentile(sim_data["time"], censoring_quantile * 100)
                sim_data["event"] = (sim_data["time"] <= max_time).astype(int)
                sim_data["time"] = np.minimum(sim_data["time"], max_time)

            elif model_type in ["poisson", "negative_binomial"]:
                params = fitted_model.params.copy()

                if hasattr(self, "_final_covariates") and self._final_covariates:
                    for cov in self._final_covariates:
                        if f"z_{cov}" not in sim_data.columns and cov in sim_data.columns:
                            mu = sim_data[cov].mean()
                            sd = sim_data[cov].std() + 1e-10
                            sim_data[f"z_{cov}"] = (sim_data[cov] - mu) / sd

                log_rate = params["Intercept"]
                if effect_type == "count_change":
                    baseline_rate = np.exp(params["Intercept"])
                    log_rate = np.log(baseline_rate + true_effect * sim_data[self._treatment_col] + 1e-10)
                else:
                    log_rate += params[self._treatment_col] * sim_data[self._treatment_col]

                for param_name in params.index:
                    if param_name not in ["Intercept", self._treatment_col]:
                        if param_name in sim_data.columns:
                            log_rate += params[param_name] * sim_data[param_name]

                rate = np.exp(np.clip(log_rate, -20, 20))

                if model_type == "poisson":
                    sim_data["outcome"] = np.random.poisson(rate)
                else:
                    if hasattr(fitted_model, "params") and "alpha" in fitted_model.params:
                        alpha = fitted_model.params["alpha"]
                    else:
                        alpha = 1.0
                    n_param = 1 / alpha
                    p = 1 / (1 + alpha * rate)
                    sim_data["outcome"] = np.random.negative_binomial(n_param, p)

            else:
                return None

            return sim_data

        except Exception as e:
            self._logger.debug(f"Failed to generate data from model: {e}")
            return None

    def _fit_sim_model_from_spec(
        self,
        data: pd.DataFrame,
        fitted_model,
        model_type: str,
        effect_type: str,
    ) -> dict | None:
        """Fit model using the same specification as the fitted model."""
        try:
            if model_type == "ols":
                formula = (
                    fitted_model.model.formula if hasattr(fitted_model.model, "formula") else "outcome ~ treatment"
                )
                model = smf.ols(formula, data=data)
                results = model.fit(cov_type="HC3")
                return {
                    "absolute_effect": results.params[self._treatment_col],
                    "pvalue": results.pvalues[self._treatment_col],
                }

            elif model_type == "logistic":
                outcome_sum = data["outcome"].sum()
                if outcome_sum == 0 or outcome_sum == len(data):
                    return None
                for treat_val in data[self._treatment_col].unique():
                    group_data = data[data[self._treatment_col] == treat_val]
                    group_sum = group_data["outcome"].sum()
                    if group_sum == 0 or group_sum == len(group_data):
                        return None

                params = fitted_model.params
                covariate_names = [p for p in params.index if p not in ["Intercept", self._treatment_col]]
                if covariate_names:
                    formula = f"outcome ~ {self._treatment_col} + " + " + ".join(covariate_names)
                else:
                    formula = f"outcome ~ {self._treatment_col}"

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    model = smf.logit(formula, data=data)
                    results = model.fit(disp=False, maxiter=100, warn_convergence=False)
                    if not results.mle_retvals["converged"]:
                        return None

                if effect_type == "probability_change":
                    margeff = results.get_margeff(at="overall")
                    me_summary = margeff.summary_frame()
                    return {
                        "absolute_effect": me_summary.loc[self._treatment_col, "dy/dx"],
                        "pvalue": me_summary.loc[self._treatment_col, "Pr(>|z|)"],
                    }
                else:
                    return {
                        "absolute_effect": results.params[self._treatment_col],
                        "pvalue": results.pvalues[self._treatment_col],
                    }

            elif model_type == "cox":
                from lifelines import CoxPHFitter

                formula_vars = list(fitted_model.params_.index)
                formula = " + ".join(formula_vars)
                cph = CoxPHFitter()
                cph.fit(data, duration_col="time", event_col="event", formula=formula, robust=True)
                return {
                    "absolute_effect": cph.params_[self._treatment_col],
                    "pvalue": cph.summary.loc[self._treatment_col, "p"],
                }

            elif model_type in ["poisson", "negative_binomial"]:
                params = fitted_model.params
                exclude_params = ["Intercept", self._treatment_col]
                if model_type == "negative_binomial":
                    exclude_params.append("alpha")
                covariate_names = [p for p in params.index if p not in exclude_params]
                if covariate_names:
                    formula = f"outcome ~ {self._treatment_col} + " + " + ".join(covariate_names)
                else:
                    formula = f"outcome ~ {self._treatment_col}"

                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    if model_type == "poisson":
                        model = smf.poisson(formula, data=data)
                    else:
                        model = smf.negativebinomial(formula, data=data)
                    results = model.fit(disp=False, maxiter=100, warn_convergence=False)
                    if not results.mle_retvals["converged"]:
                        return None

                if effect_type == "count_change":
                    margeff = results.get_margeff(at="overall")
                    me_summary = margeff.summary_frame()
                    return {
                        "absolute_effect": me_summary.loc[self._treatment_col, "dy/dx"],
                        "pvalue": me_summary.loc[self._treatment_col, "Pr(>|z|)"],
                    }
                else:
                    return {
                        "absolute_effect": results.params[self._treatment_col],
                        "pvalue": results.pvalues[self._treatment_col],
                    }

            else:
                return None

        except Exception as e:
            self._logger.debug(f"Failed to fit model from spec: {e}")
            return None
