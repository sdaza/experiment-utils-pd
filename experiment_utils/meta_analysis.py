"""Meta-analysis mixin for ExperimentAnalyzer."""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.meta_analysis import combine_effects as sm_combine_effects

from .utils import log_and_raise_error


class MetaAnalysisMixin:
    """
    Mixin providing fixed and random effects meta-analysis for ExperimentAnalyzer.

    Expects the host class to expose:
        self._alpha       — significance level (float)
        self._results     — results DataFrame (pd.DataFrame | None)
        self._logger      — logger instance
        self.meta_stats_  — attribute initialised to None in __init__
    """

    def _resolve_grouping_cols(self, data: pd.DataFrame, grouping_cols: list[str] | None) -> list[str]:
        """
        Resolve grouping columns for meta-analysis and aggregation methods.

        When grouping_cols is explicitly provided, ensures 'outcome' is included.
        When None, defaults to ['outcome'] and auto-appends 'treatment_group' /
        'control_group' whenever those columns exist with more than one unique value.
        Logs the final grouping so callers can verify the behaviour.
        """
        if grouping_cols is not None:
            cols = list(grouping_cols)
            if "outcome" not in cols:
                cols.append("outcome")
            self._logger.info(f"Grouping by: {cols}")
            return cols

        cols = ["outcome"]
        auto_added = []
        for col in ["treatment_group", "control_group"]:
            if col in data.columns and data[col].nunique() > 1:
                cols.append(col)
                auto_added.append(col)

        if auto_added:
            self._logger.warning(
                f"Multiple treatment/control groups detected — auto-grouping by {cols}. "
                "Pass grouping_cols explicitly to override."
            )
        self._logger.info(f"Grouping by: {cols}")
        return cols

    def combine_effects(
        self,
        data: pd.DataFrame | None = None,
        grouping_cols: list[str] | None = None,
        method: str = "fixed",
    ) -> pd.DataFrame:
        """
        Combine effects across experiments using fixed or random effects meta-analysis.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The DataFrame containing the results. Defaults to self._results
        grouping_cols : list, optional
            The columns to group by. Defaults to experiment_identifier + ['outcome']
        method : {"fixed", "random"}, optional
            Meta-analysis method.

            - ``"fixed"`` (default): Inverse-variance weighted fixed effects.
            - ``"random"``: Random effects using the Paule-Mandel iterative estimator
              for between-study variance τ² with Hartung-Knapp-Sidik-Jonkman (HKSJ)
              adjusted confidence intervals (t-distribution, k-1 df). More conservative
              and better calibrated than fixed effects when true effects vary across
              experiments, and especially robust for small numbers of experiments.

        Returns
        -------
        pd.DataFrame
            Combined results. Heterogeneity diagnostics (tau2, i2, Q, k, method) are
            stored in ``self.meta_stats_`` for further inspection.
        """
        if method not in ("fixed", "random"):
            log_and_raise_error(self._logger, f"Unknown meta-analysis method '{method}'. Use 'fixed' or 'random'.")

        if data is None:
            data = self._results

        grouping_cols = self._resolve_grouping_cols(data, grouping_cols)

        if any(data.groupby(grouping_cols).size() < 2):
            self._logger.warning("There are some combinations with only one experiment!")

        if method == "fixed":
            estimator = self._get_fixed_meta_analysis_estimate
        else:
            estimator = self._get_random_meta_analysis_estimate

        diagnostics_list = []

        def _apply(df):
            result = estimator(df)
            diag = result.pop("_diagnostics", {})
            diag.update({col: df[col].iloc[0] for col in grouping_cols if col in df.columns})
            diagnostics_list.append(diag)
            return pd.Series(result)

        pooled_results = data.groupby(grouping_cols).apply(_apply, include_groups=False).reset_index()

        result_columns = grouping_cols + [
            "experiments",
            "control_units",
            "treatment_units",
            "absolute_effect",
            "abs_effect_lower",
            "abs_effect_upper",
            "relative_effect",
            "rel_effect_lower",
            "rel_effect_upper",
            "stat_significance",
            "standard_error",
            "pvalue",
        ]
        if "balance" in data.columns:
            index_to_insert = len(grouping_cols)
            result_columns.insert(index_to_insert + 1, "balance")
        pooled_results["stat_significance"] = pooled_results["stat_significance"].astype(int)

        if diagnostics_list:
            self.meta_stats_ = pd.DataFrame(diagnostics_list)

        method_label = "fixed-effects" if method == "fixed" else "random-effects (Paule-Mandel + HKSJ)"
        self._logger.info(f"Combining effects using {method_label} meta-analysis!")
        return pooled_results[result_columns]

    def _get_fixed_meta_analysis_estimate(self, data: pd.DataFrame) -> dict[str, int | float]:
        # Drop rows with invalid SE (zero, NaN, inf) — these are degenerate experiments
        # (e.g. all-zero outcomes) that would receive infinite IVW weight and corrupt the pooled estimate.
        valid_se = data["standard_error"].notna() & np.isfinite(data["standard_error"]) & (data["standard_error"] > 0)
        data = data[valid_se]

        if data.empty:
            return {
                "experiments": 0,
                "control_units": 0,
                "treatment_units": 0,
                "absolute_effect": np.nan,
                "abs_effect_lower": np.nan,
                "abs_effect_upper": np.nan,
                "relative_effect": np.nan,
                "rel_effect_lower": np.nan,
                "rel_effect_upper": np.nan,
                "standard_error": np.nan,
                "pvalue": np.nan,
                "stat_significance": 0,
                "_diagnostics": {"tau2": np.nan, "i2": np.nan, "Q": np.nan, "k": 0, "method": "fixed"},
            }

        # Absolute effect: IVW with 1/SE_abs²
        weights_abs = 1 / (data["standard_error"] ** 2)
        absolute_estimate = np.sum(weights_abs * data["absolute_effect"]) / np.sum(weights_abs)
        pooled_se_abs = np.sqrt(1 / np.sum(weights_abs))

        # Relative effect: IVW with 1/SE_rel² where SE_rel = SE_abs / |control_mean|
        # This is the delta-method approximation; using absolute-scale weights would give
        # wrong CIs (wrong units) when displaying relative effects in the plot.
        rel_se_pooled = np.nan
        relative_estimate = np.nan
        if "control_value" in data.columns:
            ctrl = data["control_value"].abs()
            valid = ctrl > 0
            if valid.any():
                se_rel = data.loc[valid, "standard_error"] / ctrl[valid]
                effects_rel = data.loc[valid, "relative_effect"].values.astype(float)
                variances_rel = (se_rel.values**2).astype(float)

                # Guard: drop any study where relative_effect or its variance is non-finite
                finite_mask = np.isfinite(effects_rel) & np.isfinite(variances_rel) & (variances_rel > 0)
                effects_rel = effects_rel[finite_mask]
                variances_rel = variances_rel[finite_mask]

                if len(effects_rel) > 0:
                    weights_rel = 1 / variances_rel
                    relative_estimate = np.sum(weights_rel * effects_rel) / np.sum(weights_rel)
                    rel_se_pooled = float(np.sqrt(1 / np.sum(weights_rel)))

        # Fallback: derive from pooled absolute when relative pooling is not possible
        if np.isnan(relative_estimate) and "relative_effect" in data.columns:
            rel_fallback = data["relative_effect"].values.astype(float)
            finite_rel = np.isfinite(rel_fallback)
            if finite_rel.any():
                w = weights_abs.values[finite_rel]
                relative_estimate = float(np.sum(w * rel_fallback[finite_rel]) / np.sum(w))

        np.seterr(invalid="ignore")
        try:
            pvalue = stats.norm.sf(abs(absolute_estimate / pooled_se_abs)) * 2
        except FloatingPointError:
            pvalue = np.nan

        z = stats.norm.ppf(1 - self._alpha / 2)
        meta_results = {
            "experiments": int(data.shape[0]),
            "control_units": int(data["control_units"].sum()),
            "treatment_units": int(data["treatment_units"].sum()),
            "absolute_effect": absolute_estimate,
            "abs_effect_lower": absolute_estimate - z * pooled_se_abs,
            "abs_effect_upper": absolute_estimate + z * pooled_se_abs,
            "relative_effect": relative_estimate,
            "rel_effect_lower": relative_estimate - z * rel_se_pooled if not np.isnan(rel_se_pooled) else np.nan,
            "rel_effect_upper": relative_estimate + z * rel_se_pooled if not np.isnan(rel_se_pooled) else np.nan,
            "standard_error": pooled_se_abs,
            "pvalue": pvalue,
        }

        if "balance" in data.columns:
            meta_results["balance"] = data["balance"].mean()
        meta_results["stat_significance"] = 1 if meta_results["pvalue"] < self._alpha else 0
        meta_results["_diagnostics"] = {
            "tau2": 0.0,
            "i2": np.nan,
            "Q": np.nan,
            "k": int(data.shape[0]),
            "method": "fixed",
        }
        return meta_results

    def _get_random_meta_analysis_estimate(self, data: pd.DataFrame) -> dict[str, int | float]:
        """
        Random effects meta-analysis using Paule-Mandel τ² estimator with
        Hartung-Knapp-Sidik-Jonkman (HKSJ) adjusted confidence intervals.

        The HKSJ adjustment uses a t-distribution with k-1 degrees of freedom,
        which produces wider (more conservative) CIs for small k and converges
        to the z-based CI as k grows. This makes it well-calibrated even when
        combining only 2-3 experiments.

        Relative effects are pooled as an independent RE model on the
        delta-method relative scale (SE_rel = SE_abs / |control|), which
        correctly estimates its own between-study variance τ²_rel. This is
        the standard approach used in clinical meta-analyses for ratio outcomes.
        """
        _empty = {
            "experiments": 0,
            "control_units": 0,
            "treatment_units": 0,
            "absolute_effect": np.nan,
            "abs_effect_lower": np.nan,
            "abs_effect_upper": np.nan,
            "relative_effect": np.nan,
            "rel_effect_lower": np.nan,
            "rel_effect_upper": np.nan,
            "standard_error": np.nan,
            "pvalue": np.nan,
            "stat_significance": 0,
            "_diagnostics": {"tau2": np.nan, "i2": np.nan, "Q": np.nan, "k": 0, "method": "random"},
        }

        valid_se = data["standard_error"].notna() & np.isfinite(data["standard_error"]) & (data["standard_error"] > 0)
        data = data[valid_se]

        if data.empty:
            return _empty

        k = int(data.shape[0])
        effects = data["absolute_effect"].values.astype(float)
        variances = (data["standard_error"].values ** 2).astype(float)

        # k=1: random effects is undefined — fall back to fixed effects result
        if k == 1:
            self._logger.warning("Only one valid experiment for random effects pooling; falling back to fixed effects.")
            result = self._get_fixed_meta_analysis_estimate(data)
            result["_diagnostics"]["method"] = "random-fallback-k1"
            return result

        res = sm_combine_effects(effects, variances, method_re="iterated", alpha=self._alpha)

        absolute_estimate = float(res.mean_effect_re)
        se_hksj = float(res.sd_eff_w_re_hksj)

        # Hartung-Knapp CI: t(k-1) distribution
        ci_re = res.conf_int(use_t=True)[1]
        abs_lower = float(ci_re[0])
        abs_upper = float(ci_re[1])

        # p-value from t(k-1) using HKSJ SE
        if se_hksj > 0:
            t_stat = absolute_estimate / se_hksj
            pvalue = float(2 * stats.t.sf(abs(t_stat), df=k - 1))
        else:
            pvalue = np.nan

        # Relative effect: independent RE model on delta-method relative scale.
        # SE_rel_i = SE_abs_i / |control_i| gives the per-study variance on the
        # relative scale. Running a separate RE model allows τ²_rel to differ
        # from τ²_abs (they are on different scales), which is the statistically
        # correct approach.
        relative_estimate = np.nan
        rel_lower = np.nan
        rel_upper = np.nan

        if "control_value" in data.columns:
            ctrl = data["control_value"].abs()
            valid_ctrl = ctrl > 0
            if valid_ctrl.any():
                se_rel = data.loc[valid_ctrl, "standard_error"] / ctrl[valid_ctrl]
                effects_rel = data.loc[valid_ctrl, "relative_effect"].values.astype(float)
                variances_rel = (se_rel.values**2).astype(float)

                # Guard: drop any study where relative_effect or its variance is non-finite
                finite_mask = np.isfinite(effects_rel) & np.isfinite(variances_rel) & (variances_rel > 0)
                effects_rel = effects_rel[finite_mask]
                variances_rel = variances_rel[finite_mask]

                if len(effects_rel) == 1:
                    # single valid study after filtering: report point estimate only
                    relative_estimate = float(effects_rel[0])
                elif len(effects_rel) > 1:
                    res_rel = sm_combine_effects(effects_rel, variances_rel, method_re="iterated", alpha=self._alpha)
                    relative_estimate = float(res_rel.mean_effect_re)
                    ci_rel = res_rel.conf_int(use_t=True)[1]
                    rel_lower = float(ci_rel[0])
                    rel_upper = float(ci_rel[1])

        # Fallback: IVW on absolute weights when relative pooling failed entirely
        if np.isnan(relative_estimate) and "relative_effect" in data.columns:
            rel_fallback = data["relative_effect"].values.astype(float)
            finite_rel = np.isfinite(rel_fallback)
            if finite_rel.any():
                w = (1 / variances)[finite_rel]
                relative_estimate = float(np.sum(w * rel_fallback[finite_rel]) / np.sum(w))

        diagnostics = {
            "tau2": float(res.tau2),
            "i2": float(res.i2),
            "Q": float(res.q),
            "k": k,
            "method": "random",
        }

        meta_results = {
            "experiments": k,
            "control_units": int(data["control_units"].sum()),
            "treatment_units": int(data["treatment_units"].sum()),
            "absolute_effect": absolute_estimate,
            "abs_effect_lower": abs_lower,
            "abs_effect_upper": abs_upper,
            "relative_effect": relative_estimate,
            "rel_effect_lower": rel_lower,
            "rel_effect_upper": rel_upper,
            "standard_error": se_hksj,
            "pvalue": pvalue,
        }

        if "balance" in data.columns:
            meta_results["balance"] = data["balance"].mean()
        meta_results["stat_significance"] = 1 if (not np.isnan(pvalue) and pvalue < self._alpha) else 0
        meta_results["_diagnostics"] = diagnostics
        return meta_results

    def aggregate_effects(
        self, data: pd.DataFrame | None = None, grouping_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Aggregate effects using a weighted average based on the size of the treatment group.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing the results.
        grouping_cols : list, optional
            The columns to group by. Defaults to ['outcome']

        Returns
        -------
        A Pandas DataFrame with combined results
        """
        if data is None:
            data = self._results

        grouping_cols = self._resolve_grouping_cols(data, grouping_cols)

        aggregate_results = (
            data.groupby(grouping_cols).apply(self._compute_weighted_effect, include_groups=False).reset_index()
        )

        self._logger.info("Aggregating effects using weighted averages!")
        self._logger.info("For a better standard error estimation, use meta-analysis or `combine_effects`")

        result_columns = grouping_cols + ["experiments", "balance"]
        existing_columns = [col for col in result_columns if col in aggregate_results.columns]
        remaining_columns = [col for col in aggregate_results.columns if col not in existing_columns]
        final_columns = existing_columns + remaining_columns
        return aggregate_results[final_columns]

    def _compute_weighted_effect(self, group: pd.DataFrame) -> pd.Series:
        weights = group["treatment_units"].astype(float)
        group = group.copy()
        group["gweight"] = weights
        total_weight = weights.sum()

        absolute_effect = np.sum(weights * group["absolute_effect"]) / total_weight
        relative_effect = np.sum(weights * group["relative_effect"]) / total_weight

        # SE of a weighted mean: sqrt(Σ(w_i² × SE_i²)) / Σ(w_i)
        combined_se = np.sqrt(np.sum(weights**2 * group["standard_error"] ** 2)) / total_weight
        z_score = absolute_effect / combined_se
        combined_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        output = pd.Series(
            {
                "experiments": int(group.shape[0]),
                "treatment_units": int(np.sum(group["gweight"])),
                "absolute_effect": absolute_effect,
                "relative_effect": relative_effect,
                "stat_significance": 1 if combined_p_value < self._alpha else 0,
                "standard_error": combined_se,
                "pvalue": combined_p_value,
            }
        )

        if "balance" in group.columns:
            combined_balance = np.sum(group["balance"] * group["gweight"]) / np.sum(group["gweight"])
            output["balance"] = combined_balance

        return output
