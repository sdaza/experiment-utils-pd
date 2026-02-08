"""
PowerSim class for simulation of power analysis.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from multiprocess.pool import ThreadPool
from scipy import stats

from .utils import get_logger, log_and_raise_error


class PowerSim:
    """ "
    PowerSim class for simulation of power analysis.
    """

    def __init__(
        self,
        metric: str = "proportion",
        relative_effect: bool = False,
        nsim: int = 100,
        variants: int | None = None,
        comparisons: list[tuple[int, int]] = None,
        alternative: str = "two-tailed",
        alpha: float = 0.05,  # noqa: E501
        correction: str = "bonferroni",
        fdr_method: str = "indep",
        early_stopping: bool = True,
        early_stopping_precision: float = 0.01,
    ) -> None:
        """
        PowerSim class for simulation of power analysis.

        Parameters
        ----------
        metric : str
            Count, proportion, or average
        relative effect : bool
            True when change is percentual (not absolute).
        variants : int, optional
            Number of variants (treatment groups). If not specified, defaults to 1
            (total groups = control + 1 variant). Must be >= 1.
        comparisons : list
            List of tuple with the tests to run
        nsim : int
            Number of replicates to simulate power
        alternative : str
            Alternative hypothesis, 'two-tailed', 'greater', 'smaller'
        alpha : float
            One minus statistical confidence
        correction : str
            Type of correction: 'bonferroni', 'holm', 'fdr' or None
        fdr_method : 'indep' | 'negcorr'
            If 'indep' it implements Benjamini/Hochberg for independent or if
            'negcorr' it corresponds to Benjamini/Yekutieli.
        early_stopping : bool
            If True, stop simulations early when power estimates have stabilized (default: True)
        early_stopping_precision : float
            Target precision (standard error) for early stopping (default: 0.01)
        """

        self.logger = get_logger("Power Simulator")
        self.metric = metric
        self.relative_effect = relative_effect
        self.early_stopping = early_stopping
        self.early_stopping_precision = early_stopping_precision

        # Handle default for variants
        if variants is None:
            self.logger.info("'variants' not specified. Defaulting to 1 (control + 1 variant).")
            self.variants = 1
        else:
            self.variants = variants

        # Validate variants
        if not isinstance(self.variants, int) or self.variants < 1:
            log_and_raise_error(self.logger, "'variants' must be an integer >= 1")

        self.comparisons = (
            list(itertools.combinations(range(self.variants + 1), 2)) if comparisons is None else comparisons
        )  # noqa: E501
        self.nsim = nsim
        self.alternative = alternative
        self.alpha = alpha
        self.correction = correction
        self.fdr_method = fdr_method

    def __run_experiment(
        self,
        baseline: list[float] = None,
        sample_size: list[int] = None,
        effect: list[float] = None,
        compliance: list[float] = None,
        standard_deviation: list[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:  # noqa: E501
        """
        Simulate data to run power analysis.

        Parameters
        ----------
        baseline : list
            List baseline rates for counts or proportions, or base average for mean comparisons
        sample_size : list
            List with sample for control and arm groups
        effect : list
            List with effect sizes
        standard_deviation : list
            List of standard deviations by groups
        compliance : list
            List with compliance values

        Returns
        -------
        Two vectors with simulated data
        """

        # initial checks
        if effect is None:
            effect = [0.1]
        if standard_deviation is None:
            standard_deviation = [1]
        if sample_size is None:
            sample_size = [100]
        if compliance is None:
            compliance = [1.0]
        if baseline is None:
            baseline = [1.0]
        if len(effect) != self.variants:
            if len(effect) > 1:
                log_and_raise_error(
                    self.logger, "Effects should be same length as the number of self.variants or length 1!"
                )  # noqa: E501
            effect = list(itertools.repeat(effect[0], self.variants))

        if len(compliance) != self.variants:
            if len(compliance) > 1:
                log_and_raise_error(
                    self.logger, "Compliance rates should be same length as the number of self.variants or length 1!"
                )  # noqa: E501
            compliance = list(itertools.repeat(compliance[0], self.variants))

        if len(standard_deviation) != self.variants + 1:
            if len(standard_deviation) > 1:
                log_and_raise_error(
                    self.logger,
                    "Standard deviations should be same length as the number of self.variants + 1 or length 1!",
                )  # noqa: E501
            standard_deviation = list(itertools.repeat(standard_deviation[0], self.variants + 1))

        if len(sample_size) != self.variants + 1:
            if len(sample_size) > 1:
                log_and_raise_error(
                    self.logger, "N should be same length as the number of self.variants + 1 or length 1!"
                )  # noqa: E501
            sample_size = list(itertools.repeat(sample_size[0], self.variants + 1))

        if len(baseline) != self.variants + 1:
            if len(baseline) > 1:
                log_and_raise_error(
                    self.logger, "Baseline values should be same length as the number of self.variants + 1 or length 1!"
                )  # noqa: E501
            baseline = list(itertools.repeat(baseline[0], self.variants + 1))

        re = list(range(self.variants))

        # outputs
        dd = np.array([])

        # index of condition
        vv = np.array([])

        if self.metric == "count":
            c_data = np.random.poisson(baseline[0], sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i + 1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i + 1] + effect[i]
                t_data_c = np.random.poisson(re[i], int(np.round(sample_size[i + 1] * compliance[i])))
                t_data_nc = np.random.poisson(baseline[i + 1], int(np.round(sample_size[i + 1] * (1 - compliance[i]))))
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == "proportion":
            c_data = np.random.binomial(n=1, size=int(sample_size[0]), p=baseline[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i + 1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i + 1] + effect[i]

                t_data_c = np.random.binomial(n=1, size=int(np.round(sample_size[i + 1] * compliance[i])), p=re[i])
                t_data_nc = np.random.binomial(
                    n=1, size=int(np.round(sample_size[i + 1] * (1 - compliance[i]))), p=baseline[i + 1]
                )  # noqa: E501
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == "average":
            c_data = np.random.normal(baseline[0], standard_deviation[0], sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline[i + 1] * (1.00 + effect[i])
                else:
                    re[i] = baseline[i + 1] + effect[i]

                t_data_c = np.random.normal(
                    re[i], standard_deviation[i + 1], int(np.round(sample_size[i + 1] * compliance[i]))
                )  # noqa: E501
                t_data_nc = np.random.normal(
                    baseline[i + 1], standard_deviation[i + 1], int(np.round(sample_size[i + 1] * (1 - compliance[i])))
                )  # noqa: E501

                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        return dd, vv

    def _run_single_power_simulation(
        self,
        baseline: list[float],
        effect: list[float],
        sample_size: list[int],
        compliance: list[float],
        standard_deviation: list[float],
        comparisons_to_compute: list[tuple[int, int]],
    ) -> list[bool]:
        """
        Run a single power simulation iteration.

        Parameters
        ----------
        baseline : list[float]
            Baseline values for all groups
        effect : list[float]
            Effect sizes
        sample_size : list[int]
            Sample sizes
        compliance : list[float]
            Compliance rates
        standard_deviation : list[float]
            Standard deviations
        comparisons_to_compute : list[tuple[int, int]]
            List of (control, treatment) comparison pairs

        Returns
        -------
        list[bool]
            List of significance results (True/False) for each comparison
        """
        # y = output, x = index of condition
        y, x = self.__run_experiment(
            baseline=baseline,
            effect=effect,
            sample_size=sample_size,
            compliance=compliance,
            standard_deviation=standard_deviation,
        )

        # iterate only over comparisons we need to compute
        l_pvalues = []
        for j, h in comparisons_to_compute:
            # Skip comparison if either group has 0 sample size (bounds check first)
            if j >= len(sample_size) or h >= len(sample_size) or sample_size[j] == 0 or sample_size[h] == 0:
                l_pvalues.append(1.0)  # p-value of 1.0 (not significant)
                continue

            # getting pvalues
            if self.metric == "count":
                ty = np.append(y[np.isin(x, j)], y[np.isin(x, h)])
                tx = np.append(x[np.isin(x, j)], x[np.isin(x, h)])
                tx[np.isin(tx, j)] = 0
                tx[np.isin(tx, h)] = 1

                model = sm.Poisson(ty, sm.add_constant(tx))
                pm = model.fit(disp=False)
                pvalue = pm.pvalues[1]
                z = pm.params[1]

            elif self.metric == "proportion":
                z, pvalue = sm.stats.proportions_ztest(
                    [np.sum(y[np.isin(x, h)]), np.sum(y[np.isin(x, j)])],
                    [len(y[np.isin(x, h)]), len(y[np.isin(x, j)])],
                )

            elif self.metric == "average":
                z, pvalue = stats.ttest_ind(y[np.isin(x, h)], y[np.isin(x, j)], equal_var=False)

            l_pvalues.append(pvalue)

        pvalue_adjustment = {"two-tailed": 1, "greater": 0.5, "smaller": 0.5}

        correction_methods = {
            "bonferroni": self.bonferroni,
            "holm": self.holm_bonferroni,
            "hochberg": self.hochberg,
            "sidak": self.sidak,
            "fdr": self.lsu,
        }

        if self.correction in correction_methods:
            # Apply correction based on total number of comparisons defined, not just computed ones
            # This ensures consistent power calculation even when only computing subset
            if len(self.comparisons) == 1:
                # Only one comparison defined total - no correction needed
                significant = [l_pvalues[0] < self.alpha / pvalue_adjustment[self.alternative]]
            elif len(l_pvalues) < len(self.comparisons):
                # Computing subset of comparisons - manually apply correction using total count
                # to ensure consistency (e.g., Bonferroni: alpha / total_comparisons)
                adjusted_alpha = self.alpha / pvalue_adjustment[self.alternative]
                if self.correction == "bonferroni":
                    # Bonferroni: compare each p-value to alpha / m (where m = total comparisons)
                    significant = [p < adjusted_alpha / len(self.comparisons) for p in l_pvalues]
                elif self.correction == "sidak":
                    # Sidak: 1 - (1-alpha)^(1/m)
                    sidak_alpha = 1 - (1 - adjusted_alpha) ** (1 / len(self.comparisons))
                    significant = [p < sidak_alpha for p in l_pvalues]
                else:
                    # For other methods (Holm, Hochberg, FDR), need all p-values
                    # Use simple Bonferroni as fallback
                    significant = [p < adjusted_alpha / len(self.comparisons) for p in l_pvalues]
            else:
                # Computing all comparisons - use standard correction
                significant = correction_methods[self.correction](
                    np.array(l_pvalues), self.alpha / pvalue_adjustment[self.alternative]
                )  # noqa: E501
        else:
            # No correction - compare each p-value to alpha directly
            significant = [p < self.alpha for p in l_pvalues]

        return significant

    def _run_single_retrodesign_simulation(
        self,
        baseline: list[float],
        sample_size: list[int],
        true_effect: list[float],
        compliance: list[float],
        standard_deviation: list[float],
        comp_true_effect: float,
        h: int,
        j: int,
    ) -> tuple[float, float, bool]:
        """
        Run a single retrodesign simulation iteration for one comparison.

        Parameters
        ----------
        baseline : list[float]
            Baseline values for all groups
        sample_size : list[int]
            Sample sizes
        true_effect : list[float]
            True effect sizes
        compliance : list[float]
            Compliance rates
        standard_deviation : list[float]
            Standard deviations
        comp_true_effect : float
            The true effect for this comparison
        h : int
            Control group index
        j : int
            Treatment group index

        Returns
        -------
        tuple[float, float, bool]
            (observed_effect, pvalue, is_significant)
        """
        y, x = self.__run_experiment(
            baseline=baseline,
            sample_size=sample_size,
            effect=true_effect,
            compliance=compliance,
            standard_deviation=standard_deviation,
        )

        if self.metric == "count":
            ty = np.append(y[np.isin(x, j)], y[np.isin(x, h)])
            tx = np.append(x[np.isin(x, j)], x[np.isin(x, h)])
            tx[np.isin(tx, j)] = 0
            tx[np.isin(tx, h)] = 1

            model = sm.Poisson(ty, sm.add_constant(tx))
            pm = model.fit(disp=False)
            pvalue = pm.pvalues[1]
            observed_effect = pm.params[1]

        elif self.metric == "proportion":
            count_h = np.sum(y[np.isin(x, h)])
            count_j = np.sum(y[np.isin(x, j)])
            nobs_h = len(y[np.isin(x, h)])
            nobs_j = len(y[np.isin(x, j)])

            z, pvalue = sm.stats.proportions_ztest([count_h, count_j], [nobs_h, nobs_j])

            p_h = count_h / nobs_h if nobs_h > 0 else 0
            p_j = count_j / nobs_j if nobs_j > 0 else 0
            observed_effect = p_j - p_h

        elif self.metric == "average":
            sample_h = y[np.isin(x, h)]
            sample_j = y[np.isin(x, j)]

            t_stat, pvalue = stats.ttest_ind(sample_j, sample_h, equal_var=False)
            observed_effect = np.mean(sample_j) - np.mean(sample_h)

        pvalue_threshold = {"two-tailed": self.alpha, "greater": self.alpha, "smaller": self.alpha}
        is_significant = pvalue < pvalue_threshold.get(self.alternative, self.alpha)

        return observed_effect, pvalue, is_significant

    def get_power(
        self,
        baseline: float | list[float] = None,
        effect: float | list[float] = None,
        sample_size: int | list[int] = None,
        compliance: float | list[float] = None,
        standard_deviation: float | list[float] = None,
        target_comparisons: list[tuple[int, int]] = None,
        use_early_stopping: bool = None,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Estimate power using simulation.

        Parameters
        ----------
        baseline : float or list
            Baseline rate for counts or proportions, or base average for mean comparisons.
            Can be a single float or list of floats.
        effect : float or list
            Effect size(s). Can be a single float or list of floats.
        sample_size : int or list
            Total sample per group (control + variants). Provide a single int to apply equally, or a list per group.
        compliance : float or list
            Compliance value(s). Provide a single float or list per variant.
        standard_deviation : float or list
            Standard deviation(s) for control and variants. Provide a single float or list per group.
        target_comparisons : list of tuples, optional
            Which comparisons to compute power for. Defaults to all comparisons defined in the instance.
            Example: [(0, 1), (0, 2)] to only compute power for control vs variant 1 and 2.
        use_early_stopping : bool, optional
            Override the instance's early_stopping setting. If None, uses instance setting.

        Returns
        -------
        power : pd.DataFrame
            DataFrame with comparisons and their power values.
        """

        # Set default values for mutable arguments
        if baseline is None:
            baseline = [1.0]
        elif isinstance(baseline, int | float):
            baseline = [float(baseline)]
        if effect is None:
            effect = [0.10]
        elif isinstance(effect, int | float):
            effect = [float(effect)]
        if sample_size is None:
            sample_size = [1000]
        elif isinstance(sample_size, int | float):
            sample_size = [int(sample_size)]
        if compliance is None:
            compliance = [1.0]
        elif isinstance(compliance, int | float):
            compliance = [float(compliance)]
        if standard_deviation is None:
            standard_deviation = [1]
        elif isinstance(standard_deviation, int | float):
            standard_deviation = [float(standard_deviation)]

        # Expand sample_size to match number of groups if needed
        if len(sample_size) == 1:
            sample_size = sample_size * (self.variants + 1)

        # Handle target_comparisons
        if target_comparisons is None:
            comparisons_to_compute = self.comparisons
        else:
            # Normalize comparison order (always smaller group first)
            normalized_comps = []
            for g1, g2 in target_comparisons:
                if g1 > g2:
                    normalized_comps.append((g2, g1))
                    if (g1, g2) not in self.comparisons:
                        self.logger.debug(f"Auto-corrected comparison order: ({g1}, {g2}) → ({g2}, {g1})")
                else:
                    normalized_comps.append((g1, g2))
            target_comparisons = normalized_comps

            # Validate that target_comparisons are in self.comparisons
            invalid_comps = [c for c in target_comparisons if c not in self.comparisons]
            if invalid_comps:
                log_and_raise_error(
                    self.logger,
                    f"target_comparisons {invalid_comps} are not in defined comparisons {self.comparisons}",
                )
            comparisons_to_compute = target_comparisons

        # create empty values for results (only for comparisons we're computing)
        pvalues = {}
        for c in range(len(comparisons_to_compute)):
            pvalues[c] = []

        # Determine if we should use early stopping
        early_stopping_enabled = self.early_stopping if use_early_stopping is None else use_early_stopping

        # Decide whether to use parallel processing
        use_parallel = self.nsim >= 500 and not early_stopping_enabled

        if use_parallel:
            from joblib import Parallel, delayed

            self.logger.info(f"Running {self.nsim} power simulations in parallel...")

            # Run all simulations in parallel
            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(self._run_single_power_simulation)(
                    baseline=baseline,
                    effect=effect,
                    sample_size=sample_size,
                    compliance=compliance,
                    standard_deviation=standard_deviation,
                    comparisons_to_compute=comparisons_to_compute,
                )
                for _ in range(self.nsim)
            )

            # Aggregate results
            for significant in results:
                for v, p in enumerate(significant):
                    pvalues[v].append(p)
        else:
            # Sequential execution (needed for early stopping or small nsim)
            if self.nsim >= 500:
                self.logger.info(f"Running {self.nsim} power simulations sequentially (early stopping enabled)...")

            # Early stopping parameters
            check_interval = 20  # Check every 20 simulations
            min_sims = 50  # Minimum simulations before considering early stopping

            # iterate over simulations
            for _i in range(self.nsim):
                significant = self._run_single_power_simulation(
                    baseline=baseline,
                    effect=effect,
                    sample_size=sample_size,
                    compliance=compliance,
                    standard_deviation=standard_deviation,
                    comparisons_to_compute=comparisons_to_compute,
                )

                for v, p in enumerate(significant):
                    pvalues[v].append(p)

                # Early stopping check
                if early_stopping_enabled and (_i + 1) >= min_sims and (_i + 1) % check_interval == 0:
                    # Check if all comparisons have stabilized
                    all_stable = True
                    for comp_idx in range(len(comparisons_to_compute)):
                        results_so_far = pvalues[comp_idx]
                        n = len(results_so_far)
                        if n > 0:
                            p_hat = np.mean(results_so_far)
                            # Standard error of proportion: sqrt(p*(1-p)/n)
                            # Clip p_hat to avoid division issues at boundaries
                            p_hat_clipped = np.clip(p_hat, 0.01, 0.99)
                            se = np.sqrt(p_hat_clipped * (1 - p_hat_clipped) / n)

                            # Be more conservative near boundaries (power near 0 or 1)
                            # to avoid premature stopping when power is marginal
                            precision_threshold = self.early_stopping_precision
                            if 0.3 < p_hat < 0.95:  # Only use aggressive stopping when power is clearly high or low
                                precision_threshold = self.early_stopping_precision
                            else:
                                # Need more precision near boundaries
                                precision_threshold = self.early_stopping_precision / 2

                            if se > precision_threshold:
                                all_stable = False
                                break

                    if all_stable:
                        self.logger.debug(
                            f"Early stopping at {_i + 1}/{self.nsim} simulations "
                            f"(precision target: {self.early_stopping_precision})"
                        )
                        break

        power = pd.DataFrame(pd.DataFrame(pvalues).mean()).reset_index()
        power.columns = ["comparisons", "power"]
        power["comparisons"] = power["comparisons"].map(dict(enumerate(comparisons_to_compute)))

        return power

    def get_power_from_data(
        self,
        df: pd.DataFrame,
        metric_col: str,
        sample_size: int | list[int] = None,
        effect: float | list[float] = None,
        compliance: float | list[float] = None,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Simulate statistical power using samples from the provided data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing the metric data.
        metric_col : str
            Name of the column in the dataframe that contains the measurement used for testing.
        sample_size : int or list
            Sample sizes for control and variants. Provide a single int to apply equally, or a list per group.
        effect : float or list
            Effect sizes for each variant group. Provide a single float or list per variant.
        compliance : float or list
            Compliance rates for each variant group. Provide a single float or list per variant.

        Returns
        -------
        pd.DataFrame
            DataFrame with each comparison (as defined in self.comparisons) and the corresponding estimated power.
        """
        # Set default values for mutable arguments
        if sample_size is None:
            sample_size = [100]
        elif isinstance(sample_size, int | float):
            sample_size = [int(sample_size)]
        if effect is None:
            effect = [0.10]
        elif isinstance(effect, int | float):
            effect = [float(effect)]
        if compliance is None:
            compliance = [1.0]
        elif isinstance(compliance, int | float):
            compliance = [float(compliance)]

        # Verify metric column exists
        if metric_col not in df.columns:
            log_and_raise_error(self.logger, f"Column '{metric_col}' not found in dataframe.")

            # initial checks
        if len(effect) != self.variants:
            if len(effect) > 1:
                log_and_raise_error(
                    self.logger, "Effects should be same length as the number of self.variants or length 1!"
                )  # noqa: E501
            effect = list(itertools.repeat(effect[0], self.variants))

        if len(compliance) != self.variants:
            if len(compliance) > 1:
                log_and_raise_error(
                    self.logger, "Compliance rates should be same length as the number of self.variants or length 1!"
                )  # noqa: E501
            compliance = list(itertools.repeat(compliance[0], self.variants))

        # compliance cannot be higher than 1 or lower than 0
        if any([c > 1 or c < 0 for c in compliance]):
            log_and_raise_error(self.logger, "Compliance rates should be between 0 and 1!")

        if len(sample_size) != self.variants + 1:
            if len(sample_size) > 1:
                log_and_raise_error(
                    self.logger, "N should be same length as the number of self.variants + 1 or length 1!"
                )  # noqa: E501
            sample_size = list(itertools.repeat(sample_size[0], self.variants + 1))

        # The sum of sample size cannot be higher than the number of rows in the dataframe
        if sum(sample_size) > df.shape[0]:
            log_and_raise_error(
                self.logger, "Sum of sample sizes cannot be higher than the number of rows in the dataframe!"
            )  # noqa: E501

        # Adjust sample size by compliance
        sample_size = [int(np.round(s * c)) for s, c in zip(sample_size, [1] + compliance, strict=False)]

        # Initialize storage for significance results over simulation iterations
        pvalues_dict = {c: [] for c in range(len(self.comparisons))}
        n_iter = self.nsim  # number of bootstrap iterations

        for _i in range(n_iter):
            # Create a bootstrap sample per group (sampling with replacement) for each variant and sample size

            boot_samples = {}
            remaining = None

            for j in range(self.variants + 1):
                if remaining is None:
                    temp = df.sample(n=sample_size[j], replace=False)
                    remaining = df[~df.index.isin(temp.index)]
                else:
                    temp = remaining.sample(n=sample_size[j], replace=False)
                    remaining = remaining[~remaining.index.isin(temp.index)]

                if self.relative_effect:
                    effects = [1] + [1 + e for e in effect]
                    boot_samples[j] = temp[metric_col].values * effects[j]
                else:
                    effects = [0] + effect
                    boot_samples[j] = temp[metric_col].values + effects[j]

            # For each comparison defined in self.comparisons, perform the appropriate test
            iter_pvals = []
            for j, h in self.comparisons:
                if self.metric == "average":
                    # Two-sample t-test (unequal variance)
                    try:
                        t_stat, pval = stats.ttest_ind(boot_samples[j], boot_samples[h], equal_var=False)
                    except Exception as e:
                        self.logger.error(f"Error while performing t-test: {e}")
                        pval = np.nan
                elif self.metric == "proportion":
                    # Assume binary data (0/1)
                    count = [np.sum(boot_samples[h]), np.sum(boot_samples[j])]
                    nobs = [len(boot_samples[h]), len(boot_samples[j])]
                    try:
                        t_stat, pval = sm.stats.proportions_ztest(count, nobs)
                    except Exception as e:
                        self.logger.error(f"Error while performing proportions_ztest: {e}")
                        pval = np.nan
                iter_pvals.append(pval)

            # Adjust multiple comparisons for this iteration
            pvalue_adjustment = {"two-tailed": 1, "greater": 0.5, "smaller": 0.5}
            correction_methods = {
                "bonferroni": self.bonferroni,
                "holm": self.holm_bonferroni,
                "hochberg": self.hochberg,
                "sidak": self.sidak,
                "fdr": self.lsu,
            }
            if self.correction in correction_methods:
                significances = correction_methods[self.correction](
                    np.array(iter_pvals), self.alpha / pvalue_adjustment[self.alternative]
                )
            else:
                # No adjustment provided; simply check p < alpha
                significances = np.array(np.array(iter_pvals) < self.alpha)

            # Save boolean outcomes for each comparison
            for idx, sig in enumerate(significances):
                pvalues_dict[idx].append(sig)

        # Compute power as the mean rejection rate for each comparison
        power_estimates = {idx: np.mean(sig_arr) for idx, sig_arr in pvalues_dict.items()}
        power_df = pd.DataFrame(list(power_estimates.items()), columns=["comparison_index", "power"])
        # Map index to actual comparison tuple
        power_df["comparisons"] = power_df["comparison_index"].map(dict(enumerate(self.comparisons)))
        power_df = power_df[["comparisons", "power"]]
        return power_df

    def grid_sim_power(
        self,
        baseline_rates: list[float] = None,
        effects: list[float] = None,
        sample_sizes: list[int] = None,  # noqa: E501
        compliances: list[list[float]] = None,
        standard_deviations: list[list[float]] = None,
        threads: int = 3,
        plot: bool = False,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Return Pandas DataFrame with parameter combinations and statistical power

        Parameters
        ----------
        baseline_rates : list
            List of baseline rates for counts or proportions, or base average for mean comparisons.
        effects : list
            List with effect sizes.
        sample_sizes : list
            List with sample for control and variants.
        compliances : list
            List with compliance values.
        standard_deviations : list
            List of standard deviations of control and variants.
        threads : int
            Number of threads for parallelization.
        plot : bool
            Whether to plot the results.
        """

        if compliances is None:
            compliances = [[1]]
        if standard_deviations is None:
            standard_deviations = [[1]]
        pdict = {
            "baseline": baseline_rates,
            "effect": effects,
            "sample_size": sample_sizes,
            "compliance": compliances,
            "standard_deviation": standard_deviations,
        }
        grid = self.__expand_grid(pdict)

        parameters = list(grid.itertuples(index=False, name=None))

        grid["nsim"] = self.nsim
        grid["alpha"] = self.alpha
        grid["alternative"] = self.alternative
        grid["metric"] = self.metric
        grid["variants"] = self.variants
        grid["comparisons"] = str(self.comparisons)
        grid["relative_effect"] = self.relative_effect
        grid = grid.loc[
            :,
            [
                "baseline",
                "effect",
                "sample_size",
                "compliance",
                "standard_deviation",
                "variants",
                "comparisons",
                "nsim",
                "alpha",
                "alternative",
                "metric",
                "relative_effect",
            ],
        ]
        pool = ThreadPool(processes=threads)
        results = pool.starmap(self.get_power, parameters)
        pool.close()
        pool.join()

        results = pd.concat(results)

        # create index
        index = []
        repeating_number = grid.shape[0]
        repeating_count = len(self.comparisons)
        for i in range(0, repeating_number):
            index.extend([i] * repeating_count)

        results["index"] = index
        results = results.pivot(index=["index"], columns=["comparisons"], values=["power"])
        results.columns = [str((i, j)) for i, j in self.comparisons]

        grid = pd.concat([grid, results], axis=1)
        grid.sample_size = grid.sample_size.map(str)
        grid.effect = grid.effect.map(str)
        if plot:
            self.plot_power(grid)
        return grid

    def plot_power(self, data: pd.DataFrame) -> None:
        """
        Plot statistical power by scenario
        """

        value_vars = [str((i, j)) for i, j in self.comparisons]

        cols = [
            "baseline",
            "effect",
            "sample_size",
            "compliance",
            "standard_deviation",
            "variants",
            "comparisons",
            "nsim",
            "alpha",
            "alternative",
            "metric",
            "relative_effect",
        ]

        temp = pd.melt(data, id_vars=cols, var_name="comparison", value_name="power", value_vars=value_vars)

        d_relative_effect = {True: "relative", False: "absolute"}
        effects = list(temp.effect.unique())
        for i in effects:
            plot = sns.lineplot(
                x="sample_size",
                y="power",
                hue="comparison",
                errorbar=None,
                data=temp[temp["effect"] == i],
                legend="full",
            )  # noqa: E501
            plt.hlines(y=0.8, linestyles="dashed", xmin=0, xmax=len(temp.sample_size.unique()) - 1, colors="gray")
            plt.title(
                f"Simulated power estimation for {self.metric}s, {d_relative_effect[self.relative_effect]} effects {str(i)}\n (sims per scenario:{self.nsim})"  # noqa: E501
            )  # noqa: E501
            plt.legend(bbox_to_anchor=(1.05, 1), title="comparison", loc="upper left")
            plt.xlabel("\n sample size")
            plt.ylabel("power\n")
            plt.setp(plot.get_xticklabels(), rotation=45)
            plt.show()

    def __expand_grid(self, dictionary: dict[str, list[float | int]]) -> pd.DataFrame:
        """
        Auxiliary function to expand a dictionary
        """
        return pd.DataFrame([row for row in itertools.product(*dictionary.values())], columns=dictionary.keys())

    def bonferroni(self, pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """A function for controlling the FWER at some level alpha using the
        classical Bonferroni procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output:
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        m, pvals = len(pvals), np.asarray(pvals)
        return pvals < alpha / float(m)

    def hochberg(self, pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """A function for controlling the FWER using Hochberg's procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output
        -------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        m, pvals = len(pvals), np.asarray(pvals)
        # sort the p-values into ascending order
        ind = np.argsort(pvals)

        test = [p <= alpha / (m + 1 - (k + 1)) for k, p in enumerate(pvals[ind])]
        significant = np.zeros(np.shape(pvals), dtype="bool")
        significant[ind[0 : np.sum(test)]] = True
        return significant

    def holm_bonferroni(self, pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """A function for controlling the FWER using the Holm-Bonferroni
        procedure.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output
        -------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """

        m, pvals = len(pvals), np.asarray(pvals)
        ind = np.argsort(pvals)

        # Holm-Bonferroni: compare p_(i) to alpha / (m - i + 1)
        # With zero-based k, threshold is alpha / (m - k)
        thresholds = [alpha / (m - k) for k in range(m)]

        # Count how many nulls to reject (sequentially until first failure)
        reject_count = 0
        for k, p in enumerate(pvals[ind]):
            if p <= thresholds[k]:
                reject_count += 1
            else:
                break

        significant = np.zeros(np.shape(pvals), dtype="bool")
        significant[ind[0:reject_count]] = True
        return significant

    def sidak(self, pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """A function for controlling the FWER at some level alpha using the
        procedure by Sidak.

        Parameters
        ----------
        pvals : array_like
            Set of p-values of the individual tests.
        alpha: float
            The desired family-wise error rate.

        Output
        ------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """
        n, pvals = len(pvals), np.asarray(pvals)
        return pvals < 1.0 - (1.0 - alpha) ** (1.0 / n)

    def lsu(self, pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
        """The (non-adaptive) one-stage linear step-up procedure (LSU) for
        controlling the false discovery rate, i.e. the classic FDR method
        proposed by Benjamini & Hochberg (1995).

        Parameters
        ----------
        pvals: array_like
            Set of p-values of the individual tests.
        q: float
            The desired false discovery rate.

        Output:
        --------
        significant: array, bool
            True if a hypothesis is rejected, False if not.
        """

        m = len(pvals)
        sort_ind = np.argsort(pvals)
        k = [i for i, p in enumerate(pvals[sort_ind]) if p < (i + 1.0) * q / m]
        significant = np.zeros(m, dtype="bool")
        if k:
            significant[sort_ind[0 : k[-1] + 1]] = True
        return significant

    def _optimize_allocation(
        self,
        power_dict: dict,
        target_comparisons: list,
        baseline: list,
        effect: list,
        compliance: list,
        standard_deviation: list,
        num_groups: int,
        min_sample_size: int,
        max_sample_size: int,
        tolerance: float,
        step_size: int,
    ) -> list[float]:
        """
        Find optimal allocation using greedy heuristic.

        For each comparison (sorted by effect size), finds the minimum sample size
        needed with equal allocation between the two groups. Each group gets the
        maximum sample size needed across all its comparisons.

        This is fast and provides good allocations, though some overpowering is
        inevitable when groups participate in multiple comparisons.
        """
        import warnings

        # Check for symmetric case: all effects equal and all power targets equal
        # For control vs variants comparisons only
        control_comparisons = [(g1, g2) for g1, g2 in target_comparisons if 0 in (g1, g2)]

        if len(control_comparisons) == len(target_comparisons):
            # All comparisons involve control
            variant_effects = [abs(effect[max(g1, g2) - 1]) for g1, g2 in target_comparisons]
            power_targets = [power_dict[comp] for comp in target_comparisons]

            # Check if all effects are the same (within tolerance) and all power targets are the same
            if (len(set(variant_effects)) == 1 or max(variant_effects) - min(variant_effects) < 0.001) and (
                len(set(power_targets)) == 1
            ):
                # Symmetric case - use equal allocation
                self.logger.info("Symmetric effects and power targets detected - using equal allocation")
                return [1.0 / num_groups] * num_groups

        group_samples = [0] * num_groups

        # Sort comparisons by absolute effect size (smallest = hardest to detect)
        def get_effect_size(comp):
            g1, g2 = comp
            if g1 == 0:
                return abs(effect[g2 - 1]) if g2 > 0 else 0
            elif g2 == 0:
                return abs(effect[g1 - 1]) if g1 > 0 else 0
            else:
                e1 = effect[g1 - 1] if g1 > 0 else 0
                e2 = effect[g2 - 1] if g2 > 0 else 0
                return abs(e2 - e1)

        sorted_comparisons = sorted(target_comparisons, key=get_effect_size)

        # For each comparison, find minimum sample size
        # Note: We need to test with ALL groups to properly account for multiple comparison corrections
        for group1, group2 in sorted_comparisons:
            comp_target_power = power_dict[(group1, group2)]
            comp_result_idx = self.comparisons.index((group1, group2))

            low = min_sample_size
            high = max_sample_size
            found_size = None

            current_size = low
            while current_size <= high:
                per_group = int(current_size / 2)

                # Test with ALL groups, using current max for groups already processed
                # and per_group for the groups in this comparison
                sample_sizes = list(group_samples)  # Start with current max for each group
                sample_sizes[group1] = max(sample_sizes[group1], per_group)
                sample_sizes[group2] = max(sample_sizes[group2], per_group)

                # Ensure groups involved in comparisons have minimum samples
                # Groups not in target_comparisons can be 0
                groups_needed = set()
                for comp in target_comparisons:
                    groups_needed.add(comp[0])
                    groups_needed.add(comp[1])

                for i in range(num_groups):
                    if i in groups_needed:
                        sample_sizes[i] = max(sample_sizes[i], 100)

                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        power_result = self.get_power(
                            baseline=baseline,
                            effect=effect,
                            sample_size=sample_sizes,
                            compliance=compliance,
                            standard_deviation=standard_deviation,
                        )
                    current_power = power_result.iloc[comp_result_idx]["power"]

                    if current_power >= comp_target_power - tolerance:
                        found_size = per_group
                        break
                except Exception:
                    pass

                current_size += step_size

            if found_size is None:
                found_size = int(high / 2)

            group_samples[group1] = max(group_samples[group1], found_size)
            group_samples[group2] = max(group_samples[group2], found_size)

        # Convert to allocation ratios
        total_samples = sum(group_samples)

        if total_samples == 0:
            allocation = [1.0 / num_groups] * num_groups
        else:
            allocation = [s / total_samples for s in group_samples]

        return allocation

    def find_sample_size(
        self,
        target_power: float | dict[tuple[int, int], float] = 0.80,
        baseline: float | list[float] = None,
        effect: float | list[float] = None,
        compliance: list[float] = None,
        standard_deviation: list[float] = None,
        allocation_ratio: list[float] = None,
        target_comparisons: list[tuple[int, int]] = None,
        power_criteria: str = "all",
        optimize_allocation: bool = False,
        min_sample_size: int = 100,
        max_sample_size: int = 100000,
        tolerance: float = 0.01,
        step_size: int = 100,
        correction: str = None,
    ) -> pd.DataFrame:
        """
        Find the minimum total sample size needed to achieve target power level(s).

        Uses binary search to efficiently find the sample size that achieves
        the target power within the specified tolerance. The total sample size
        is distributed across groups according to allocation_ratio.

        Multiple comparison corrections (specified during PowerSim initialization)
        are automatically applied when calculating power. More conservative corrections
        (e.g., Bonferroni) will require larger sample sizes than less conservative
        ones (e.g., FDR) or no correction.

        Parameters
        ----------
        target_power : float or dict
            The desired power level. Can be either:
            - A single float (e.g., 0.80) to apply the same target to all comparisons
            - A dict mapping comparisons to their specific power targets
              (e.g., {(0,1): 0.90, (0,2): 0.80})
        baseline : float or list
            Baseline rate for counts or proportions, or base average for mean comparisons.
            - Single float: Applied to all groups (control + variants)
            - List: Specific value for each group [control, variant1, variant2, ...]
        effect : float or list
            Effect size(s) for each variant.
            - Single float: Same effect applied to all variants
            - List: Specific effect for each variant [effect1, effect2, ...]
        compliance : list
            List with compliance values.
        standard_deviation : list
            List of standard deviations by groups.
        allocation_ratio : list
            Proportion of total sample size for each group (control + variants).
            Must sum to 1.0. Default is equal allocation across all groups.
            Example: [0.3, 0.7] for 30% control, 70% treatment.
        target_comparisons : list of tuples, optional
            Which comparisons to power for. Defaults to all comparisons defined in the instance.
            Example: [(0, 1), (0, 2)] to only consider control vs variant 1 and 2.
        power_criteria : str
            How to handle multiple target comparisons:
            - "all": All target comparisons must reach their target power (conservative).
              Uses the sample size required by the most demanding comparison.
              NOTE: Other comparisons will likely EXCEED their target power.
            - "any": At least one target comparison reaches its target power (liberal)
        optimize_allocation : bool
            If True, automatically finds the optimal allocation ratio that minimizes
            total sample size while meeting all power targets. Ignores allocation_ratio
            parameter if provided. Default is False (uses equal or provided allocation).
        min_sample_size : int
            Minimum total sample size to consider (default: 100).
        max_sample_size : int
            Maximum total sample size to consider (default: 100000).
        tolerance : float
            Acceptable difference from target power (default: 0.01).
        step_size : int
            Step size for initial coarse search (default: 100).
        correction : str, optional
            Multiple comparison correction method. If provided, overrides the instance correction.
            Options: 'bonferroni', 'holm', 'hochberg', 'sidak', 'fdr', 'none' (or None).

        Returns
        -------
        pd.DataFrame
            DataFrame with comparison, target_power, sample_size, allocation, and achieved_power columns.

        Examples
        --------
        # Basic usage: same power target for all comparisons (scalar values)
        >>> p = PowerSim(metric='proportion', variants=1, nsim=500)
        >>> result = p.find_sample_size(target_power=0.80, baseline=0.10, effect=0.02)

        # Different power targets per comparison
        >>> p = PowerSim(metric='proportion', variants=2, nsim=500)
        >>> result = p.find_sample_size(
        ...     target_power={(0,1): 0.90, (0,2): 0.80},  # Primary comparison needs 90% power
        ...     baseline=0.10,  # Same baseline for all groups
        ...     effect=[0.05, 0.03]  # Different effects per variant
        ... )

        # Optimize allocation with same effect for all variants
        >>> p = PowerSim(metric='proportion', variants=3, nsim=500)
        >>> result = p.find_sample_size(
        ...     target_power=0.80,
        ...     baseline=0.10,
        ...     effect=0.05,  # Same effect for all variants
        ...     optimize_allocation=True  # Find optimal allocation!
        ... )

        # Power only specific comparisons
        >>> p = PowerSim(metric='proportion', variants=3, nsim=500)
        >>> result = p.find_sample_size(
        ...     target_power=0.80,
        ...     baseline=[0.10],
        ...     effect=[0.05, 0.03, 0.07],
        ...     target_comparisons=[(0, 1), (0, 2)]  # Only power first two variants
        ... )

        # Override correction method and use custom allocation
        >>> result = p.find_sample_size(
        ...     target_power=0.80,
        ...     baseline=0.10,
        ...     effect=0.02,
        ...     allocation_ratio=[0.3, 0.7],  # 30% control, 70% treatment
        ...     correction='none'  # Override instance correction
        ... )
        """
        original_correction = None
        if correction is not None:
            original_correction = self.correction
            self.correction = correction

        if baseline is None:
            baseline = [1.0]
        elif isinstance(baseline, int | float):
            baseline = [float(baseline)] * (self.variants + 1)

        if effect is None:
            effect = [0.10]
        elif isinstance(effect, int | float):
            effect = [float(effect)] * self.variants

        if compliance is None:
            compliance = [1.0]
        if standard_deviation is None:
            standard_deviation = [1]

        num_groups = self.variants + 1
        if allocation_ratio is None or optimize_allocation:
            target_comps = target_comparisons if target_comparisons is not None else self.comparisons
            control_needed = any(0 in comp for comp in target_comps)

            if not control_needed:
                variant_allocation = 1.0 / self.variants
                allocation_ratio = [0.0] + [variant_allocation] * self.variants
            else:
                allocation_ratio = [1.0 / num_groups] * num_groups

        if not optimize_allocation:
            if len(allocation_ratio) != num_groups:
                log_and_raise_error(
                    self.logger,
                    f"allocation_ratio must have {num_groups} elements (control + {self.variants} variants)",
                )

            if not np.isclose(sum(allocation_ratio), 1.0):
                log_and_raise_error(self.logger, "allocation_ratio must sum to 1.0")

            if any(r < 0 for r in allocation_ratio):
                log_and_raise_error(self.logger, "All allocation_ratio values must be non-negative")

            target_comps = target_comparisons if target_comparisons is not None else self.comparisons
            groups_used = set()
            for g1, g2 in target_comps:
                groups_used.add(g1)
                groups_used.add(g2)

            for group in groups_used:
                if group < len(allocation_ratio) and allocation_ratio[group] == 0:
                    log_and_raise_error(
                        self.logger, f"Group {group} is used in target_comparisons but has zero allocation"
                    )

        if target_comparisons is None:
            target_comparisons = self.comparisons
        else:
            normalized_comps = []
            for g1, g2 in target_comparisons:
                if g1 > g2:
                    normalized_comps.append((g2, g1))
                    if (g1, g2) not in self.comparisons:
                        self.logger.info(f"Auto-corrected comparison order: ({g1}, {g2}) → ({g2}, {g1})")
                else:
                    normalized_comps.append((g1, g2))
            target_comparisons = normalized_comps

            invalid_comps = [c for c in target_comparisons if c not in self.comparisons]
            if invalid_comps:
                log_and_raise_error(
                    self.logger,
                    f"target_comparisons {invalid_comps} are not in defined comparisons {self.comparisons}",
                )

        if power_criteria not in ["all", "any"]:
            log_and_raise_error(self.logger, "power_criteria must be 'all' or 'any'")

        if isinstance(target_power, dict):
            invalid_keys = [k for k in target_power.keys() if k not in target_comparisons]
            if invalid_keys:
                log_and_raise_error(self.logger, f"target_power dict contains invalid comparisons: {invalid_keys}")
            if any(not (0 < p < 1) for p in target_power.values()):
                log_and_raise_error(self.logger, "All target_power values must be between 0 and 1")
            power_dict = target_power
        else:
            if not 0 < target_power < 1:
                log_and_raise_error(self.logger, "target_power must be between 0 and 1")
            power_dict = {comp: target_power for comp in target_comparisons}

        if optimize_allocation:
            allocation_ratio = self._optimize_allocation(
                power_dict=power_dict,
                target_comparisons=target_comparisons,
                baseline=baseline,
                effect=effect,
                compliance=compliance,
                standard_deviation=standard_deviation,
                num_groups=num_groups,
                min_sample_size=min_sample_size,
                max_sample_size=max_sample_size,
                tolerance=tolerance,
                step_size=step_size,
            )

        results = []

        def _get_analytical_sample_size(metric, baseline_val, effect_val, sd_val, target_power, alpha, allocation):
            """Get analytical sample size estimate for initial guess."""
            from scipy.stats import norm

            # Get z-scores
            z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed
            z_beta = norm.ppf(target_power)

            try:
                if metric == "proportion":
                    # Two-sample proportion test
                    p1 = baseline_val
                    p2 = baseline_val + effect_val
                    p_pooled = (p1 + p2) / 2

                    # Clip to valid range
                    p1 = np.clip(p1, 0.001, 0.999)
                    p2 = np.clip(p2, 0.001, 0.999)
                    p_pooled = np.clip(p_pooled, 0.001, 0.999)

                    # Formula: n = (z_alpha*sqrt(2*p*(1-p)) + z_beta*sqrt(p1*(1-p1) + p2*(1-p2)))^2 / (p1-p2)^2
                    numerator = (
                        z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
                        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
                    ) ** 2
                    denominator = (p2 - p1) ** 2
                    n_per_group = numerator / denominator if denominator > 0 else None

                elif metric == "average":
                    # Two-sample t-test
                    # Formula: n = 2 * (z_alpha + z_beta)^2 * sd^2 / effect^2
                    if effect_val != 0:
                        n_per_group = 2 * ((z_alpha + z_beta) ** 2) * (sd_val**2) / (effect_val**2)
                    else:
                        n_per_group = None

                else:  # count
                    # Poisson test - use similar logic to proportion
                    lambda1 = baseline_val
                    lambda2 = baseline_val + effect_val
                    if lambda2 != lambda1:
                        n_per_group = ((z_alpha + z_beta) ** 2) * (lambda1 + lambda2) / ((lambda2 - lambda1) ** 2)
                    else:
                        n_per_group = None

                if n_per_group and n_per_group > 0:
                    # Adjust for allocation ratio (n_per_group is for equal allocation)
                    # Total sample size = n_per_group * 2 for equal allocation
                    # For unequal, use: n_total = n_per_group / (r1 * r2) where r1, r2 are allocation ratios
                    r1, r2 = allocation[group1], allocation[group2]
                    total_n = n_per_group / (r1 * r2) if (r1 * r2) > 0 else n_per_group * 2
                    return int(total_n)
            except (ZeroDivisionError, ValueError):
                pass

            return None

        # For each target comparison, find the required sample size
        for group1, group2 in target_comparisons:
            comp_target_power = power_dict[(group1, group2)]

            # Get analytical estimate for better starting point
            baseline_val = baseline[group2] if group2 < len(baseline) else baseline[0]
            if self.metric != "average" and group2 > 0 and group2 - 1 < len(effect):
                effect_val = effect[group2 - 1]
            else:
                effect_val = effect[0] if len(effect) > 0 else 0.1
            sd_val = standard_deviation[group2] if group2 < len(standard_deviation) else standard_deviation[0]

            analytical_estimate = _get_analytical_sample_size(
                self.metric, baseline_val, effect_val, sd_val, comp_target_power, self.alpha, allocation_ratio
            )

            # Start with full range - we'll use analytical estimate to guide initial coarse search
            low = min_sample_size
            high = max_sample_size

            # Use analytical estimate to set better starting point for coarse search
            if analytical_estimate and min_sample_size <= analytical_estimate <= max_sample_size:
                # Start coarse search near analytical estimate, but ensure we check below it too
                coarse_start = max(min_sample_size, int(analytical_estimate * 0.5))
                self.logger.debug(
                    f"Analytical estimate: {analytical_estimate}, starting coarse search at {coarse_start}"
                )
            else:
                coarse_start = min_sample_size

            best_total_size = None
            best_power = None

            # First do a coarse search to find a good starting range
            current_total_size = coarse_start
            while current_total_size <= max_sample_size:
                # Distribute total sample size by allocation ratio
                sample_sizes = [int(current_total_size * ratio) for ratio in allocation_ratio]

                try:
                    power_result = self.get_power(
                        baseline=baseline,
                        effect=effect,
                        sample_size=sample_sizes,
                        compliance=compliance,
                        standard_deviation=standard_deviation,
                        target_comparisons=[(group1, group2)],
                    )
                    current_power = power_result.iloc[0]["power"]
                except (RuntimeWarning, ZeroDivisionError, ValueError) as e:
                    if "divide by zero" in str(e).lower() or "invalid value" in str(e).lower():
                        log_and_raise_error(
                            self.logger,
                            f"Sample sizes too small causing numerical errors. "
                            f"Try increasing min_sample_size (current: {min_sample_size}). "
                            f"Suggested: min_sample_size >= {max(100, min_sample_size * 2)}",
                        )
                    raise

                if current_power >= comp_target_power - tolerance:
                    high = current_total_size
                    best_total_size = current_total_size
                    best_power = current_power
                    break

                low = current_total_size
                current_total_size += step_size

            # If we didn't find a solution in coarse search
            if best_total_size is None:
                # Check if we need to search below coarse_start
                if coarse_start > min_sample_size:
                    # The coarse search might have started too high - check min_sample_size first
                    sample_sizes = [int(min_sample_size * ratio) for ratio in allocation_ratio]
                    power_result = self.get_power(
                        baseline=baseline,
                        effect=effect,
                        sample_size=sample_sizes,
                        compliance=compliance,
                        standard_deviation=standard_deviation,
                        target_comparisons=[(group1, group2)],
                    )
                    min_power = power_result.iloc[0]["power"]

                    if min_power >= comp_target_power - tolerance:
                        # Solution is at or below min_sample_size
                        best_total_size = min_sample_size
                        best_power = min_power
                        low = min_sample_size
                        high = coarse_start
                    else:
                        # Solution is between min_sample_size and coarse_start
                        low = min_sample_size
                        high = coarse_start
                        # Do a coarse search in this range
                        current_total_size = min_sample_size
                        while current_total_size < coarse_start:
                            sample_sizes = [int(current_total_size * ratio) for ratio in allocation_ratio]
                            try:
                                power_result = self.get_power(
                                    baseline=baseline,
                                    effect=effect,
                                    sample_size=sample_sizes,
                                    compliance=compliance,
                                    standard_deviation=standard_deviation,
                                    target_comparisons=[(group1, group2)],
                                )
                                current_power = power_result.iloc[0]["power"]
                            except (RuntimeWarning, ZeroDivisionError, ValueError) as e:
                                if "divide by zero" in str(e).lower() or "invalid value" in str(e).lower():
                                    log_and_raise_error(
                                        self.logger,
                                        f"Sample sizes too small causing numerical errors. "
                                        f"Try increasing min_sample_size (current: {min_sample_size}). "
                                        f"Suggested: min_sample_size >= {max(100, min_sample_size * 2)}",
                                    )
                                raise

                            if current_power >= comp_target_power - tolerance:
                                high = current_total_size
                                best_total_size = current_total_size
                                best_power = current_power
                                break

                            low = current_total_size
                            current_total_size += step_size

                # If still no solution, truly can't achieve within max_sample_size
                if best_total_size is None:
                    self.logger.warning(
                        f"Could not achieve target power {comp_target_power} for comparison {(group1, group2)} "
                        f"within max_sample_size {max_sample_size}"
                    )
                    best_total_size = max_sample_size
                    sample_sizes = [int(max_sample_size * ratio) for ratio in allocation_ratio]
                    power_result = self.get_power(
                        baseline=baseline,
                        effect=effect,
                        sample_size=sample_sizes,
                        compliance=compliance,
                        standard_deviation=standard_deviation,
                        target_comparisons=[(group1, group2)],
                        use_early_stopping=False,  # Accurate calculation for reporting
                    )
                    best_power = power_result.iloc[0]["power"]
            else:
                # Refine with binary search
                while high - low > step_size // 2:
                    mid = (low + high) // 2
                    sample_sizes = [int(mid * ratio) for ratio in allocation_ratio]

                    try:
                        power_result = self.get_power(
                            baseline=baseline,
                            effect=effect,
                            sample_size=sample_sizes,
                            compliance=compliance,
                            standard_deviation=standard_deviation,
                            target_comparisons=[(group1, group2)],
                        )
                        current_power = power_result.iloc[0]["power"]
                    except (RuntimeWarning, ZeroDivisionError, ValueError) as e:
                        if "divide by zero" in str(e).lower() or "invalid value" in str(e).lower():
                            log_and_raise_error(
                                self.logger,
                                f"Sample sizes too small causing numerical errors. "
                                f"Try increasing min_sample_size (current: {min_sample_size}). "
                                f"Suggested: min_sample_size >= {max(100, min_sample_size * 2)}",
                            )
                        raise

                    # Accept if we're within tolerance of the target
                    if abs(current_power - comp_target_power) <= tolerance:
                        # Prefer solutions closer to target (don't overshoot too much)
                        if best_total_size is None or abs(current_power - comp_target_power) < abs(
                            best_power - comp_target_power
                        ):
                            best_total_size = mid
                            best_power = current_power
                        break
                    elif current_power < comp_target_power:
                        low = mid
                    else:
                        high = mid
                        # Update best if this is closer to target than previous best
                        if best_total_size is None or abs(current_power - comp_target_power) < abs(
                            best_power - comp_target_power
                        ):
                            best_total_size = mid
                            best_power = current_power

            # Calculate final allocation
            final_sample_sizes = [int(best_total_size * ratio) for ratio in allocation_ratio]

            results.append(
                {
                    "comparison": (group1, group2),
                    "target_power": comp_target_power,
                    "total_sample_size": best_total_size,
                    "allocation_ratio": str(allocation_ratio),
                    "sample_sizes": str(final_sample_sizes),
                    "achieved_power": best_power,
                }
            )

        df_results = pd.DataFrame(results)

        if power_criteria == "all":
            max_sample_idx = df_results["total_sample_size"].idxmax()
            max_sample_size_needed = df_results.loc[max_sample_idx, "total_sample_size"]
            limiting_comparison = df_results.loc[max_sample_idx, "comparison"]

            min_sample_needed = df_results["total_sample_size"].min()
            if max_sample_size_needed > min_sample_needed * 1.1:  # >10% difference
                self.logger.info(
                    f"Using sample size {max_sample_size_needed} (driven by {limiting_comparison}) "
                    f"to ensure ALL comparisons meet target power. "
                    f"Some comparisons will exceed their target power."
                )
        else:
            max_sample_idx = df_results["total_sample_size"].idxmin()
            max_sample_size_needed = df_results.loc[max_sample_idx, "total_sample_size"]
            limiting_comparison = df_results.loc[max_sample_idx, "comparison"]

        final_sample_sizes = [int(max_sample_size_needed * ratio) for ratio in allocation_ratio]

        power_result = self.get_power(
            baseline=baseline,
            effect=effect,
            sample_size=final_sample_sizes,
            compliance=compliance,
            standard_deviation=standard_deviation,
            use_early_stopping=False,  # Disable for final accurate calculation
        )

        sample_sizes_dict = {"control": final_sample_sizes[0]}
        for i in range(1, len(final_sample_sizes)):
            sample_sizes_dict[f"variant_{i}"] = final_sample_sizes[i]

        if optimize_allocation:
            self.logger.info(f"Optimized sample sizes: {sample_sizes_dict}")

        target_power_by_comparison = {}
        achieved_power_by_comparison = {}

        for _idx, row in power_result.iterrows():
            comp = row["comparisons"]
            if comp in target_comparisons:
                target_power_by_comparison[str(comp)] = round(power_dict[comp], 3)
                achieved_power_by_comparison[str(comp)] = round(row["power"], 3)

        self.logger.info(f"Achieved power: {achieved_power_by_comparison}")

        result = {
            "total_sample_size": max_sample_size_needed,
            "sample_sizes_by_group": sample_sizes_dict,
            "power_criteria": power_criteria,
            "correction": self.correction,
            "limiting_comparison": str(limiting_comparison),
            "target_power_by_comparison": target_power_by_comparison,
            "achieved_power_by_comparison": achieved_power_by_comparison,
        }

        if original_correction is not None:
            self.correction = original_correction

        return pd.DataFrame([result])

    def simulate_retrodesign(
        self,
        true_effect: float | list[float],
        sample_size: int | list[int],
        baseline: float | list[float] = None,
        compliance: float | list[float] = None,
        standard_deviation: float | list[float] = None,
        allocation_ratio: list[float] = None,
        target_comparisons: list[tuple[int, int]] = None,
        nsim: int | None = None,
        round_decimals: int = 4,
    ) -> pd.DataFrame:
        """
        Simulate retrodesign metrics (Type S and Type M errors) for a study design.

        This prospective analysis simulates what happens when you run a study with
        given parameters and only look at statistically significant results. It helps
        you understand:
        - How often you'll get the wrong sign (Type S error)
        - How much you'll overestimate the effect (Type M error / exaggeration ratio)
        - The distribution of significant effects

        This is the simulation-based version of Gelman & Carlin's retrodesign analysis,
        useful for planning studies and understanding risks of underpowered experiments.

        Parameters
        ----------
        true_effect : float or list
            The true effect size(s) to simulate. Can be a single value or list per variant.
        sample_size : int or list
            Sample size(s) per group. Can be a single value or list per group.
        baseline : float or list, optional
            Baseline rate for proportions/counts or mean for averages.
        compliance : float or list, optional
            Compliance rate(s). Can be a single value or list per variant.
        standard_deviation : float or list, optional
            Standard deviation(s). Can be a single value or list per group.
        allocation_ratio : list, optional
            Proportion of total sample size for each group (control + variants).
            Must sum to 1.0. Default is equal allocation across all groups.
            Example: [0.3, 0.7] for 30% control, 70% treatment.
        target_comparisons : list of tuples, optional
            Which comparisons to simulate for. Defaults to all comparisons defined in the instance.
            Example: [(0, 1), (0, 2)] to only consider control vs variant 1 and 2.
        nsim : int, optional
            Number of simulations. If None, uses self.nsim.
        round_decimals : int, optional
            Number of decimal places to round numeric columns to (default: 4).
            Set to None to disable rounding.

        Returns
        -------
        pd.DataFrame
            Results containing:
            - comparison: The comparison being made (e.g., "(0, 1)")
            - true_effect: The true effect size used in simulation
            - sample_size: The sample size used
            - power: Probability of achieving statistical significance
            - type_s_error: Type S error - probability of wrong sign when significant
            - exaggeration_ratio: Type M error - mean(|observed|/|true|) when significant (absolute values)
            - relative_bias: mean(observed/true) preserving signs (Jaksic et al. 2026)
              Typically lower than exaggeration_ratio because Type S errors offset overestimates
            - median_significant_effect: Median observed effect when significant
            - prop_overestimate: Proportion of significant results that overestimate true effect
            - effect_distribution: Dict with distribution stats of significant effects
              Contains: {'mean', 'median', 'std', 'q25', 'q75'}

        Examples
        --------
        >>> p = PowerSim(metric='proportion', variants=1, nsim=5000)
        >>> # Simulate what happens with underpowered study
        >>> retro = p.simulate_retrodesign(
        ...     true_effect=0.02,
        ...     sample_size=500,
        ...     baseline=0.10
        ... )
        >>> print(f"Power: {retro['power'].iloc[0]:.2f}")
        >>> print(f"Type S error: {retro['type_s_error'].iloc[0]:.3f}")
        >>> print(f"Exaggeration ratio: {retro['exaggeration_ratio'].iloc[0]:.2f}")
        >>>
        >>> # Compare with well-powered study
        >>> retro2 = p.simulate_retrodesign(
        ...     true_effect=0.02,
        ...     sample_size=5000,
        ...     baseline=0.10
        ... )
        >>> print(f"Exaggeration ratio: {retro2['exaggeration_ratio'].iloc[0]:.2f}")
        >>>
        >>> # Custom allocation ratio (30% control, 70% treatment)
        >>> retro = p.simulate_retrodesign(
        ...     true_effect=0.02,
        ...     sample_size=[150, 350],  # Total 500
        ...     baseline=0.10,
        ...     allocation_ratio=[0.3, 0.7]
        ... )
        >>>
        >>> # Only specific comparisons
        >>> p = PowerSim(metric='proportion', variants=3, nsim=5000)
        >>> retro = p.simulate_retrodesign(
        ...     true_effect=[0.02, 0.03, 0.04],
        ...     sample_size=1000,
        ...     baseline=0.10,
        ...     target_comparisons=[(0, 1), (0, 2)]  # Only first two variants
        ... )
        """
        nsim_val = nsim if nsim is not None else self.nsim

        true_effect = self.__ensure_list(true_effect)
        sample_size = self.__ensure_list(sample_size)
        baseline = self.__ensure_list(baseline) if baseline is not None else [None]

        if compliance is None:
            compliance = [1.0]
        else:
            compliance = self.__ensure_list(compliance)

        if standard_deviation is None:
            standard_deviation = [1.0]
        else:
            standard_deviation = self.__ensure_list(standard_deviation)

        num_groups = self.variants + 1
        if allocation_ratio is not None:
            if len(allocation_ratio) != num_groups:
                log_and_raise_error(
                    self.logger,
                    f"allocation_ratio must have {num_groups} elements (control + {self.variants} variants)",
                )
            if not np.isclose(sum(allocation_ratio), 1.0):
                log_and_raise_error(self.logger, "allocation_ratio must sum to 1.0")
            if any(r <= 0 for r in allocation_ratio):
                log_and_raise_error(self.logger, "All allocation_ratio values must be positive")

            if len(sample_size) == 1:
                total_n = sample_size[0]
                sample_size = [int(total_n * r) for r in allocation_ratio]
        else:
            if len(sample_size) == 1:
                sample_size = sample_size * num_groups

        if target_comparisons is None:
            target_comparisons = self.comparisons
        else:
            normalized_comps = []
            for g1, g2 in target_comparisons:
                if g1 > g2:
                    normalized_comps.append((g2, g1))
                    if (g1, g2) not in self.comparisons:
                        self.logger.debug(f"Auto-corrected comparison order: ({g1}, {g2}) → ({g2}, {g1})")
                else:
                    normalized_comps.append((g1, g2))
            target_comparisons = normalized_comps

            invalid_comps = [c for c in target_comparisons if c not in self.comparisons]
            if invalid_comps:
                log_and_raise_error(
                    self.logger,
                    f"target_comparisons {invalid_comps} are not in defined comparisons {self.comparisons}",
                )

        if len(true_effect) != self.variants:
            if len(true_effect) == 1:
                true_effect = true_effect * self.variants
            else:
                log_and_raise_error(
                    self.logger,
                    f"true_effect length ({len(true_effect)}) must match variants ({self.variants}) or be 1",
                )

        original_nsim = self.nsim
        self.nsim = nsim_val

        results = []

        try:
            # Decide whether to use parallel processing
            use_parallel = nsim_val >= 500

            for h, j in target_comparisons:
                if h == 0:
                    comp_true_effect = true_effect[j - 1]
                else:
                    comp_true_effect = true_effect[j - 1] - true_effect[h - 1]

                if use_parallel:
                    from joblib import Parallel, delayed

                    self.logger.debug(f"Running {nsim_val} retrodesign simulations in parallel...")

                    # Run simulations in parallel
                    simulation_results = Parallel(n_jobs=-1, backend="loky")(
                        delayed(self._run_single_retrodesign_simulation)(
                            baseline=baseline,
                            sample_size=sample_size,
                            true_effect=true_effect,
                            compliance=compliance,
                            standard_deviation=standard_deviation,
                            comp_true_effect=comp_true_effect,
                            h=h,
                            j=j,
                        )
                        for _ in range(nsim_val)
                    )

                    # Aggregate results
                    significant_effects = []
                    all_effects = []
                    sign_errors = 0
                    significant_count = 0

                    for observed_effect, _pvalue, is_significant in simulation_results:
                        all_effects.append(observed_effect)

                        if is_significant:
                            significant_count += 1
                            significant_effects.append(observed_effect)

                            if comp_true_effect != 0:
                                if np.sign(observed_effect) != np.sign(comp_true_effect):
                                    sign_errors += 1
                else:
                    # Sequential execution for small nsim
                    significant_effects = []
                    all_effects = []
                    sign_errors = 0
                    significant_count = 0

                    for _ in range(nsim_val):
                        observed_effect, pvalue, is_significant = self._run_single_retrodesign_simulation(
                            baseline=baseline,
                            sample_size=sample_size,
                            true_effect=true_effect,
                            compliance=compliance,
                            standard_deviation=standard_deviation,
                            comp_true_effect=comp_true_effect,
                            h=h,
                            j=j,
                        )

                        all_effects.append(observed_effect)

                        if is_significant:
                            significant_count += 1
                            significant_effects.append(observed_effect)

                            if comp_true_effect != 0:
                                if np.sign(observed_effect) != np.sign(comp_true_effect):
                                    sign_errors += 1

                power = significant_count / nsim_val
                type_s = sign_errors / significant_count if significant_count > 0 else np.nan

                if len(significant_effects) > 0 and comp_true_effect != 0:
                    exaggeration_ratios = [abs(eff / comp_true_effect) for eff in significant_effects]
                    mean_exaggeration = np.mean(exaggeration_ratios)

                    relative_bias_values = [eff / comp_true_effect for eff in significant_effects]
                    relative_bias = np.mean(relative_bias_values)

                    median_significant = np.median(significant_effects)

                    prop_overestimate = sum(abs(eff) > abs(comp_true_effect) for eff in significant_effects) / len(
                        significant_effects
                    )

                    effect_dist = {
                        "mean": np.mean(significant_effects),
                        "median": median_significant,
                        "std": np.std(significant_effects),
                        "q25": np.percentile(significant_effects, 25),
                        "q75": np.percentile(significant_effects, 75),
                    }
                else:
                    mean_exaggeration = np.nan
                    relative_bias = np.nan
                    median_significant = np.nan
                    prop_overestimate = np.nan
                    effect_dist = {}

                results.append(
                    {
                        "comparison": str((h, j)),
                        "true_effect": comp_true_effect,
                        "sample_size": sample_size[0] if len(sample_size) == 1 else sample_size,
                        "power": power,
                        "type_s_error": type_s,
                        "exaggeration_ratio": mean_exaggeration,
                        "relative_bias": relative_bias,
                        "median_significant_effect": median_significant,
                        "prop_overestimate": prop_overestimate,
                        "effect_distribution": effect_dist,
                    }
                )

        finally:
            self.nsim = original_nsim

        df = pd.DataFrame(results)

        if round_decimals is not None:
            numeric_cols = [
                "true_effect",
                "power",
                "type_s_error",
                "exaggeration_ratio",
                "relative_bias",
                "median_significant_effect",
                "prop_overestimate",
            ]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].round(round_decimals)

        return df

    def __ensure_list(self, value):
        """Convert single values to list for consistent handling"""
        if value is None:
            return [None]
        if isinstance(value, list | np.ndarray):
            return list(value)
        return [value]
