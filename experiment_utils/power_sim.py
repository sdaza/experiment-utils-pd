"""
PowerSim class for simulation of power analysis.
"""

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from multiprocess.pool import ThreadPool
from scipy import stats
from tqdm import tqdm

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
        parallel_strategy: str = "hybrid",
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
        parallel_strategy : str
            Parallelization strategy: 'hybrid' (default, 2-stage: threading+multiprocessing),
            'loky' (multiprocessing only), 'threading' (threading only), or 'sequential'.
            Hybrid is recommended for best performance: fast data generation (threading)
            followed by CPU-intensive statistical tests (multiprocessing).
        """

        self.logger = get_logger("Power Simulator")
        self.metric = metric
        self.relative_effect = relative_effect
        self.early_stopping = early_stopping
        self.early_stopping_precision = early_stopping_precision
        self.parallel_strategy = parallel_strategy

        # Handle default for variants
        if variants is None:
            self.logger.info("'variants' not specified. Defaulting to 1 (control + 1 variant).")
            self.variants = 1
        else:
            self.variants = variants

        # Validate variants
        if not isinstance(self.variants, int) or self.variants < 1:
            log_and_raise_error(self.logger, "'variants' must be an integer >= 1")

        if comparisons is None:
            self.comparisons = list(itertools.combinations(range(self.variants + 1), 2))
        else:
            self.comparisons = [tuple(sorted(c)) for c in comparisons]
        self.nsim = nsim
        self.alternative = alternative
        self.alpha = alpha
        self.correction = correction
        self.fdr_method = fdr_method

    def __run_experiment(
        self,
        baseline: float = None,
        sample_size: list[int] = None,
        effect: list[float] = None,
        compliance: list[float] = None,
        standard_deviation: list[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulate data to run power analysis.

        Parameters
        ----------
        baseline : float
            Baseline rate for counts or proportions, or base average for mean comparisons.
            Same value is used for all groups (consistent with randomized experiment design).
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
            baseline = 1.0

        if len(effect) != self.variants:
            if len(effect) > 1:
                log_and_raise_error(
                    self.logger, "Effects should be same length as the number of self.variants or length 1!"
                )
            effect = list(itertools.repeat(effect[0], self.variants))

        if len(compliance) != self.variants:
            if len(compliance) > 1:
                log_and_raise_error(
                    self.logger, "Compliance rates should be same length as the number of self.variants or length 1!"
                )
            compliance = list(itertools.repeat(compliance[0], self.variants))

        if len(standard_deviation) != self.variants + 1:
            if len(standard_deviation) > 1:
                log_and_raise_error(
                    self.logger,
                    "Standard deviations should be same length as the number of self.variants + 1 or length 1!",
                )
            standard_deviation = list(itertools.repeat(standard_deviation[0], self.variants + 1))

        if len(sample_size) != self.variants + 1:
            if len(sample_size) > 1:
                log_and_raise_error(
                    self.logger, "N should be same length as the number of self.variants + 1 or length 1!"
                )
            sample_size = list(itertools.repeat(sample_size[0], self.variants + 1))

        re = list(range(self.variants))

        # outputs
        dd = np.array([])

        # index of condition
        vv = np.array([])

        if self.metric == "count":
            c_data = np.random.poisson(baseline, sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline * (1.00 + effect[i])
                else:
                    re[i] = baseline + effect[i]
                t_data_c = np.random.poisson(re[i], int(np.round(sample_size[i + 1] * compliance[i])))
                t_data_nc = np.random.poisson(baseline, int(np.round(sample_size[i + 1] * (1 - compliance[i]))))
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == "proportion":
            c_data = np.random.binomial(n=1, size=int(sample_size[0]), p=baseline)
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline * (1.00 + effect[i])
                else:
                    re[i] = baseline + effect[i]

                t_data_c = np.random.binomial(n=1, size=int(np.round(sample_size[i + 1] * compliance[i])), p=re[i])
                t_data_nc = np.random.binomial(
                    n=1, size=int(np.round(sample_size[i + 1] * (1 - compliance[i]))), p=baseline
                )
                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        if self.metric == "average":
            c_data = np.random.normal(baseline, standard_deviation[0], sample_size[0])
            dd = c_data
            vv = list(itertools.repeat(0, len(c_data)))

            for i in range(self.variants):
                if self.relative_effect:
                    re[i] = baseline * (1.00 + effect[i])
                else:
                    re[i] = baseline + effect[i]

                t_data_c = np.random.normal(
                    re[i], standard_deviation[i + 1], int(np.round(sample_size[i + 1] * compliance[i]))
                )
                t_data_nc = np.random.normal(
                    baseline, standard_deviation[i + 1], int(np.round(sample_size[i + 1] * (1 - compliance[i])))
                )

                t_data = np.append(t_data_c, t_data_nc)
                dd = np.append(dd, t_data)
                vv = np.append(vv, list(itertools.repeat(i + 1, len(t_data))))

        return dd, vv

    def _normalize_and_validate_comparisons(
        self, target_comparisons: list[tuple[int, int]] | None
    ) -> list[tuple[int, int]]:
        """
        Normalize target_comparisons to canonical order (smaller, larger) and validate
        against self.comparisons. Returns self.comparisons if target_comparisons is None.
        """
        if target_comparisons is None:
            return self.comparisons

        normalized = [tuple(sorted(c)) for c in target_comparisons]
        invalid = [c for c in normalized if c not in self.comparisons]
        if invalid:
            log_and_raise_error(
                self.logger,
                f"target_comparisons {invalid} are not in defined comparisons {self.comparisons}",
            )
        return normalized

    def _generate_simulation_data(
        self,
        baseline: float,
        effect: list[float],
        sample_size: list[int],
        compliance: list[float],
        standard_deviation: list[float],
    ) -> dict:
        """
        Stage 1 (Hybrid): Generate simulation data (fast, numpy operations).
        This is lightweight and benefits from threading (shared memory).

        Returns
        -------
        dict
            Dictionary with 'y' and 'x' arrays for statistical tests
        """
        y, x = self.__run_experiment(
            baseline=baseline,
            sample_size=sample_size,
            effect=effect,
            compliance=compliance,
            standard_deviation=standard_deviation,
        )
        return {"y": y, "x": x}

    def _compute_statistical_tests(
        self,
        sim_data: dict,
        sample_size: list[int],
        comparisons_to_compute: list[tuple[int, int]],
    ) -> list[bool]:
        """
        Stage 2 (Hybrid): Run statistical tests on generated data (CPU-intensive).
        This is computation-heavy and benefits from multiprocessing (true parallelism).

        Parameters
        ----------
        sim_data : dict
            Dictionary with 'y' and 'x' arrays from _generate_simulation_data
        sample_size : list[int]
            Sample sizes (for bounds checking)
        comparisons_to_compute : list[tuple[int, int]]
            Comparisons to compute

        Returns
        -------
        list[bool]
            List of significance results for each comparison
        """
        y = sim_data["y"]
        x = sim_data["x"]

        l_pvalues = []
        for j, h in comparisons_to_compute:
            # Skip comparison if either group has 0 sample size
            if j >= len(sample_size) or h >= len(sample_size) or sample_size[j] == 0 or sample_size[h] == 0:
                l_pvalues.append(1.0)
                continue

            # Run statistical test based on metric type
            if self.metric == "count":
                ty = np.append(y[np.isin(x, j)], y[np.isin(x, h)])
                tx = np.append(x[np.isin(x, j)], x[np.isin(x, h)])
                tx[np.isin(tx, j)] = 0
                tx[np.isin(tx, h)] = 1

                model = sm.Poisson(ty, sm.add_constant(tx))
                pm = model.fit(disp=False)
                pvalue = pm.pvalues[1]

            elif self.metric == "proportion":
                _, pvalue = sm.stats.proportions_ztest(
                    [np.sum(y[np.isin(x, h)]), np.sum(y[np.isin(x, j)])],
                    [len(y[np.isin(x, h)]), len(y[np.isin(x, j)])],
                )

            elif self.metric == "average":
                _, pvalue = stats.ttest_ind(y[np.isin(x, h)], y[np.isin(x, j)], equal_var=False)

            l_pvalues.append(pvalue)

        # Apply correction and determine significance
        pvalue_adjustment = {"two-tailed": 1, "greater": 0.5, "smaller": 0.5}
        correction_methods = {
            "bonferroni": self.bonferroni,
            "holm": self.holm_bonferroni,
            "hochberg": self.hochberg,
            "sidak": self.sidak,
            "fdr": self.lsu,
        }

        if self.correction in correction_methods:
            if len(self.comparisons) == 1:
                significant = [l_pvalues[0] < self.alpha / pvalue_adjustment[self.alternative]]
            elif len(l_pvalues) < len(self.comparisons):
                adjusted_alpha = self.alpha / pvalue_adjustment[self.alternative]
                if self.correction == "bonferroni":
                    significant = [p < adjusted_alpha / len(self.comparisons) for p in l_pvalues]
                elif self.correction == "sidak":
                    sidak_alpha = 1 - (1 - adjusted_alpha) ** (1 / len(self.comparisons))
                    significant = [p < sidak_alpha for p in l_pvalues]
                else:
                    # Sequential methods (Holm, Hochberg, FDR) need all p-values to rank them;
                    # with a subset use Bonferroni (full family size) as a conservative fallback.
                    significant = [p < adjusted_alpha / len(self.comparisons) for p in l_pvalues]
            else:
                correction_func = correction_methods[self.correction]
                significant = correction_func(l_pvalues, self.alpha / pvalue_adjustment[self.alternative])
        elif self.correction is None or self.correction == "none":
            # No correction
            significant = [p < self.alpha / pvalue_adjustment[self.alternative] for p in l_pvalues]
        else:
            log_and_raise_error(self.logger, f"Correction method {self.correction} not recognized!")

        return significant

    def _run_single_power_simulation(
        self,
        baseline: float,
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
        baseline : float
            Baseline value (same for all groups)
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
        baseline: float,
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
        baseline : float
            Baseline value (same for all groups)
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

    def _run_retrodesign_test_on_data(
        self,
        sim_data: dict,
        h: int,
        j: int,
    ) -> tuple[float, float, bool]:
        """
        Stage 2 (Hybrid): Run retrodesign test on generated simulation data.
        Used for hybrid parallelization in retrodesign simulations.

        Parameters
        ----------
        sim_data : dict
            Dictionary with 'y' and 'x' arrays from _generate_simulation_data
        h : int
            Control group index
        j : int
            Treatment group index

        Returns
        -------
        tuple[float, float, bool]
            (observed_effect, pvalue, is_significant)
        """
        y = sim_data["y"]
        x = sim_data["x"]

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
        baseline: float = None,
        effect: float | list[float] = None,
        sample_size: int | list[int] = None,
        compliance: float | list[float] = None,
        standard_deviation: float | list[float] = None,
        target_comparisons: list[tuple[int, int]] = None,
        use_early_stopping: bool = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Estimate power using simulation.

        Parameters
        ----------
        baseline : float
            Baseline rate for counts or proportions, or base average for mean comparisons.
            Same value is used for all groups (consistent with randomized experiment design).
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
            baseline = 1.0
        elif isinstance(baseline, list | tuple):
            baseline = float(baseline[0])
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
        comparisons_to_compute = self._normalize_and_validate_comparisons(target_comparisons)

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

            # Choose parallelization strategy
            if self.parallel_strategy == "hybrid":
                # HYBRID: Two-stage pipeline for maximum efficiency
                # Stage 1: Generate all simulation data (threading - shares memory, fast)
                # Stage 2: Run statistical tests (multiprocessing - true parallelism, CPU-intensive)
                if show_progress:
                    self.logger.info(
                        f"Running {self.nsim} power simulations with hybrid parallelization "
                        f"(threading for data generation + multiprocessing for tests)..."
                    )

                # Stage 1: Generate simulation datasets in parallel (threading)
                self.logger.debug("Stage 1/2: Generating simulation data (threading)...")
                sim_datasets = Parallel(n_jobs=-1, backend="threading", batch_size=50)(
                    delayed(self._generate_simulation_data)(
                        baseline=baseline,
                        effect=effect,
                        sample_size=sample_size,
                        compliance=compliance,
                        standard_deviation=standard_deviation,
                    )
                    for _ in range(self.nsim)
                )

                # Stage 2: Run statistical tests in parallel (multiprocessing)
                self.logger.debug("Stage 2/2: Computing statistical tests (multiprocessing)...")
                results = list(
                    tqdm(
                        Parallel(n_jobs=-1, backend="loky", batch_size=50, return_as="generator")(
                            delayed(self._compute_statistical_tests)(
                                sim_data=sim_data,
                                sample_size=sample_size,
                                comparisons_to_compute=comparisons_to_compute,
                            )
                            for sim_data in sim_datasets
                        ),
                        total=self.nsim,
                        desc="Simulations",
                        unit="sim",
                        leave=False,
                        disable=not show_progress,
                    )
                )

            elif self.parallel_strategy == "threading":
                # Pure threading approach (memory-efficient but GIL-limited)
                if show_progress:
                    self.logger.info(f"Running {self.nsim} power simulations in parallel (threading)...")
                results = list(
                    tqdm(
                        Parallel(n_jobs=-1, backend="threading", return_as="generator")(
                            delayed(self._run_single_power_simulation)(
                                baseline=baseline,
                                effect=effect,
                                sample_size=sample_size,
                                compliance=compliance,
                                standard_deviation=standard_deviation,
                                comparisons_to_compute=comparisons_to_compute,
                            )
                            for _ in range(self.nsim)
                        ),
                        total=self.nsim,
                        desc="Simulations",
                        unit="sim",
                        leave=False,
                        disable=not show_progress,
                    )
                )

            else:  # "loky" or default
                # Pure multiprocessing approach (high memory but true parallelism)
                if show_progress:
                    self.logger.info(f"Running {self.nsim} power simulations in parallel (multiprocessing)...")
                results = list(
                    tqdm(
                        Parallel(n_jobs=-1, backend="loky", return_as="generator")(
                            delayed(self._run_single_power_simulation)(
                                baseline=baseline,
                                effect=effect,
                                sample_size=sample_size,
                                compliance=compliance,
                                standard_deviation=standard_deviation,
                                comparisons_to_compute=comparisons_to_compute,
                            )
                            for _ in range(self.nsim)
                        ),
                        total=self.nsim,
                        desc="Simulations",
                        unit="sim",
                        leave=False,
                        disable=not show_progress,
                    )
                )

            # Aggregate results
            for significant in results:
                for v, p in enumerate(significant):
                    pvalues[v].append(p)
        else:
            # Sequential execution (needed for early stopping or small nsim)
            if self.nsim >= 500:
                self.logger.debug(f"Running {self.nsim} power simulations sequentially (early stopping enabled)...")

            # Early stopping parameters
            check_interval = 20  # Check every 20 simulations
            min_sims = 50  # Minimum simulations before considering early stopping

            # iterate over simulations
            for _i in tqdm(range(self.nsim), desc="Simulations", unit="sim", leave=False, disable=not show_progress):
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

    @staticmethod
    def _as_scenario_list(values):
        """Convert a grid parameter to a flat list of per-scenario values.

        Handles:
        - scalar              → [scalar]
        - [scalar, ...]       → [scalar, ...] (each element is one scenario)
        - [[v1,v2], [v3,v4]]  → [[v1,v2], [v3,v4]] (each sub-list is one scenario)
        Single-element inner lists are unwrapped so ``[[0.33]]`` becomes ``[0.33]``
        and ``[[1000], [2000]]`` becomes ``[1000, 2000]``.
        """
        if values is None:
            return None
        if not isinstance(values, list):
            return [values]
        return [v[0] if isinstance(v, list | tuple) and len(v) == 1 else v for v in values]

    def grid_sim_power(
        self,
        baseline_rates: float | list = None,
        effects: float | list = None,
        sample_sizes: int | list = None,
        compliances: float | list = None,
        standard_deviations: float | list = None,
        allocation_ratio: list[float] = None,
        threads: int = 3,
        plot: bool = False,
        correction: str = None,
        facet_by: str | None = "comparison",
        hue: str = "effect",
    ) -> pd.DataFrame:  # noqa: E501
        """
        Return Pandas DataFrame with parameter combinations and statistical power.

        Each parameter can be provided as a scalar, a flat list of scenario values,
        or a list of lists (one sub-list per scenario).  Single-element inner lists
        are unwrapped automatically, so both ``[[0.33]]`` and ``[0.33]`` are valid
        for a single baseline scenario.

        Parameters
        ----------
        baseline_rates : float or list of float
            Baseline rate scenarios (e.g., ``0.10`` or ``[0.10, 0.20]``).
        effects : float, list of float, or list of lists
            Effect size scenarios.  Use a list of lists for multi-variant effects,
            e.g. ``[[0.01, 0.03], [0.03, 0.05]]`` for two scenarios with two variants.
        sample_sizes : int, list of int, or list of lists
            Sample size scenarios, **per group**.  A scalar or flat list such as
            ``[1000, 2000, 5000]`` assigns that many units to *each* group (control
            and every variant), so ``5000`` means 5 000 in control *and* 5 000 in
            each variant (total N = 5 000 × (variants + 1)).  Use a list of lists to
            set per-group sizes explicitly, e.g. ``[[1000, 500], [2000, 1000]]``.
        compliances : float, list of float, or list of lists
            Compliance scenarios.
        standard_deviations : float, list of float, or list of lists
            Standard deviation scenarios.
        allocation_ratio : list of float, optional
            Proportion of total sample assigned to each group (control + variants).
            Must sum to 1.0 and have length equal to variants + 1.
            Defaults to equal allocation. Example: ``[0.5, 0.25, 0.25]`` gives the
            control group twice as many units as each variant.
        threads : int
            Number of threads for parallelization.
        plot : bool
            Whether to plot the results after the grid is computed.
        correction : str, optional
            Multiple comparison correction method. If provided, overrides the instance
            correction for this call. Options: 'bonferroni', 'holm', 'hochberg',
            'sidak', 'fdr', 'none' (or None to use the instance default).
        facet_by : str or None
            Column whose unique values each get their own plot when ``plot=True``
            (default ``"comparison"``).  Pass ``None`` to produce a single combined plot.
        hue : str
            Column used to colour lines within each plot when ``plot=True``
            (default ``"effect"``).
        """

        original_correction = None
        if correction is not None:
            original_correction = self.correction
            self.correction = correction

        baseline_rates = self._as_scenario_list(baseline_rates)
        effects = self._as_scenario_list(effects)
        sample_sizes = self._as_scenario_list(sample_sizes)

        if compliances is None:
            compliances = [1]
        else:
            compliances = self._as_scenario_list(compliances)

        if standard_deviations is None:
            standard_deviations = [1]
        else:
            standard_deviations = self._as_scenario_list(standard_deviations)

        # Validate allocation_ratio
        num_groups = self.variants + 1
        custom_allocation = allocation_ratio is not None
        if custom_allocation:
            if len(allocation_ratio) != num_groups:
                log_and_raise_error(
                    self.logger,
                    f"allocation_ratio length ({len(allocation_ratio)}) must equal variants + 1 ({num_groups})",
                )
            if abs(sum(allocation_ratio) - 1.0) > 1e-6:
                log_and_raise_error(self.logger, "allocation_ratio must sum to 1.0")
            # Split scalar totals into per-group lists only for custom allocations.
            # get_power already expands a scalar total uniformly, so for equal
            # allocation we keep the original values (cleaner x-axis labels).
            sample_sizes_for_power = [
                [int(s * r) for r in allocation_ratio] if isinstance(s, int | float) else s for s in sample_sizes
            ]
        else:
            allocation_ratio = [1.0 / num_groups] * num_groups
            sample_sizes_for_power = sample_sizes

        # Build the grid using the values passed to get_power, then restore the
        # original (display-friendly) sample sizes for the output DataFrame.
        pdict = {
            "baseline": baseline_rates,
            "effect": effects,
            "sample_size": sample_sizes_for_power,
            "compliance": compliances,
            "standard_deviation": standard_deviations,
        }
        grid = self.__expand_grid(pdict)

        if custom_allocation:
            # Replace per-group lists with original totals so labels stay readable.
            display_pdict = {**pdict, "sample_size": sample_sizes}
            grid["sample_size"] = self.__expand_grid(display_pdict)["sample_size"]

        parameters = list(grid.itertuples(index=False, name=None))

        grid["nsim"] = self.nsim
        grid["alpha"] = self.alpha
        grid["alternative"] = self.alternative
        grid["metric"] = self.metric
        grid["variants"] = self.variants
        grid["comparisons"] = str(self.comparisons)
        grid["correction"] = self.correction if self.correction is not None else "none"
        grid["allocation_ratio"] = str(allocation_ratio)
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
                "correction",
                "allocation_ratio",
                "nsim",
                "alpha",
                "alternative",
                "metric",
                "relative_effect",
            ],
        ]
        import functools

        _get_power_silent = functools.partial(self.get_power, show_progress=False)
        pool = ThreadPool(processes=threads)
        results = pool.starmap(_get_power_silent, parameters)
        pool.close()
        pool.join()

        if original_correction is not None:
            self.correction = original_correction

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
        # Keep sample_size and effect as their original types for correct numeric
        # axis ordering in plots; map to str only when the value is a list
        # (custom per-group allocation) so it can be stored in the DataFrame cell.
        grid.sample_size = grid.sample_size.map(lambda v: str(v) if isinstance(v, list) else v)
        grid.effect = grid.effect.map(lambda v: str(v) if isinstance(v, list) else v)
        if plot:
            self.plot_power(grid, facet_by=facet_by, hue=hue)
        return grid

    def plot_power(
        self,
        data: pd.DataFrame,
        facet_by: str | None = "comparison",
        hue: str = "effect",
    ) -> None:
        """
        Plot statistical power by scenario.

        Parameters
        ----------
        data : pd.DataFrame
            Output of :meth:`grid_sim_power`.
        facet_by : str or None
            Column whose unique values each get their own plot
            (default ``"comparison"``).  Pass ``None`` for a single combined plot.
        hue : str
            Column used to colour lines within each plot (default ``"effect"``).
        """
        from .plotting import plot_power as _plot_power

        _plot_power(
            data=data,
            comparisons=self.comparisons,
            alpha=self.alpha,
            alternative=self.alternative,
            facet_by=facet_by,
            hue=hue,
        )

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
        baseline: float,
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

        # Collect all groups that appear in any comparison (used to floor their sizes)
        groups_needed = set()
        for comp in target_comparisons:
            groups_needed.update(comp)

        # For each comparison, find minimum per-group sample size.
        # Note: We need to test with ALL groups to properly account for multiple
        # comparison corrections (e.g. Holm).
        for group1, group2 in sorted_comparisons:
            comp_target_power = power_dict[(group1, group2)]
            comp_result_idx = self.comparisons.index((group1, group2))

            # ---- local helpers ----------------------------------------
            # Bind loop variables as defaults to avoid B023 (closure over loop var)
            def _make_sizes(
                total_for_pair: int,
                _g1: int = group1,
                _g2: int = group2,
            ) -> list[int]:
                """Build a full sample-size list for a candidate total_for_pair."""
                per_g = max(100, int(total_for_pair / 2))
                sizes = list(group_samples)
                sizes[_g1] = max(sizes[_g1], per_g)
                sizes[_g2] = max(sizes[_g2], per_g)
                for i in range(num_groups):
                    if i in groups_needed:
                        sizes[i] = max(sizes[i], 100)
                return sizes

            def _eval_alloc_power(
                total_for_pair: int,
                use_early_stopping=None,
                _idx: int = comp_result_idx,
            ) -> float:
                sizes = _make_sizes(total_for_pair)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=RuntimeWarning)
                        res = self.get_power(
                            baseline=baseline,
                            effect=effect,
                            sample_size=sizes,
                            compliance=compliance,
                            standard_deviation=standard_deviation,
                            show_progress=False,
                            use_early_stopping=use_early_stopping,
                        )
                    return res.iloc[_idx]["power"]
                except Exception:
                    return 0.0

            # ---- analytical estimate for initial bracket --------------
            from scipy.stats import norm as _norm

            z_alpha = _norm.ppf(1 - self.alpha / 2)
            z_beta = _norm.ppf(comp_target_power)
            eff_val = (
                effect[group2 - 1]
                if (self.metric != "average" and group2 > 0 and group2 - 1 < len(effect))
                else (effect[0] if effect else 0.1)
            )
            sd_val = standard_deviation[group2] if group2 < len(standard_deviation) else standard_deviation[0]
            n_analytical_pair = None
            try:
                if self.metric == "proportion":
                    p1 = np.clip(baseline, 0.001, 0.999)
                    p2 = np.clip(baseline + eff_val, 0.001, 0.999)
                    pp = np.clip((p1 + p2) / 2, 0.001, 0.999)
                    num = (z_alpha * np.sqrt(2 * pp * (1 - pp)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
                    denom = (p2 - p1) ** 2
                    if denom > 0:
                        n_analytical_pair = int(num / denom) * 2  # *2 for total pair
                elif self.metric == "average":
                    if eff_val != 0:
                        n_analytical_pair = int(2 * ((z_alpha + z_beta) ** 2) * (sd_val**2) / (eff_val**2)) * 2
                else:
                    lam1, lam2 = baseline, baseline + eff_val
                    if lam2 != lam1:
                        n_analytical_pair = int(((z_alpha + z_beta) ** 2) * (lam1 + lam2) / ((lam2 - lam1) ** 2)) * 2
            except (ZeroDivisionError, ValueError):
                pass

            # ---- bracket ----------------------------------------------
            if n_analytical_pair and min_sample_size <= n_analytical_pair <= max_sample_size:
                lo = max(min_sample_size, int(n_analytical_pair * 0.5))
                hi = min(max_sample_size, int(n_analytical_pair * 2.0))
            else:
                lo = min_sample_size
                hi = max_sample_size

            # Verify upper bound
            p_hi = _eval_alloc_power(hi)
            if p_hi < comp_target_power - tolerance:
                while hi < max_sample_size:
                    lo = hi
                    hi = min(max_sample_size, hi * 2)
                    p_hi = _eval_alloc_power(hi)
                    if p_hi >= comp_target_power - tolerance:
                        break
                if p_hi < comp_target_power - tolerance:
                    group_samples[group1] = max(group_samples[group1], max(100, int(hi / 2)))
                    group_samples[group2] = max(group_samples[group2], max(100, int(hi / 2)))
                    continue

            # Contract lower bound if already sufficient
            p_lo = _eval_alloc_power(lo)
            if p_lo >= comp_target_power - tolerance:
                while lo > min_sample_size:
                    candidate = max(min_sample_size, lo // 2)
                    p_cand = _eval_alloc_power(candidate)
                    if p_cand >= comp_target_power - tolerance:
                        hi, p_hi = lo, p_lo
                        lo, p_lo = candidate, p_cand
                    else:
                        lo = candidate
                        break
                if lo == min_sample_size and p_lo >= comp_target_power - tolerance:
                    found_size = max(100, int(lo / 2))
                    group_samples[group1] = max(group_samples[group1], found_size)
                    group_samples[group2] = max(group_samples[group2], found_size)
                    continue

            # ---- binary search ----------------------------------------
            best_pair_size = hi
            while hi - lo > step_size:
                mid = (lo + hi) // 2
                if _eval_alloc_power(mid) >= comp_target_power - tolerance:
                    best_pair_size = mid
                    hi = mid
                else:
                    lo = mid

            # ---- post-convergence verification + recovery -------------
            verified = _eval_alloc_power(best_pair_size, use_early_stopping=False)
            _recovery = 0
            while verified < comp_target_power - tolerance and _recovery < 10:
                if best_pair_size >= max_sample_size:
                    break
                best_pair_size = min(max_sample_size, best_pair_size + step_size)
                verified = _eval_alloc_power(best_pair_size, use_early_stopping=False)
                _recovery += 1

            found_size = max(100, int(best_pair_size / 2))
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
        baseline: float = None,
        effect: float | list[float] = None,
        compliance: float | list[float] = None,
        standard_deviation: float | list[float] = None,
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
        baseline : float
            Baseline rate for counts or proportions, or base average for mean comparisons.
            Same value is used for all groups (consistent with randomized experiment design).
        effect : float or list
            Effect size(s) for each variant.
            - Single float: Same effect applied to all variants
            - List: Specific effect for each variant [effect1, effect2, ...]
        compliance : float or list
            Compliance value(s). Provide a single float or list per variant.
        standard_deviation : float or list
            Standard deviation(s) for control and variants. Provide a single float or list per group.
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
        ...     baseline=0.10,
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
            baseline = 1.0

        if effect is None:
            effect = [0.10]
        elif isinstance(effect, int | float):
            effect = [float(effect)] * self.variants

        if compliance is None:
            compliance = [1.0]
        elif isinstance(compliance, int | float):
            compliance = [float(compliance)]
        if standard_deviation is None:
            standard_deviation = [1]
        elif isinstance(standard_deviation, int | float):
            standard_deviation = [float(standard_deviation)]

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

        target_comparisons = self._normalize_and_validate_comparisons(target_comparisons)

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

        self.logger.info(
            f"Finding sample size for {len(target_comparisons)} comparison(s) "
            f"using {self.nsim} simulations per evaluation..."
        )

        results = []

        def _get_analytical_sample_size(metric, baseline_val, effect_val, sd_val, target_pwr, alpha, allocation):
            """Closed-form sample size estimate — used only to initialise the search bracket."""
            from scipy.stats import norm

            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta = norm.ppf(target_pwr)

            try:
                if metric == "proportion":
                    p1 = np.clip(baseline_val, 0.001, 0.999)
                    p2 = np.clip(baseline_val + effect_val, 0.001, 0.999)
                    p_pooled = np.clip((p1 + p2) / 2, 0.001, 0.999)
                    numerator = (
                        z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
                        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
                    ) ** 2
                    denom = (p2 - p1) ** 2
                    n_per_group = numerator / denom if denom > 0 else None

                elif metric == "average":
                    n_per_group = (
                        2 * ((z_alpha + z_beta) ** 2) * (sd_val**2) / (effect_val**2) if effect_val != 0 else None
                    )

                else:  # count / Poisson
                    lam1, lam2 = baseline_val, baseline_val + effect_val
                    n_per_group = (
                        ((z_alpha + z_beta) ** 2) * (lam1 + lam2) / ((lam2 - lam1) ** 2) if lam2 != lam1 else None
                    )

                if n_per_group and n_per_group > 0:
                    r1, r2 = allocation[group1], allocation[group2]
                    total_n = n_per_group / (r1 * r2) if (r1 * r2) > 0 else n_per_group * 2
                    return int(total_n)
            except (ZeroDivisionError, ValueError):
                pass
            return None

        # Warn early if nsim is too low for reliable search
        _RELIABLE_NSIM = 500
        if self.nsim < _RELIABLE_NSIM:
            self.logger.warning(
                f"nsim={self.nsim} is low — power estimates will be noisy (SE ≈ "
                f"{np.sqrt(0.8 * 0.2 / self.nsim):.3f} per evaluation). "
                f"Increase nsim to {_RELIABLE_NSIM}+ for more reliable results."
            )

        def _eval_power(n_total, use_early_stopping=None):
            """Simulate power at n_total. Suppresses inner progress bars."""
            sizes = [int(n_total * r) for r in allocation_ratio]
            try:
                res = self.get_power(
                    baseline=baseline,
                    effect=effect,
                    sample_size=sizes,
                    compliance=compliance,
                    standard_deviation=standard_deviation,
                    show_progress=False,
                    use_early_stopping=use_early_stopping,
                )
                return res.iloc[comp_result_idx]["power"]
            except (RuntimeWarning, ZeroDivisionError, ValueError) as exc:
                if "divide by zero" in str(exc).lower() or "invalid value" in str(exc).lower():
                    log_and_raise_error(
                        self.logger,
                        f"Sample sizes too small causing numerical errors. "
                        f"Try increasing min_sample_size (current: {min_sample_size}). "
                        f"Suggested: min_sample_size >= {max(100, min_sample_size * 2)}",
                    )
                raise

        # For each target comparison, find the required sample size
        for group1, group2 in tqdm(target_comparisons, desc="Finding sample size", unit="comp"):
            comp_target_power = power_dict[(group1, group2)]
            # Index into self.comparisons so we extract the right row when get_power
            # returns results for ALL comparisons (required for sequential corrections
            # like Holm to be applied across the full family of tests).
            comp_result_idx = self.comparisons.index((group1, group2))

            # Analytical starting estimate (no simulations, O(1))
            baseline_val = baseline
            effect_val = (
                effect[group2 - 1]
                if (self.metric != "average" and group2 > 0 and group2 - 1 < len(effect))
                else (effect[0] if effect else 0.1)
            )
            sd_val = standard_deviation[group2] if group2 < len(standard_deviation) else standard_deviation[0]
            n_analytical = _get_analytical_sample_size(
                self.metric, baseline_val, effect_val, sd_val, comp_target_power, self.alpha, allocation_ratio
            )

            # ----------------------------------------------------------------
            # Phase 1 — Bracket: find [low, high] where
            #   power(low) < target - tolerance   (insufficient)
            #   power(high) >= target - tolerance  (sufficient)
            #
            # Strategy (all O(log n), no linear scan):
            #   a. Seed bracket from analytical estimate (±50 %)
            #   b. If estimate > max_sample_size, skip directly to max check
            #   c. Expand upward with doubling if high is still insufficient
            #   d. Contract downward with halving if low is already sufficient
            # ----------------------------------------------------------------
            n_evals = 0

            if n_analytical and n_analytical <= max_sample_size:
                lo = max(min_sample_size, int(n_analytical * 0.5))
                hi = min(max_sample_size, int(n_analytical * 2.0))
            else:
                lo = min_sample_size
                hi = max_sample_size

            self.logger.debug(f"Comparison {(group1, group2)}: analytical={n_analytical}, bracket=[{lo}, {hi}]")

            # Check upper bound of bracket
            p_hi = _eval_power(hi)
            n_evals += 1

            if p_hi < comp_target_power - tolerance:
                # Insufficient at hi → expand upward with doubling
                while hi < max_sample_size:
                    lo = hi
                    hi = min(max_sample_size, hi * 2)
                    p_hi = _eval_power(hi)
                    n_evals += 1
                    if p_hi >= comp_target_power - tolerance:
                        break

                if p_hi < comp_target_power - tolerance:
                    # Cannot reach target within max_sample_size
                    self.logger.warning(
                        f"Could not achieve target power {comp_target_power} for comparison "
                        f"{(group1, group2)} within max_sample_size {max_sample_size} "
                        f"(achieved {p_hi:.3f} at {max_sample_size:,})"
                    )
                    results.append(
                        {
                            "comparison": (group1, group2),
                            "target_power": comp_target_power,
                            "total_sample_size": max_sample_size,
                            "allocation_ratio": str(allocation_ratio),
                            "sample_sizes": str([int(max_sample_size * r) for r in allocation_ratio]),
                            "achieved_power": p_hi,
                        }
                    )
                    self.logger.debug(f"Comparison {(group1, group2)}: {n_evals} evaluations (failed to converge)")
                    continue

            # Check lower bound — contract downward if already sufficient
            p_lo = _eval_power(lo)
            n_evals += 1

            if p_lo >= comp_target_power - tolerance:
                # Already sufficient at lo → halve downward to find tighter lower bound
                while lo > min_sample_size:
                    candidate = max(min_sample_size, lo // 2)
                    p_candidate = _eval_power(candidate)
                    n_evals += 1
                    if p_candidate >= comp_target_power - tolerance:
                        hi, p_hi = lo, p_lo
                        lo, p_lo = candidate, p_candidate
                    else:
                        lo = candidate  # insufficient here → good lower bound
                        break
                if lo == min_sample_size and p_lo >= comp_target_power - tolerance:
                    # min_sample_size itself is sufficient — done
                    results.append(
                        {
                            "comparison": (group1, group2),
                            "target_power": comp_target_power,
                            "total_sample_size": min_sample_size,
                            "allocation_ratio": str(allocation_ratio),
                            "sample_sizes": str([int(min_sample_size * r) for r in allocation_ratio]),
                            "achieved_power": p_lo,
                        }
                    )
                    self.logger.debug(f"Comparison {(group1, group2)}: {n_evals} evaluations")
                    continue

            # ----------------------------------------------------------------
            # Phase 2 — Binary search in [lo, hi]
            # Invariant: power(lo) < target, power(hi) >= target
            # ----------------------------------------------------------------
            best_total_size = hi
            best_power = p_hi

            while hi - lo > step_size:
                mid = (lo + hi) // 2
                p_mid = _eval_power(mid)
                n_evals += 1
                if p_mid >= comp_target_power - tolerance:
                    best_total_size = mid
                    best_power = p_mid
                    hi = mid
                else:
                    lo = mid

            self.logger.debug(f"Comparison {(group1, group2)}: converged in {n_evals} evaluations")

            # ------------------------------------------------------------------
            # Post-convergence verification
            # Binary search evaluations are intentionally fast (early stopping on,
            # low nsim is allowed). Re-evaluate the candidate with early stopping
            # disabled so all nsim simulations run. If power is still below target,
            # advance by step_size until it passes (max 10 steps).
            # This catches cases where a single noisy evaluation misled the search.
            # ------------------------------------------------------------------
            verified_power = _eval_power(best_total_size, use_early_stopping=False)
            n_evals += 1
            _MAX_RECOVERY = 10
            _recovery = 0
            while verified_power < comp_target_power - tolerance and _recovery < _MAX_RECOVERY:
                if best_total_size >= max_sample_size:
                    break
                best_total_size = min(max_sample_size, best_total_size + step_size)
                verified_power = _eval_power(best_total_size, use_early_stopping=False)
                n_evals += 1
                _recovery += 1

            if _recovery > 0:
                self.logger.debug(
                    f"Comparison {(group1, group2)}: verification recovered to n={best_total_size:,} "
                    f"in {_recovery} step(s) — consider increasing nsim (current: {self.nsim})"
                )
            best_power = verified_power

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
            use_early_stopping=False,
            show_progress=False,
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
            "correction": self.correction if self.correction is not None else "none",
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
        baseline: float = None,
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
        baseline : float, optional
            Baseline rate for proportions/counts or mean for averages.
            Same value is used for all groups (consistent with randomized experiment design).
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
        if baseline is None:
            baseline = 1.0

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

        target_comparisons = self._normalize_and_validate_comparisons(target_comparisons)

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

                    _desc = f"Simulations [{h} vs {j}]"

                    # Use configured parallel strategy (default: hybrid)
                    if self.parallel_strategy == "hybrid":
                        self.logger.debug(f"Running {nsim_val} retrodesign simulations with hybrid parallelization...")

                        # Stage 1: Generate simulation data (threading) — fast, no progress bar needed
                        sim_datasets = Parallel(n_jobs=-1, backend="threading", batch_size=50)(
                            delayed(self._generate_simulation_data)(
                                baseline=baseline,
                                effect=true_effect,
                                sample_size=sample_size,
                                compliance=compliance,
                                standard_deviation=standard_deviation,
                            )
                            for _ in range(nsim_val)
                        )

                        # Stage 2: Run tests with retrodesign-specific logic (multiprocessing)
                        simulation_results = list(
                            tqdm(
                                Parallel(n_jobs=-1, backend="loky", batch_size=50, return_as="generator")(
                                    delayed(self._run_retrodesign_test_on_data)(
                                        sim_data=sim_data,
                                        h=h,
                                        j=j,
                                    )
                                    for sim_data in sim_datasets
                                ),
                                total=nsim_val,
                                desc=_desc,
                                unit="sim",
                            )
                        )
                    elif self.parallel_strategy == "threading":
                        self.logger.debug(f"Running {nsim_val} retrodesign simulations in parallel (threading)...")
                        simulation_results = list(
                            tqdm(
                                Parallel(n_jobs=-1, backend="threading", return_as="generator")(
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
                                ),
                                total=nsim_val,
                                desc=_desc,
                                unit="sim",
                            )
                        )
                    else:  # "loky" or default
                        self.logger.debug(
                            f"Running {nsim_val} retrodesign simulations in parallel (multiprocessing)..."
                        )
                        simulation_results = list(
                            tqdm(
                                Parallel(n_jobs=-1, backend="loky", return_as="generator")(
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
                                ),
                                total=nsim_val,
                                desc=_desc,
                                unit="sim",
                            )
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

                    for _ in tqdm(range(nsim_val), desc=f"Simulations [{h} vs {j}]", unit="sim"):
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
