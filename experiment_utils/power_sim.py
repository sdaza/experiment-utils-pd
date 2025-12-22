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
        variants: int = None,
        comparisons: list[tuple[int, int]] = None,
        alternative: str = "two-tailed",
        alpha: float = 0.05,  # noqa: E501
        correction: str = "bonferroni",
        fdr_method: str = "indep",
    ) -> None:
        """
        PowerSim class for simulation of power analysis.

        Parameters
        ----------
        metric : str
            Count, proportion, or average
        relative effect : bool
            True when change is percentual (not absolute).
        variants : int
            Number of cohorts or variants to use (remember, total number of groups = control + number of variants)
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
        """

        self.logger = get_logger("Power Simulator")
        self.metric = metric
        self.relative_effect = relative_effect
        self.variants = variants
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

    def get_power(
        self,
        baseline: list[float] = None,
        effect: list[float] = None,
        sample_size: list[int] = None,
        compliance: list[float] = None,
        standard_deviation: list[float] = None,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Estimate power using simulation.

        Parameters
        ----------
        baseline : list
            List baseline rates for counts or proportions, or base average for mean comparisons.
        effect : list
            List with effect sizes.
        sample_size : list
            List with sample for control and arm groups.
        compliance : list
            List with compliance values.
        standard_deviation : list
            List of standard deviations of control and variants.

        Returns
        -------
        power : float
        """

        # Set default values for mutable arguments
        if baseline is None:
            baseline = [1.0]
        if effect is None:
            effect = [0.10]
        if sample_size is None:
            sample_size = [1000]
        if compliance is None:
            compliance = [1.0]
        if standard_deviation is None:
            standard_deviation = [1]

        # create empty values for results
        pvalues = {}
        for c in range(len(self.comparisons)):
            pvalues[c] = []

        # iterate over simulations
        for _i in range(self.nsim):
            # y = output, x = index of condition
            y, x = self.__run_experiment(
                baseline=baseline,
                effect=effect,
                sample_size=sample_size,
                compliance=compliance,
                standard_deviation=standard_deviation,
            )

            # iterate over variants
            l_pvalues = []
            for j, h in self.comparisons:
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
                significant = correction_methods[self.correction](
                    np.array(l_pvalues), self.alpha / pvalue_adjustment[self.alternative]
                )  # noqa: E501
            else:
                # No correction - compare each p-value to alpha directly
                significant = [p < self.alpha for p in l_pvalues]

            for v, p in enumerate(significant):
                pvalues[v].append(p)

        power = pd.DataFrame(pd.DataFrame(pvalues).mean()).reset_index()
        power.columns = ["comparisons", "power"]
        power["comparisons"] = power["comparisons"].map(dict(enumerate(self.comparisons)))

        return power

    def get_power_from_data(
        self,
        df: pd.DataFrame,
        metric_col: str,
        sample_size: list[int] = None,
        effect: list[float] = None,
        compliance: list[float] = None,
    ) -> pd.DataFrame:  # noqa: E501
        """
        Simulate statistical power using samples from the provided data.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing the metric data.
        metric_col : str
            Name of the column in the dataframe that contains the measurement used for testing.
        sample_size : list
            List of sample sizes for control and each variant group.
        effect : list
            List of effect sizes for each variant group.
        compliance : list
            List of compliance rates for each variant group.

        Returns
        -------
        pd.DataFrame
            DataFrame with each comparison (as defined in self.comparisons) and the corresponding estimated power.
        """
        # Set default values for mutable arguments
        if sample_size is None:
            sample_size = [100]
        if effect is None:
            effect = [0.10]
        if compliance is None:
            compliance = [1.0]

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
        test = [p > alpha / (m + 1 - k) for k, p in enumerate(pvals[ind])]

        """The minimal index k is m-np.sum(test) + 1 and the hypotheses 1, ..., k-1
        are rejected. Hence m-np.sum(test) gives the correct number."""
        significant = np.zeros(np.shape(pvals), dtype="bool")
        significant[ind[0 : m - np.sum(test)]] = True
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
            if (len(set(variant_effects)) == 1 or max(variant_effects) - min(variant_effects) < 0.001) and \
               (len(set(power_targets)) == 1):
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
        for (group1, group2) in sorted_comparisons:
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
                
                # Ensure all groups have at least some samples for proper correction calculation
                sample_sizes = [max(s, 100) for s in sample_sizes]
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
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
        baseline: list[float] = None,
        effect: list[float] = None,
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
        baseline : list
            List baseline rates for counts or proportions, or base average for mean comparisons.
        effect : list
            List with effect sizes for each variant.
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
            - "all": All target comparisons must reach their target power (conservative)
            - "any": At least one target comparison reaches its target power
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

        Returns
        -------
        pd.DataFrame
            DataFrame with comparison, target_power, sample_size, allocation, and achieved_power columns.

        Examples
        --------
        # Basic usage: same power target for all comparisons
        >>> p = PowerSim(metric='proportion', variants=1, nsim=500)
        >>> result = p.find_sample_size(target_power=0.80, baseline=[0.10], effect=[0.02])

        # Different power targets per comparison
        >>> p = PowerSim(metric='proportion', variants=2, nsim=500)
        >>> result = p.find_sample_size(
        ...     target_power={(0,1): 0.90, (0,2): 0.80},  # Primary comparison needs 90% power
        ...     baseline=[0.10],
        ...     effect=[0.05, 0.03]
        ... )

        # Optimize allocation automatically
        >>> p = PowerSim(metric='proportion', variants=3, nsim=500)
        >>> result = p.find_sample_size(
        ...     target_power=0.80,
        ...     baseline=[0.10],
        ...     effect=[0.05, 0.03, 0.07],  # Different effects
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

        # Custom allocation with power criteria
        >>> result = p.find_sample_size(
        ...     target_power=0.80,
        ...     baseline=[0.10],
        ...     effect=[0.02],
        ...     allocation_ratio=[0.3, 0.7],  # 30% control, 70% treatment
        ...     power_criteria="all"  # All target comparisons must be powered
        ... )
        """
        # Set default values
        if baseline is None:
            baseline = [1.0]
        if effect is None:
            effect = [0.10]
        if compliance is None:
            compliance = [1.0]
        if standard_deviation is None:
            standard_deviation = [1]

        # Default to equal allocation across all groups
        num_groups = self.variants + 1
        if allocation_ratio is None or optimize_allocation:
            allocation_ratio = [1.0 / num_groups] * num_groups

        # Validate allocation_ratio (unless optimizing)
        if not optimize_allocation:
            if len(allocation_ratio) != num_groups:
                log_and_raise_error(
                    self.logger, f"allocation_ratio must have {num_groups} elements (control + {self.variants} variants)"
                )

            if not np.isclose(sum(allocation_ratio), 1.0):
                log_and_raise_error(self.logger, "allocation_ratio must sum to 1.0")

            if any(r <= 0 for r in allocation_ratio):
                log_and_raise_error(self.logger, "All allocation_ratio values must be positive")

        # Handle target_comparisons
        if target_comparisons is None:
            target_comparisons = self.comparisons
        else:
            # Validate that target_comparisons are in self.comparisons
            invalid_comps = [c for c in target_comparisons if c not in self.comparisons]
            if invalid_comps:
                log_and_raise_error(
                    self.logger,
                    f"target_comparisons {invalid_comps} are not in defined comparisons {self.comparisons}",
                )

        # Validate power_criteria
        if power_criteria not in ["all", "any"]:
            log_and_raise_error(self.logger, "power_criteria must be 'all' or 'any'")

        # Handle target_power - can be single value or dict
        if isinstance(target_power, dict):
            # Validate dict keys are tuples in target_comparisons
            invalid_keys = [k for k in target_power.keys() if k not in target_comparisons]
            if invalid_keys:
                log_and_raise_error(
                    self.logger, f"target_power dict contains invalid comparisons: {invalid_keys}"
                )
            # Validate all values are valid probabilities
            if any(not (0 < p < 1) for p in target_power.values()):
                log_and_raise_error(self.logger, "All target_power values must be between 0 and 1")
            power_dict = target_power
        else:
            # Single value - validate and apply to all target comparisons
            if not 0 < target_power < 1:
                log_and_raise_error(self.logger, "target_power must be between 0 and 1")
            power_dict = {comp: target_power for comp in target_comparisons}

        # If optimize_allocation is True, find optimal allocation
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

        # For each target comparison, find the required sample size
        for (group1, group2) in target_comparisons:
            comp_target_power = power_dict[(group1, group2)]
            
            # Find the index of this comparison in self.comparisons for power_result indexing
            comp_result_idx = self.comparisons.index((group1, group2))

            # Binary search for optimal total sample size
            low = min_sample_size
            high = max_sample_size
            best_total_size = None
            best_power = None

            # First do a coarse search to find a good starting range
            current_total_size = min_sample_size
            while current_total_size <= max_sample_size:
                # Distribute total sample size by allocation ratio
                sample_sizes = [int(current_total_size * ratio) for ratio in allocation_ratio]

                power_result = self.get_power(
                    baseline=baseline,
                    effect=effect,
                    sample_size=sample_sizes,
                    compliance=compliance,
                    standard_deviation=standard_deviation,
                )
                current_power = power_result.iloc[comp_result_idx]["power"]

                if current_power >= comp_target_power - tolerance:
                    high = current_total_size
                    best_total_size = current_total_size
                    best_power = current_power
                    break

                low = current_total_size
                current_total_size += step_size

            # If we didn't find a solution in coarse search, max out
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
                )
                best_power = power_result.iloc[comp_result_idx]["power"]
            else:
                # Refine with binary search
                while high - low > step_size // 2:
                    mid = (low + high) // 2
                    sample_sizes = [int(mid * ratio) for ratio in allocation_ratio]

                    power_result = self.get_power(
                        baseline=baseline,
                        effect=effect,
                        sample_size=sample_sizes,
                        compliance=compliance,
                        standard_deviation=standard_deviation,
                    )
                    current_power = power_result.iloc[comp_result_idx]["power"]

                    if abs(current_power - comp_target_power) <= tolerance:
                        best_total_size = mid
                        best_power = current_power
                        break
                    elif current_power < comp_target_power:
                        low = mid
                    else:
                        high = mid
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

        # Determine final sample size based on power_criteria
        df_results = pd.DataFrame(results)
        
        if power_criteria == "all":
            # Find the maximum sample size needed (most conservative)
            max_sample_idx = df_results["total_sample_size"].idxmax()
            max_sample_size_needed = df_results.loc[max_sample_idx, "total_sample_size"]
            limiting_comparison = df_results.loc[max_sample_idx, "comparison"]
        else:  # power_criteria == "any"
            # Find the minimum sample size needed (least conservative)
            max_sample_idx = df_results["total_sample_size"].idxmin()
            max_sample_size_needed = df_results.loc[max_sample_idx, "total_sample_size"]
            limiting_comparison = df_results.loc[max_sample_idx, "comparison"]

        # Recalculate power for all comparisons using this final sample size
        final_sample_sizes = [int(max_sample_size_needed * ratio) for ratio in allocation_ratio]

        power_result = self.get_power(
            baseline=baseline,
            effect=effect,
            sample_size=final_sample_sizes,
            compliance=compliance,
            standard_deviation=standard_deviation,
        )

        # Create result with dictionaries for comparison-level details
        sample_sizes_dict = {"control": final_sample_sizes[0]}
        for i in range(1, len(final_sample_sizes)):
            sample_sizes_dict[f"variant_{i}"] = final_sample_sizes[i]
        
        # Log optimized sample sizes (only if optimization was used)
        if optimize_allocation:
            self.logger.info(f"Optimized sample sizes: {sample_sizes_dict}")

        # Build dictionaries for target power and achieved power by comparison
        target_power_by_comparison = {}
        achieved_power_by_comparison = {}
        
        for _idx, row in power_result.iterrows():
            comp = row["comparisons"]
            if comp in target_comparisons:
                target_power_by_comparison[str(comp)] = round(power_dict[comp], 3)
                achieved_power_by_comparison[str(comp)] = round(row["power"], 3)
        
        # Log achieved power
        self.logger.info(f"Achieved power: {achieved_power_by_comparison}")

        # Create single result row
        result = {
            "total_sample_size": max_sample_size_needed,
            "sample_sizes_by_group": sample_sizes_dict,
            "power_criteria": power_criteria,
            "correction": self.correction,
            "limiting_comparison": str(limiting_comparison),
            "target_power_by_comparison": target_power_by_comparison,
            "achieved_power_by_comparison": achieved_power_by_comparison,
        }

        return pd.DataFrame([result])
