"""Portfolio shrinkage, priors, and cumulative-impact helpers.

Winner's-curse corrections, empirical-Bayes / Student-t priors, joint
primary|guardrail (NSS) shrinkage, and Kessler-style cumulative aggregation.
Analyzer wrappers live on :class:`~experiment_utils.ExperimentAnalyzer`.
"""

from __future__ import annotations

import warnings
from functools import cache
from math import sqrt

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.stats import chi2, norm
from scipy.stats import t as t_dist


def winners_curse_estimate(
    effect: float,
    standard_error: float,
    alpha: float = 0.05,
    ci: float = 0.95,
    alternative: str = "two-sided",
) -> dict:
    """
    Winner's-curse correction for a single estimate selected by significance.

    Models ``effect ~ N(beta, standard_error**2)`` truncated to the selection
    region implied by ``alternative``:

    - ``"two-sided"`` (default): ``|effect| >= z* * SE`` with
      ``z* = Phi^{-1}(1 - alpha/2)``.
    - ``"greater"``: ``effect >= z* * SE`` with ``z* = Phi^{-1}(1 - alpha)``
      (one-sided launch / positive win; Kessler-style).
    - ``"less"``: ``effect <= -z* * SE`` with the same one-sided ``z*``.

    Returns the median-unbiased estimate of ``beta`` (the value whose
    conditional CDF at ``effect`` equals 0.5) and a selection-adjusted
    equal-tailed confidence interval. Operates on whatever scale
    ``effect``/``standard_error`` are supplied on (for GLMs that is the
    log/coefficient scale).

    The function assumes the effect lies in the selection region. If this
    precondition is violated a ``RuntimeWarning`` is emitted and the
    computation proceeds; it does not raise so downstream pipelines that pass
    already-screened rows are not disrupted when the screening threshold
    differs slightly from ``alpha``.

    Parameters
    ----------
    effect : float
        The observed (significant) point estimate.
    standard_error : float
        Its standard error; must be > 0.
    alpha : float
        Significance level that defined selection (default 0.05). Interpreted
        as two-sided for ``alternative="two-sided"`` and one-sided otherwise.
    ci : float
        Confidence level for the adjusted interval (default 0.95).
    alternative : {"two-sided", "greater", "less"}
        Selection region (default ``"two-sided"``).

    Returns
    -------
    dict
        ``corrected`` (median-unbiased estimate), ``ci_lower``, ``ci_upper``
        (selection-adjusted interval), ``observed_z`` (= effect/standard_error),
        ``shrinkage`` (= corrected/effect), ``alternative``.
    """
    if not np.isfinite(effect):
        raise ValueError("effect must be finite")
    if not (np.isfinite(standard_error) and standard_error > 0):
        raise ValueError("standard_error must be positive and finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    s = float(standard_error)
    b = float(effect)
    if alternative == "two-sided":
        c = norm.ppf(1.0 - alpha / 2.0) * s
        selected = abs(b) >= c
    else:
        c = norm.ppf(1.0 - alpha) * s
        selected = b >= c if alternative == "greater" else b <= -c

    if not selected:
        warnings.warn(
            "winners_curse_estimate: effect is outside the selection region; "
            "the correction assumes the estimate was selected by significance.",
            RuntimeWarning,
            stacklevel=2,
        )

    observed_z = b / s

    def cond_cdf(beta: float) -> float:
        if alternative == "two-sided":
            # S = (-inf, -c] U [c, inf)
            p_sel = norm.cdf((-c - beta) / s) + norm.sf((c - beta) / s)
            if b >= c:
                num = norm.cdf((-c - beta) / s) + norm.cdf((b - beta) / s) - norm.cdf((c - beta) / s)
            elif b <= -c:
                num = norm.cdf((b - beta) / s)
            else:  # excluded gap (-c, c): only the lower selected tail (-inf, -c] is <= b
                num = norm.cdf((-c - beta) / s)
            return num / p_sel
        if alternative == "greater":
            # S = [c, inf). Use survival ratio to avoid 0/0 when beta << c.
            if b < c:
                return 0.0
            z_c = (c - beta) / s
            z_b = (b - beta) / s
            den = norm.sf(z_c)
            if den == 0.0:
                return 1.0
            return float(1.0 - np.clip(norm.sf(z_b) / den, 0.0, 1.0))
        # alternative == "less": S = (-inf, -c]. Use CDF ratio for stability.
        if b > -c:
            return 1.0
        z_uc = (-c - beta) / s  # upper endpoint of selection region
        z_b = (b - beta) / s
        den = norm.cdf(z_uc)
        if den == 0.0:
            return 0.0
        return float(np.clip(norm.cdf(z_b) / den, 0.0, 1.0))

    def invert(target: float) -> float:
        # cond_cdf is monotone DECREASING in beta (1 at -inf, 0 at +inf).
        f = lambda beta: cond_cdf(beta) - target  # noqa: E731
        # One-sided near-threshold observations need a wide left bracket:
        # the median-unbiased beta can be many SEs below the truncation point.
        lo, hi = b - 2.0 * s, b + 2.0 * s
        steps = 0
        while f(lo) < 0 and steps < 200:
            lo -= 4.0 * s
            steps += 1
        steps = 0
        while f(hi) > 0 and steps < 200:
            hi += 4.0 * s
            steps += 1
        try:
            return float(brentq(f, lo, hi, xtol=1e-8, rtol=1e-12, maxiter=200))
        except ValueError:
            warnings.warn(
                "winners_curse_estimate: root-finding failed to bracket; returning NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")

    gamma = 1.0 - ci
    corrected = invert(0.5)
    ci_lower = invert(1.0 - gamma / 2.0)  # decreasing CDF: high target -> small beta
    ci_upper = invert(gamma / 2.0)
    shrinkage = corrected / b if b != 0 else float("nan")

    return {
        "corrected": corrected,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "observed_z": observed_z,
        "shrinkage": shrinkage,
        "alternative": alternative,
    }


def _paule_mandel_tau2(y: np.ndarray, v: np.ndarray, prior_mean: float = 0.0) -> float:
    """
    Method-of-moments (Paule-Mandel style) between-estimate variance with the
    prior mean FIXED at ``prior_mean``. Solves, for tau2 >= 0,
    ``sum (y_i - prior_mean)^2 / (v_i + tau2) = k``. Monotone decreasing, so
    a single root via brentq; returns 0 when the estimates are no more
    dispersed than their standard errors imply.
    """
    k = y.size
    resid2 = (y - prior_mean) ** 2

    def g(tau2: float) -> float:
        return float(np.sum(resid2 / (v + tau2)) - k)

    if g(0.0) <= 0:
        return 0.0
    hi = 100.0 * float(np.max(v))
    steps = 0
    while g(hi) > 0 and steps < 80:
        hi *= 2.0
        steps += 1
    return float(brentq(g, 0.0, hi, xtol=1e-12, rtol=1e-12, maxiter=200))


def empirical_bayes_shrinkage(
    effects,
    standard_errors,
    prior_mean: float = 0.0,
    ci: float = 0.95,
    tau2: float | None = None,
) -> dict:
    """
    Empirical-Bayes (normal-prior) shrinkage of a family of estimates.

    Assumes ``beta_i ~ N(prior_mean, tau2)`` with ``effect_i | beta_i ~
    N(beta_i, se_i**2)``, estimates ``tau2`` by method of moments
    (:func:`_paule_mandel_tau2`), and returns posterior means and credible
    intervals. High-variance "winners" shrink most. All inputs must be on the
    same scale (e.g. all log-odds, or all mean differences).

    Alternatively, pass a fixed ``tau2`` learned from historical experiments
    (e.g. ``empirical_bayes_shrinkage(past_effects, past_ses)["tau2"]``) to
    shrink any number of new estimates — including a single one — with that
    external prior instead of re-learning it (van Zwet, Schwab & Senn 2021;
    Azevedo et al. 2020).

    Parameters
    ----------
    effects, standard_errors : array-like
        Estimates and their standard errors, on a common scale.
    prior_mean : float
        Mean of the normal prior (default 0.0).
    ci : float
        Credible-interval level (default 0.95).
    tau2 : float, optional
        Fixed prior variance. When given, it is used as-is (no estimation)
        and a single estimate is allowed; when None (default), ``tau2`` is
        learned from the supplied estimates, which requires at least 3.

    Returns
    -------
    dict
        ``shrunk`` (posterior means), ``shrinkage_factor`` (= tau2/(tau2+se^2)),
        ``posterior_sd``, ``ci_lower``, ``ci_upper`` (np.ndarray aligned with
        inputs), plus scalar ``tau2`` and ``prior_mean``.
    """
    y = np.asarray(effects, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("effects and standard_errors must be 1-D arrays of equal length")
    if tau2 is None:
        if y.size < 3:
            raise ValueError(
                "empirical Bayes requires at least 3 estimates to learn a prior; "
                "pass a fixed tau2 (e.g. learned from historical experiments) to shrink fewer"
            )
    else:
        if not (np.isfinite(tau2) and tau2 >= 0):
            raise ValueError("tau2 must be finite and >= 0")
        if y.size < 1:
            raise ValueError("effects must contain at least one estimate")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))) or np.any(s <= 0):
        raise ValueError("all standard_errors must be positive and finite, and effects finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    v = s**2
    if tau2 is None:
        tau2 = _paule_mandel_tau2(y, v, prior_mean=prior_mean)
    shrinkage_factor = tau2 / (tau2 + v)
    shrunk = prior_mean + shrinkage_factor * (y - prior_mean)
    posterior_sd = np.sqrt(tau2 * v / (tau2 + v))
    z = norm.ppf(1.0 - (1.0 - ci) / 2.0)
    return {
        "shrunk": shrunk,
        "shrinkage_factor": shrinkage_factor,
        "posterior_sd": posterior_sd,
        "ci_lower": shrunk - z * posterior_sd,
        "ci_upper": shrunk + z * posterior_sd,
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
    }


def _validate_effects_ses(effects, standard_errors) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(effects, dtype=float)
    s = np.asarray(standard_errors, dtype=float)
    if y.ndim != 1 or y.shape != s.shape:
        raise ValueError("effects and standard_errors must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(s))) or np.any(s <= 0):
        raise ValueError("all standard_errors must be positive and finite, and effects finite")
    return y, s


def t_prior_shrinkage(
    effects,
    standard_errors,
    scale: float,
    df: float,
    prior_mean: float = 0.0,
    ci: float = 0.95,
) -> dict:
    """
    Bayesian shrinkage under a fat-tailed Student-t prior.

    Assumes ``beta_i ~ t_df(prior_mean, scale)`` with ``effect_i | beta_i ~
    N(beta_i, se_i**2)`` and returns the posterior mean, sd, and equal-tailed
    credible interval per estimate (computed by numerical integration; the
    t-prior posterior has no closed form). Unlike normal-prior shrinkage the
    posterior mean is nonlinear in the estimate: moderate effects shrink
    hard while very large ones pass through mostly untouched, which suits
    experiment archives with occasional genuine big winners (Azevedo et al.
    2020, "A/B Testing with Fat Tails").

    Prior parameters are external by design — learn them once from historical
    experiments with :func:`fit_t_prior` — so a single new estimate is enough.

    Parameters
    ----------
    effects, standard_errors : array-like
        Estimates and their standard errors, on a common scale.
    scale : float
        Scale of the t prior; must be > 0.
    df : float
        Degrees of freedom of the t prior; must be > 0 (typically 3-10; the
        prior variance is only finite for df > 2).
    prior_mean : float
        Location of the prior (default 0.0).
    ci : float
        Credible-interval level (default 0.95).

    Returns
    -------
    dict
        ``shrunk`` (posterior means), ``shrinkage_factor``
        (= (shrunk - prior_mean)/(effect - prior_mean), NaN at the prior mean),
        ``posterior_sd``, ``ci_lower``, ``ci_upper`` (np.ndarray aligned with
        inputs), plus scalars ``scale``, ``df``, ``prior_mean`` and ``tau2``
        (implied prior variance, ``inf`` when df <= 2).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError("scale must be positive and finite")
    if not (np.isfinite(df) and df > 0):
        raise ValueError("df must be positive and finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    gamma = 1.0 - ci
    shrunk = np.empty_like(y)
    posterior_sd = np.empty_like(y)
    ci_lower = np.empty_like(y)
    ci_upper = np.empty_like(y)
    for i, (b, se) in enumerate(zip(y, s, strict=True)):
        lo = min(b - 8.0 * se, prior_mean - 8.0 * scale)
        hi = max(b + 8.0 * se, prior_mean + 8.0 * scale)
        grid = np.linspace(lo, hi, 4001)
        log_post = norm.logpdf(b, loc=grid, scale=se) + t_dist.logpdf(grid, df, loc=prior_mean, scale=scale)
        dens = np.exp(log_post - log_post.max())
        dens /= np.trapezoid(dens, grid)
        mean = np.trapezoid(grid * dens, grid)
        var = np.trapezoid((grid - mean) ** 2 * dens, grid)
        cdf = cumulative_trapezoid(dens, grid, initial=0.0)
        cdf /= cdf[-1]
        shrunk[i] = mean
        posterior_sd[i] = np.sqrt(max(var, 0.0))
        ci_lower[i] = np.interp(gamma / 2.0, cdf, grid)
        ci_upper[i] = np.interp(1.0 - gamma / 2.0, cdf, grid)

    centered = y - prior_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        shrinkage_factor = np.where(centered != 0, (shrunk - prior_mean) / centered, np.nan)
    tau2 = scale**2 * df / (df - 2.0) if df > 2 else float("inf")
    return {
        "shrunk": shrunk,
        "shrinkage_factor": shrinkage_factor,
        "posterior_sd": posterior_sd,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "scale": float(scale),
        "df": float(df),
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
    }


def fit_t_prior(
    effects,
    standard_errors,
    prior_mean: float = 0.0,
    df: float | None = None,
) -> dict:
    """
    Fit a Student-t prior to a historical archive of experiment estimates.

    Maximizes the marginal likelihood of ``effect_i ~ N(beta_i, se_i**2)``
    with ``beta_i ~ t_df(prior_mean, scale)`` (integrated via Gauss-Hermite
    quadrature). Learn the prior once from past experiments, then shrink each
    new result with :func:`t_prior_shrinkage` (or pass the returned dict as
    ``prior=`` to ``ExperimentAnalyzer.winners_curse_summary``).

    Parameters
    ----------
    effects, standard_errors : array-like
        Historical estimates and their standard errors, on a common scale.
        At least 3 are required; reliably estimating ``df`` needs a large
        archive (dozens of experiments) — with a small one, fix ``df``
        (e.g. ``df=4``) so only the scale is learned.
    prior_mean : float
        Location of the prior (default 0.0).
    df : float, optional
        Fix the degrees of freedom and fit only the scale. When None
        (default) both are fitted, with ``df`` constrained to > 2 so the
        prior variance is finite.

    Returns
    -------
    dict
        ``scale``, ``df``, ``tau2`` (implied prior variance), ``prior_mean``,
        ``loglik``, ``n``.
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 3:
        raise ValueError("fit_t_prior requires at least 3 estimates")
    if df is not None and not (np.isfinite(df) and df > 0):
        raise ValueError("df must be positive and finite")

    gh_x, gh_w = np.polynomial.hermite.hermgauss(64)
    nodes = y[:, None] + np.sqrt(2.0) * s[:, None] * gh_x[None, :]
    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)

    def nll(log_scale: float, dfv: float) -> float:
        marginal = inv_sqrt_pi * (t_dist.pdf(nodes, dfv, loc=prior_mean, scale=np.exp(log_scale)) * gh_w).sum(axis=1)
        return float(-np.sum(np.log(np.maximum(marginal, 1e-300))))

    tau2_0 = _paule_mandel_tau2(y, s**2, prior_mean=prior_mean)
    scale0 = np.sqrt(tau2_0) if tau2_0 > 0 else 0.5 * float(np.median(s))
    if df is not None:
        res = minimize(lambda u: nll(u[0], df), x0=[np.log(scale0)], method="Nelder-Mead")
        scale_hat, df_hat = float(np.exp(res.x[0])), float(df)
    else:
        # df = 2 + exp(u) keeps the fitted prior variance finite
        res = minimize(lambda u: nll(u[0], 2.0 + np.exp(u[1])), x0=[np.log(scale0), np.log(3.0)], method="Nelder-Mead")
        scale_hat, df_hat = float(np.exp(res.x[0])), float(2.0 + np.exp(res.x[1]))

    tau2 = scale_hat**2 * df_hat / (df_hat - 2.0) if df_hat > 2 else float("inf")
    return {
        "scale": scale_hat,
        "df": df_hat,
        "tau2": float(tau2),
        "prior_mean": float(prior_mean),
        "loglik": float(-res.fun),
        "n": int(y.size),
    }


def fit_t_prior_with_estimated_mean(
    effects,
    standard_errors,
    *,
    df: float = 4.0,
    mean_ci: float = 0.95,
) -> dict:
    """
    Fit a Student-t prior's location and scale by profile likelihood.

    :func:`fit_t_prior` treats ``prior_mean`` as fixed. This helper profiles
    over that argument, then obtains a likelihood-ratio interval for the
    fitted location. With ``df > 1`` the Student-t location is also its mean,
    so the interval answers whether the archive's average underlying effect
    is distinguishable from zero.

    Learn the prior once from historical experiments, then shrink new results
    with :func:`t_prior_shrinkage` (pass ``prior_mean=fit["prior_mean"]``) or
    pass the returned dict as ``prior=`` to
    ``ExperimentAnalyzer.winners_curse_summary``.

    Parameters
    ----------
    effects, standard_errors : array-like
        Historical estimates and their standard errors, on a common scale.
        At least 3 are required.
    df : float
        Fixed degrees of freedom of the t prior; must be > 1 so the prior
        mean exists (default 4.0).
    mean_ci : float
        Nominal level of the profile likelihood-ratio interval for the prior
        mean (default 0.95).

    Returns
    -------
    dict
        Everything returned by :func:`fit_t_prior` at the fitted mean, plus
        ``prior_mean_ci_lower``, ``prior_mean_ci_upper``,
        ``prior_mean_ci_level``, and ``prior_mean_method``
        (``"profile_likelihood"``).
    """
    y, se = _validate_effects_ses(effects, standard_errors)
    if y.size < 3:
        raise ValueError("at least 3 estimates are required")
    if not 0 < mean_ci < 1:
        raise ValueError("mean_ci must be strictly between 0 and 1")
    if not np.isfinite(df) or df <= 1:
        raise ValueError("df must be finite and greater than 1 so the prior mean exists")

    @cache
    def conditional_fit(prior_mean: float) -> dict:
        return fit_t_prior(y, se, prior_mean=float(prior_mean), df=df)

    def objective(prior_mean: float) -> float:
        return -conditional_fit(float(prior_mean))["loglik"]

    # Robust bounds: min/max effects are unstable when the archive has
    # enormous ratios from near-zero baselines.
    center = float(np.median(y))
    q05, q95 = np.quantile(y, [0.05, 0.95])
    span = max(float(q95 - q05), 10.0 * float(np.median(se)), 0.01)
    for _ in range(4):
        lower, upper = center - span, center + span
        result = minimize_scalar(
            objective,
            bounds=(lower, upper),
            method="bounded",
            options={"xatol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(f"prior-mean fit failed: {result.message}")
        edge_tolerance = max(span * 1e-3, 1e-8)
        if lower + edge_tolerance < result.x < upper - edge_tolerance:
            break
        span *= 4.0
    else:
        raise RuntimeError("prior-mean fit remained on the search boundary")

    prior_mean = float(result.x)
    prior = conditional_fit(prior_mean).copy()
    max_loglik = float(prior["loglik"])
    target_loglik = max_loglik - 0.5 * float(chi2.ppf(mean_ci, df=1))

    def profile_distance(candidate: float) -> float:
        return float(conditional_fit(float(candidate))["loglik"] - target_loglik)

    initial_step = max(
        float(prior["scale"]) / sqrt(y.size),
        float(np.median(se)) / sqrt(y.size),
        1e-6,
    )

    def find_endpoint(direction: float) -> float:
        inner = prior_mean
        step = initial_step
        for _ in range(60):
            outer = prior_mean + direction * step
            if profile_distance(outer) <= 0:
                lo, hi = sorted((inner, outer))
                return float(brentq(profile_distance, lo, hi, xtol=1e-8))
            inner = outer
            step *= 1.8
        raise RuntimeError("could not bracket the prior-mean likelihood interval")

    prior.update(
        {
            "prior_mean": prior_mean,
            "prior_mean_ci_lower": find_endpoint(-1.0),
            "prior_mean_ci_upper": find_endpoint(1.0),
            "prior_mean_ci_level": float(mean_ci),
            "prior_mean_method": "profile_likelihood",
        }
    )
    return prior


def joint_metric_shrinkage(
    primary_effects,
    primary_ses,
    guardrail_effects,
    guardrail_ses,
    *,
    rho: float,
    prior_sd_primary: float,
    prior_sd_guard: float | None = None,
    prior_mean_primary: float = 0.0,
    prior_mean_guard: float = 0.0,
    ci: float = 0.95,
) -> dict:
    """
    Bivariate normal–normal shrinkage of primary and guardrail effects.

    Models true effects ``(delta, gamma)`` as jointly normal with correlation
    ``rho`` and known prior SDs, observed with independent sampling errors.
    Returns posterior means and equal-tailed CIs. ``rho`` is caller-supplied
    (not estimated). When ``rho=0`` the primary posterior matches univariate
    normal EB with the same prior SD.

    Parameters
    ----------
    primary_effects, primary_ses, guardrail_effects, guardrail_ses : array-like
        Aligned 1-D arrays of equal length.
    rho : float
        Corr(true primary, true guardrail); must be in (-1, 1).
    prior_sd_primary : float
        Prior SD of the primary true effect; must be > 0.
    prior_sd_guard : float, optional
        Prior SD of the guardrail; defaults to ``prior_sd_primary``.
    prior_mean_primary, prior_mean_guard : float
        Prior means (default 0).
    ci : float
        Credible-interval level (default 0.95).

    Returns
    -------
    dict
        ``primary_shrunk``, ``guard_shrunk``, ``primary_posterior_sd``,
        ``guard_posterior_sd``, ``primary_ci_lower``, ``primary_ci_upper``,
        ``guard_ci_lower``, ``guard_ci_upper``, plus scalar prior params.
    """
    x, sx = _validate_effects_ses(primary_effects, primary_ses)
    g, sg = _validate_effects_ses(guardrail_effects, guardrail_ses)
    if x.shape != g.shape:
        raise ValueError("primary and guardrail arrays must have the same length")
    if x.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not (np.isfinite(rho) and abs(rho) < 1.0):
        raise ValueError("rho must be finite and strictly inside (-1, 1)")
    if not (np.isfinite(prior_sd_primary) and prior_sd_primary > 0):
        raise ValueError("prior_sd_primary must be positive and finite")
    if prior_sd_guard is None:
        prior_sd_guard = prior_sd_primary
    if not (np.isfinite(prior_sd_guard) and prior_sd_guard > 0):
        raise ValueError("prior_sd_guard must be positive and finite")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    tau_p2 = float(prior_sd_primary) ** 2
    tau_g2 = float(prior_sd_guard) ** 2
    cov = float(rho) * prior_sd_primary * prior_sd_guard
    sigma = np.array([[tau_p2, cov], [cov, tau_g2]], dtype=float)
    mu = np.array([prior_mean_primary, prior_mean_guard], dtype=float)
    z = norm.ppf(1.0 - (1.0 - ci) / 2.0)

    primary_shrunk = np.empty_like(x)
    guard_shrunk = np.empty_like(x)
    primary_sd = np.empty_like(x)
    guard_sd = np.empty_like(x)
    for i in range(x.size):
        v = np.diag([sx[i] ** 2, sg[i] ** 2])
        post_cov = np.linalg.inv(np.linalg.inv(sigma) + np.linalg.inv(v))
        a = sigma @ np.linalg.inv(sigma + v)
        post_mean = mu + a @ (np.array([x[i], g[i]]) - mu)
        primary_shrunk[i] = post_mean[0]
        guard_shrunk[i] = post_mean[1]
        primary_sd[i] = np.sqrt(max(post_cov[0, 0], 0.0))
        guard_sd[i] = np.sqrt(max(post_cov[1, 1], 0.0))

    return {
        "primary_shrunk": primary_shrunk,
        "guard_shrunk": guard_shrunk,
        "primary_posterior_sd": primary_sd,
        "guard_posterior_sd": guard_sd,
        "primary_ci_lower": primary_shrunk - z * primary_sd,
        "primary_ci_upper": primary_shrunk + z * primary_sd,
        "guard_ci_lower": guard_shrunk - z * guard_sd,
        "guard_ci_upper": guard_shrunk + z * guard_sd,
        "rho": float(rho),
        "prior_sd_primary": float(prior_sd_primary),
        "prior_sd_guard": float(prior_sd_guard),
        "prior_mean_primary": float(prior_mean_primary),
        "prior_mean_guard": float(prior_mean_guard),
    }


def cumulative_impact(
    effects,
    standard_errors,
    *,
    shipped=None,
    prior=None,
    tau2: float | None = None,
    prior_mean: float = 0.0,
    aggregation: str = "sum",
    coverage=None,
    ci: float = 0.95,
    min_shipped: int = 1,
) -> dict:
    """
    Noise-adjusted cumulative impact of shipped experiments (Kessler / Datadog).

    Shrinks every estimate with a normal or Student-t archive prior, then
    aggregates **only** the ``shipped`` subset. Pass the real launch rule in
    ``shipped`` (including guardrails); do not confuse significance on the
    primary with shipping.

    Aggregation:

    - ``"sum"``: ``sum(w_i * theta_hat_i)`` over shipped (absolute / additive).
    - ``"product"``: ``prod(1 + w_i * theta_hat_i) - 1`` (relative lifts).

    ``w_i`` is coverage (share of eligible users) when ``coverage`` is given,
    else 1. The CI uses the plug-in sum of posterior variances (fixed-prior
    Kessler formula); when the prior is learned from the same data the
    interval is slightly anti-conservative.

    Parameters
    ----------
    effects, standard_errors : array-like
        Experiment lifts and SEs on a common scale.
    shipped : array-like of bool, optional
        Launch mask; default ships all experiments.
    prior : dict or ``"map"``, optional
        External prior: ``{"tau2": ...}`` (normal), ``{"scale", "df"}``
        (Student-t), or ``"map"`` to fit a Datadog-style Half-Cauchy MAP
        normal prior via :func:`fit_normal_prior_map`. Optional ``prior_mean``
        in dict priors.
    tau2 : float, optional
        Fixed normal prior variance when ``prior`` is omitted.
    prior_mean : float
        Shrinkage location when learning / fixing a normal prior (default 0).
        Ignored when ``prior`` already supplies ``prior_mean``.
    aggregation : {"sum", "product"}
        How to combine shipped posterior means.
    coverage : array-like, optional
        Per-experiment exposure weights in ``[0, 1]`` for a global view.
    ci : float
        Interval level (default 0.95).
    min_shipped : int
        Require at least this many shipped experiments (default 1).

    Returns
    -------
    dict
        ``cumulative``, ``ci_lower``, ``ci_upper``, ``n_total``, ``n_shipped``,
        ``shrunk``, ``posterior_sd``, ``shipped_mask``, ``aggregation``,
        prior metadata (``prior_family``, ``tau2`` / ``scale`` / ``df``,
        ``prior_mean``).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if aggregation not in {"sum", "product"}:
        raise ValueError("aggregation must be 'sum' or 'product'")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")
    if not isinstance(min_shipped, int) or min_shipped < 1:
        raise ValueError("min_shipped must be an integer >= 1")

    if shipped is None:
        shipped_mask = np.ones(y.size, dtype=bool)
    else:
        shipped_mask = np.asarray(shipped, dtype=bool)
        if shipped_mask.shape != y.shape:
            raise ValueError("shipped must be a 1-D boolean array aligned with effects")

    n_shipped = int(shipped_mask.sum())
    if n_shipped < min_shipped:
        raise ValueError(f"need at least {min_shipped} shipped experiments; got {n_shipped}")

    if coverage is None:
        weights = np.ones(y.size, dtype=float)
    else:
        weights = np.asarray(coverage, dtype=float)
        if weights.shape != y.shape:
            raise ValueError("coverage must be a 1-D array aligned with effects")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("coverage values must be finite and in [0, 1]")

    # Resolve prior / shrink all rows (fit on all; aggregate shipped only).
    prior_meta: dict = {"prior_mean": float(prior_mean)}
    if prior == "map":
        prior = fit_normal_prior_map(y, s)
    if prior is not None:
        if not isinstance(prior, dict):
            raise ValueError("prior must be a dict or the string 'map'")
        loc = float(prior.get("prior_mean", prior_mean))
        if "scale" in prior and "df" in prior:
            eb = t_prior_shrinkage(y, s, scale=float(prior["scale"]), df=float(prior["df"]), prior_mean=loc, ci=ci)
            prior_meta.update(
                {
                    "prior_family": "student_t",
                    "scale": float(eb["scale"]),
                    "df": float(eb["df"]),
                    "tau2": float(eb["tau2"]),
                    "prior_mean": loc,
                }
            )
        elif "tau2" in prior:
            eb = empirical_bayes_shrinkage(y, s, prior_mean=loc, ci=ci, tau2=float(prior["tau2"]))
            prior_meta.update(
                {
                    "prior_family": "normal",
                    "tau2": float(eb["tau2"]),
                    "prior_mean": loc,
                    "prior_fit": prior.get("method"),
                }
            )
        else:
            raise ValueError("prior must contain 'tau2' or both 'scale' and 'df'")
    elif tau2 is not None:
        eb = empirical_bayes_shrinkage(y, s, prior_mean=prior_mean, ci=ci, tau2=tau2)
        prior_meta.update({"prior_family": "normal", "tau2": float(eb["tau2"])})
    else:
        eb = empirical_bayes_shrinkage(y, s, prior_mean=prior_mean, ci=ci)
        prior_meta.update({"prior_family": "normal", "tau2": float(eb["tau2"])})

    shrunk = np.asarray(eb["shrunk"], dtype=float)
    post_sd = np.asarray(eb["posterior_sd"], dtype=float)
    agg = aggregate_shrunk_cumulative(
        shrunk,
        post_sd,
        shipped=shipped_mask,
        coverage=weights,
        aggregation=aggregation,
        ci=ci,
        min_shipped=min_shipped,
    )
    return {
        "cumulative": agg["cumulative"],
        "ci_lower": agg["ci_lower"],
        "ci_upper": agg["ci_upper"],
        "n_total": int(y.size),
        "n_shipped": agg["n_shipped"],
        "shrunk": shrunk,
        "posterior_sd": post_sd,
        "shipped_mask": shipped_mask,
        "aggregation": aggregation,
        **prior_meta,
    }


def aggregate_shrunk_cumulative(
    shrunk,
    posterior_sd,
    *,
    shipped=None,
    coverage=None,
    observed=None,
    aggregation: str = "sum",
    ci: float = 0.95,
    min_shipped: int = 1,
) -> dict:
    """
    Kessler-style aggregate of *already* shrunk effects (same CI as ``cumulative_impact``).

    Use after :func:`joint_metric_shrinkage` (NSS / guardrail-adjusted primary
    posterior) or any other row-wise shrinker. Unlike :func:`cumulative_impact`,
    this does **not** re-shrink with a univariate prior.
    """
    theta = np.asarray(shrunk, dtype=float)
    sd = np.asarray(posterior_sd, dtype=float)
    if theta.ndim != 1 or theta.shape != sd.shape:
        raise ValueError("shrunk and posterior_sd must be aligned 1-D arrays")
    if aggregation not in {"sum", "product"}:
        raise ValueError("aggregation must be 'sum' or 'product'")
    if not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")
    if not isinstance(min_shipped, int) or min_shipped < 1:
        raise ValueError("min_shipped must be an integer >= 1")

    if shipped is None:
        shipped_mask = np.ones(theta.size, dtype=bool)
    else:
        shipped_mask = np.asarray(shipped, dtype=bool)
        if shipped_mask.shape != theta.shape:
            raise ValueError("shipped must be a 1-D boolean array aligned with shrunk")

    n_shipped = int(shipped_mask.sum())
    if n_shipped < min_shipped:
        raise ValueError(f"need at least {min_shipped} shipped experiments; got {n_shipped}")

    if coverage is None:
        weights = np.ones(theta.size, dtype=float)
    else:
        weights = np.asarray(coverage, dtype=float)
        if weights.shape != theta.shape:
            raise ValueError("coverage must be a 1-D array aligned with shrunk")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("coverage values must be finite and in [0, 1]")

    w_l = weights[shipped_mask]
    theta_l = theta[shipped_mask]
    sd_l = sd[shipped_mask]
    contrib = w_l * theta_l
    z = float(norm.ppf(1.0 - (1.0 - ci) / 2.0))

    if aggregation == "sum":
        cumulative = float(np.sum(contrib))
        se_cum = float(np.sqrt(np.sum((w_l * sd_l) ** 2)))
        ci_lower = cumulative - z * se_cum
        ci_upper = cumulative + z * se_cum
    else:
        factors = 1.0 + contrib
        if np.any(~np.isfinite(factors)) or np.any(factors <= 0):
            raise ValueError("product aggregation requires 1 + coverage * shrunk > 0 for every shipped experiment")
        cumulative = float(np.prod(factors) - 1.0)
        var_log = float(np.sum((w_l * sd_l / factors) ** 2))
        se_log = float(np.sqrt(var_log))
        log_point = float(np.sum(np.log(factors)))
        ci_lower = float(np.exp(log_point - z * se_log) - 1.0)
        ci_upper = float(np.exp(log_point + z * se_log) - 1.0)

    naive = None
    if observed is not None:
        obs = np.asarray(observed, dtype=float)
        if obs.shape != theta.shape:
            raise ValueError("observed must be a 1-D array aligned with shrunk")
        naive = float(np.sum(w_l * obs[shipped_mask]))

    return {
        "cumulative": cumulative,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_total": int(theta.size),
        "n_shipped": n_shipped,
        "shipped_mask": shipped_mask,
        "aggregation": aggregation,
        "naive_sum": naive,
        "retain_vs_naive": (cumulative / naive) if (naive is not None and naive != 0) else None,
    }


def nss_adjusted_cumulative_impact(
    primary_effects,
    primary_ses,
    guardrail_effects,
    guardrail_ses,
    *,
    shipped=None,
    coverage=None,
    rho: float | None = None,
    prior_sd_primary: float | None = None,
    prior_sd_guard: float | None = None,
    prior_mean_primary: float = 0.0,
    prior_mean_guard: float = 0.0,
    aggregation: str = "sum",
    ci: float = 0.95,
    min_shipped: int = 1,
) -> dict:
    """
    Joint primary|guardrail (NSS companion) shrink, then Kessler aggregate.

    Composes :func:`joint_metric_shrinkage` on the paired archive with
    :func:`aggregate_shrunk_cumulative` on the **primary** posterior. When
    ``rho`` / prior SDs are omitted they are estimated via
    :func:`estimate_guardrail_rho`. Use the same ``shipped`` mask as univariate
    ``cumulative_impact`` for an apples-to-apples comparison (e.g. SCALED-only
    paired rows).
    """
    y, sy = _validate_effects_ses(primary_effects, primary_ses)
    g, sg = _validate_effects_ses(guardrail_effects, guardrail_ses)
    if y.shape != g.shape:
        raise ValueError("primary and guardrail arrays must have the same length")
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")

    if rho is None or prior_sd_primary is None or prior_sd_guard is None:
        if y.size < 5:
            raise ValueError(
                "estimating rho / prior SDs requires at least 5 paired experiments; "
                "pass rho, prior_sd_primary, and prior_sd_guard explicitly"
            )
        mom = estimate_guardrail_rho(
            y,
            sy,
            g,
            sg,
            prior_mean_primary=prior_mean_primary,
            prior_mean_guard=prior_mean_guard,
        )
        rho_hat = float(mom["rho"] if rho is None else rho)
        sd_p = float(prior_sd_primary if prior_sd_primary is not None else mom["tau_primary"])
        sd_g = float(prior_sd_guard if prior_sd_guard is not None else mom["tau_guard"])
        rho_info = {**mom, "rho": rho_hat, "source": "mom" if rho is None else "caller+mom_tau"}
    else:
        rho_hat = float(rho)
        sd_p = float(prior_sd_primary)
        sd_g = float(prior_sd_guard)
        rho_info = {"rho": rho_hat, "n": int(y.size), "source": "caller"}

    if not (np.isfinite(sd_p) and sd_p > 0 and np.isfinite(sd_g) and sd_g > 0):
        raise ValueError("prior_sd_primary and prior_sd_guard must be positive and finite")
    if not (np.isfinite(rho_hat) and abs(rho_hat) < 1.0):
        raise ValueError("rho must be finite and strictly inside (-1, 1)")

    joint = joint_metric_shrinkage(
        y,
        sy,
        g,
        sg,
        rho=rho_hat,
        prior_sd_primary=sd_p,
        prior_sd_guard=sd_g,
        prior_mean_primary=prior_mean_primary,
        prior_mean_guard=prior_mean_guard,
        ci=ci,
    )
    agg = aggregate_shrunk_cumulative(
        joint["primary_shrunk"],
        joint["primary_posterior_sd"],
        shipped=shipped,
        coverage=coverage,
        observed=y,
        aggregation=aggregation,
        ci=ci,
        min_shipped=min_shipped,
    )
    return {
        "method": "joint_metric_shrinkage → Kessler aggregate",
        "rho": rho_hat,
        "rho_info": rho_info,
        "prior_sd_primary": sd_p,
        "prior_sd_guard": sd_g,
        "prior_mean_primary": float(prior_mean_primary),
        "prior_mean_guard": float(prior_mean_guard),
        "joint": joint,
        "shrunk": joint["primary_shrunk"],
        "posterior_sd": joint["primary_posterior_sd"],
        **agg,
    }


def process_level_total_effect(
    effects,
    standard_errors,
    *,
    alpha: float = 0.05,
    alternative: str = "greater",
    n_bootstrap: int = 0,
    ci: float = 0.95,
    random_seed: int | None = None,
) -> dict:
    """
    Lee & Shen (2018) process-level correction for the expected total of winners.

    Estimand is ``E[T_A]`` where ``T_A = sum_{i in A} a_i`` and ``A`` is the
    (random) set of experiments that pass a one-sided launch threshold. The
    naive sum ``S_A`` is biased high; the plug-in debiased total subtracts a
    bias contribution from **every** experiment (selected or not):

    ``hat{T}_A = S_A - sum_i SE_i * phi((SE_i * b_i - X_i) / SE_i)``.

    This differs from :func:`winners_curse_estimate` (conditional on selection)
    and from :func:`cumulative_impact` (Bayesian shrink-then-sum). Prefer
    Bayesian cumulative impact when a prior is available; use this when the
    estimand is specifically the Airbnb process-level total.

    Parameters
    ----------
    effects, standard_errors : array-like
        Experiment lifts and SEs.
    alpha : float
        One-sided significance level for launch (default 0.05).
    alternative : {"greater", "less"}
        Launch direction (Airbnb uses ``"greater"``).
    n_bootstrap : int
        If > 0, parametric bootstrap percentile CI for ``hat{T}_A``.
    ci : float
        Bootstrap CI level (default 0.95).
    random_seed : int, optional
        Bootstrap RNG seed.

    Returns
    -------
    dict
        ``total`` (``hat{T}_A``), ``naive_total`` (``S_A``), ``conditional_total``
        (Zhong–Prentice style sum of conditional debiasings), ``bias_estimate``,
        ``n_selected``, ``selected_mask``, and optional ``ci_lower`` / ``ci_upper``.
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 1:
        raise ValueError("effects must contain at least one estimate")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be 'greater' or 'less' (Lee & Shen one-sided launch)")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ValueError("n_bootstrap must be an integer >= 0")
    if n_bootstrap > 0 and not 0 < ci < 1:
        raise ValueError("ci must be strictly between 0 and 1")

    def _one_shot(x: np.ndarray) -> dict:
        z = float(norm.ppf(1.0 - alpha))
        b = z * s  # thresholds on the effect scale
        if alternative == "greater":
            selected = x > b
            # bias_i = SE * phi((SE*b_z - X)/SE) = SE * phi(z - X/SE)
            bias_terms = s * norm.pdf(z - x / s)
            # conditional: for selected only, SE * phi(z - X/SE) / (1 - Phi(z - X/SE))
            mills = norm.pdf(z - x / s) / np.maximum(norm.sf(z - x / s), 1e-300)
            cond_debias = np.where(selected, x - s * mills, 0.0)
        else:
            selected = x < -b
            bias_terms = s * norm.pdf(-z - x / s)
            mills = norm.pdf(-z - x / s) / np.maximum(norm.cdf(-z - x / s), 1e-300)
            cond_debias = np.where(selected, x + s * mills, 0.0)

        naive = float(np.sum(x[selected])) if selected.any() else 0.0
        bias_est = float(np.sum(bias_terms))
        # For "less", the bias identity is E[I(X<-b)(X-a)] = -SE*phi(...);
        # plug-in subtracts the signed bias so we add bias_terms when less.
        if alternative == "greater":
            total = naive - bias_est
        else:
            total = naive + bias_est
        return {
            "total": float(total),
            "naive_total": naive,
            "conditional_total": float(np.sum(cond_debias)),
            "bias_estimate": bias_est if alternative == "greater" else -bias_est,
            "n_selected": int(selected.sum()),
            "selected_mask": selected,
        }

    out = _one_shot(y)
    out["alternative"] = alternative
    out["alpha"] = float(alpha)

    if n_bootstrap > 0:
        rng = np.random.default_rng(random_seed)
        boots = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            x_b = rng.normal(y, s)
            boots[i] = _one_shot(x_b)["total"]
        gamma = 1.0 - ci
        out["ci_lower"] = float(np.quantile(boots, gamma / 2.0))
        out["ci_upper"] = float(np.quantile(boots, 1.0 - gamma / 2.0))
        out["n_bootstrap"] = n_bootstrap
    return out


def estimate_guardrail_rho(
    primary_effects,
    primary_ses,
    guardrail_effects,
    guardrail_ses,
    *,
    prior_mean_primary: float = 0.0,
    prior_mean_guard: float = 0.0,
) -> dict:
    """
    Method-of-moments estimate of Corr(true primary, true guardrail).

    Under independent sampling noise, ``Cov(X, G) = Cov(delta, gamma)``.
    Prior SDs are Paule–Mandel ``tau`` estimates; ``rho`` is clipped to
    ``(-0.999, 0.999)``. Requires at least 5 paired experiments.

    Returns
    -------
    dict
        ``rho``, ``tau_primary``, ``tau_guard``, ``cov``, ``n``.
    """
    x, sx = _validate_effects_ses(primary_effects, primary_ses)
    g, sg = _validate_effects_ses(guardrail_effects, guardrail_ses)
    if x.shape != g.shape:
        raise ValueError("primary and guardrail arrays must have the same length")
    if x.size < 5:
        raise ValueError("estimate_guardrail_rho requires at least 5 paired experiments")

    tau_p2 = _paule_mandel_tau2(x, sx**2, prior_mean=prior_mean_primary)
    tau_g2 = _paule_mandel_tau2(g, sg**2, prior_mean=prior_mean_guard)
    cov = float(np.mean((x - prior_mean_primary) * (g - prior_mean_guard)))
    tau_p = float(np.sqrt(tau_p2))
    tau_g = float(np.sqrt(tau_g2))
    if tau_p == 0.0 or tau_g == 0.0:
        warnings.warn(
            "estimate_guardrail_rho: one prior SD is 0; returning rho=0.",
            RuntimeWarning,
            stacklevel=2,
        )
        rho = 0.0
    else:
        rho = float(np.clip(cov / (tau_p * tau_g), -0.999, 0.999))
    return {
        "rho": rho,
        "tau_primary": tau_p,
        "tau_guard": tau_g,
        "cov": cov,
        "n": int(x.size),
    }


def fit_normal_prior_map(
    effects,
    standard_errors,
    *,
    mu_prior_sd: float = 1.0,
    tau_prior_scale: float = 0.25,
) -> dict:
    """
    Datadog-style MAP for a hierarchical normal prior ``N(mu, tau^2)``.

    Rescales lifts to mean 0 / SD 1, places ``mu ~ N(0, mu_prior_sd^2)`` and
    ``tau ~ HalfCauchy(0, tau_prior_scale)`` on the rescaled scale, maximizes
    the marginal posterior (Nelder–Mead over ``(mu, log tau)``), then maps
    ``(mu, tau)`` back to the original scale. Returns a dict usable as
    ``prior=`` in :func:`cumulative_impact` / :func:`empirical_bayes_shrinkage`.

    Requires at least 5 estimates (same practical floor as Datadog Cumulative Impact).
    """
    y, s = _validate_effects_ses(effects, standard_errors)
    if y.size < 5:
        raise ValueError("fit_normal_prior_map requires at least 5 estimates")
    if not (np.isfinite(mu_prior_sd) and mu_prior_sd > 0):
        raise ValueError("mu_prior_sd must be positive and finite")
    if not (np.isfinite(tau_prior_scale) and tau_prior_scale > 0):
        raise ValueError("tau_prior_scale must be positive and finite")

    y_bar = float(np.mean(y))
    s_y = float(np.std(y, ddof=1))
    if s_y <= 0:
        # All effects identical: no heterogeneity to learn
        return {
            "prior_mean": y_bar,
            "tau2": 0.0,
            "prior_family": "normal",
            "method": "half_cauchy_map",
            "n": int(y.size),
        }

    y_t = (y - y_bar) / s_y
    s_t = s / s_y

    def neg_log_post(u: np.ndarray) -> float:
        mu, eta = float(u[0]), float(u[1])
        tau = np.exp(eta)
        marg_var = tau**2 + s_t**2
        nll = 0.5 * float(np.sum(np.log(2.0 * np.pi * marg_var) + (y_t - mu) ** 2 / marg_var))
        # mu ~ N(0, mu_prior_sd^2)
        nll += 0.5 * (mu / mu_prior_sd) ** 2 + np.log(mu_prior_sd * np.sqrt(2.0 * np.pi))
        # Half-Cauchy(0, scale): log dens = log(2/pi) - log(scale) - log(1+(tau/scale)^2)
        nll -= np.log(2.0 / np.pi) - np.log(tau_prior_scale) - np.log(1.0 + (tau / tau_prior_scale) ** 2)
        # Jacobian tau = exp(eta): subtract eta from nll <=> add eta to log posterior
        nll -= eta
        return float(nll)

    tau2_mom = _paule_mandel_tau2(y_t, s_t**2, prior_mean=0.0)
    eta0 = np.log(np.sqrt(tau2_mom)) if tau2_mom > 0 else np.log(0.1)
    res = minimize(neg_log_post, x0=np.array([0.0, eta0]), method="Nelder-Mead")
    if not res.success:
        warnings.warn(
            f"fit_normal_prior_map: optimizer reported failure ({res.message}); using last iterate.",
            RuntimeWarning,
            stacklevel=2,
        )
    mu_t, eta = float(res.x[0]), float(res.x[1])
    tau_t = float(np.exp(eta))
    mu = mu_t * s_y + y_bar
    tau = tau_t * s_y
    return {
        "prior_mean": float(mu),
        "tau2": float(tau**2),
        "prior_family": "normal",
        "method": "half_cauchy_map",
        "n": int(y.size),
        "log_posterior": float(-res.fun),
    }
