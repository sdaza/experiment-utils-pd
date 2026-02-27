"""
Standalone plotting functions for experiment-utils.

All functions accept plain DataFrames so they can be used independently
of ``ExperimentAnalyzer`` and ``PowerSim``.  The class methods are thin
wrappers that extract the relevant data and delegate here.
"""

from __future__ import annotations

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# plot_effects
# ─────────────────────────────────────────────────────────────────────────────

# Shared palette — used by both the public function and the internal renderer
_CLR_SIG = "#1e40af"  # deep indigo-blue — significant
_CLR_NSIG = "#64748b"  # muted slate — not significant
_CLR_META_BG = "#fef2f2"  # faint rose — pooled row background
_CLR_ZERO = "#475569"  # dark slate zero line
_CLR_GUIDE = "#e2e8f0"  # very light guide lines
_CLR_SPINE = "#cbd5e1"  # spine / tick color


def _fmt_label(value: float, significant: int, eff_col: str) -> str:
    """Format an effect value as a label, appending '*' when significant."""
    if "relative" in eff_col:
        text = f"{value:+.2%}"
    else:
        text = f"{value:+.2f}"
    return text + ("*" if significant else "")


def _render_effects_figure(
    data: pd.DataFrame,
    unique_outcomes: list[str],
    eff_col: str,
    lo_col: str,
    hi_col: str,
    x_label: str,
    alpha: float,
    meta_df: pd.DataFrame | None,
    figsize: tuple | None,
    title: str | None,
    show_zero_line: bool,
    sort_by_magnitude: bool,
    panel_col: str = "outcome",
    row_col: str = "_label",
    panel_titles: str | list | dict | None = None,
    row_labels: dict | None = None,
    show_labels: bool = False,
) -> plt.Figure:
    """Build and return a single effects figure for *data* (already labelled).

    Parameters
    ----------
    panel_col : str
        Column whose unique values become separate subplots (default ``"outcome"``).
    row_col : str
        Column whose values become y-axis rows within each subplot
        (default ``"_label"``).
    """
    # Use MCP-adjusted significance when available
    sig_col = "stat_significance_mcp" if "stat_significance_mcp" in data.columns else "stat_significance"
    mcp_method = data["mcp_method"].iloc[0] if "mcp_method" in data.columns else None
    sig_label = f"Significant ({mcp_method}, α={alpha})" if mcp_method else f"Significant (α={alpha})"
    # Use MCP-adjusted CI bounds to keep bars visually consistent with significance colors
    _mcp_lo = lo_col.replace("_lower", "_lower_mcp").replace("effect_lower", "effect_lower_mcp")
    _mcp_hi = hi_col.replace("_upper", "_upper_mcp").replace("effect_upper", "effect_upper_mcp")
    use_mcp_ci = (
        sig_col == "stat_significance_mcp"
        and _mcp_lo in data.columns
        and _mcp_hi in data.columns
        and not data[_mcp_lo].isna().all()
    )
    _lo_col = _mcp_lo if use_mcp_ci else lo_col
    _hi_col = _mcp_hi if use_mcp_ci else hi_col

    unique_panels = list(data[panel_col].unique())
    n_panels = len(unique_panels)
    max_rows = max(data[data[panel_col] == p][row_col].nunique() for p in unique_panels)
    if meta_df is not None and panel_col == "outcome":
        max_rows += 1
    if figsize is None:
        fig_h = max(3.5, 0.65 * max_rows + 2.4)
        # Cap per-panel width so many-panel layouts stay readable
        panel_w = max(3.2, min(5.5, 14.0 / n_panels))
        fig_w = max(5.5, panel_w * n_panels)
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)

    for panel_idx, (ax, panel_val) in enumerate(zip(axes.flatten(), unique_panels, strict=False)):
        od = data[data[panel_col] == panel_val].copy()

        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        if show_zero_line:
            ax.axvline(0, color=_CLR_ZERO, linestyle="-", linewidth=1.0, alpha=0.55, zorder=1)
        cap_h = 0.06

        if sort_by_magnitude:
            od = od.sort_values(by=eff_col, ascending=False)

        labels = list(od[row_col])
        effs = list(od[eff_col])
        los = list(od[_lo_col])
        his = list(od[_hi_col])
        sigs = list(od[sig_col]) if sig_col in od.columns else [0] * len(od)

        has_meta = False
        meta_label = None
        if meta_df is not None and panel_col == "outcome":
            outcome = panel_val
            mo = meta_df[meta_df["outcome"] == outcome] if "outcome" in meta_df.columns else meta_df
            if not mo.empty:
                has_meta = True
                meta_label = "◆ " + mo["_label"].iloc[0]
                labels = labels + [meta_label]
                effs = effs + [float(mo[eff_col].iloc[0])]
                _meta_lo = _lo_col if _lo_col in mo.columns else lo_col
                _meta_hi = _hi_col if _hi_col in mo.columns else hi_col
                los = los + [float(mo[_meta_lo].iloc[0])]
                his = his + [float(mo[_meta_hi].iloc[0])]
                meta_sig_col = "stat_significance_mcp" if "stat_significance_mcp" in mo.columns else "stat_significance"
                meta_sig = int(mo[meta_sig_col].iloc[0]) if meta_sig_col in mo.columns else 0
                sigs = sigs + [meta_sig]

        n_rows = len(labels)
        n_exp = n_rows - (1 if has_meta else 0)
        y_pos = list(range(n_rows))

        for i in range(n_exp):
            ax.axhline(i, color=_CLR_GUIDE, linewidth=0.6, linestyle=":", zorder=0)

        if has_meta:
            meta_y = n_rows - 1
            ax.axhspan(meta_y - 0.48, meta_y + 0.48, color=_CLR_META_BG, zorder=0)
            ax.axhline(meta_y - 0.52, color=_CLR_SPINE, linewidth=1.2, linestyle=(0, (6, 3)), zorder=1)

        for i, (label, eff, lo, hi, sig) in enumerate(zip(labels, effs, los, his, sigs, strict=False)):
            if eff is None:
                continue
            is_meta_row = has_meta and label == meta_label
            if is_meta_row:
                color = _CLR_SIG if sig == 1 else _CLR_NSIG
                marker, dot_size, ci_lw = "D", 70, 1.8
            elif sig == 1:
                color, marker, dot_size, ci_lw = _CLR_SIG, "o", 45, 1.4
            else:
                color, marker, dot_size, ci_lw = _CLR_NSIG, "o", 35, 1.2

            ax.hlines(i, lo, hi, color=color, linewidth=ci_lw, alpha=0.75, zorder=3)
            ax.vlines(lo, i - cap_h, i + cap_h, color=color, linewidth=ci_lw * 0.75, alpha=0.75, zorder=3)
            ax.vlines(hi, i - cap_h, i + cap_h, color=color, linewidth=ci_lw * 0.75, alpha=0.75, zorder=3)
            ax.scatter(eff, i, color=color, s=dot_size, marker=marker, zorder=5, edgecolors="white", linewidths=0.7)

            if show_labels:
                lbl = _fmt_label(eff, sig, eff_col)
                ax.text(eff, i - 0.14, lbl, ha="center", va="bottom", fontsize=7.5, color=color, zorder=6)

        ax.set_yticks(y_pos)
        display_labels = [row_labels.get(lbl, lbl) for lbl in labels] if row_labels else labels
        ax.set_yticklabels(display_labels, fontsize=9.5, color="#334155")

        # ── shared axis styling ───────────────────────────────────────
        ax.tick_params(axis="y", length=0, pad=8)
        ax.tick_params(axis="x", labelsize=8.5, colors="#64748b", pad=4)
        ax.set_xlabel(x_label, fontsize=9.5, color="#64748b", labelpad=6)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.invert_yaxis()
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(_CLR_SPINE)
        ax.spines["bottom"].set_linewidth(0.8)
        if isinstance(panel_titles, dict):
            ptitle = panel_titles.get(panel_val, str(panel_val))
        elif isinstance(panel_titles, list):
            ptitle = panel_titles[panel_idx] if panel_idx < len(panel_titles) else str(panel_val)
        elif isinstance(panel_titles, str):
            ptitle = panel_titles
        else:
            ptitle = str(panel_val)
        ax.set_title(ptitle, fontsize=11, fontweight="semibold", color="#1e293b", loc="left", pad=18)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))

    # ── legend ────────────────────────────────────────────────────────
    legend_items = [
        mlines.Line2D([], [], color=_CLR_SIG, marker="o", linestyle="-", markersize=6, label=sig_label),
        mlines.Line2D([], [], color=_CLR_NSIG, marker="o", linestyle="-", markersize=6, label="Not significant"),
    ]
    if meta_df is not None:
        legend_items.append(
            mlines.Line2D(
                [], [], color="#475569", marker="D", linestyle="None", markersize=6, label="Pooled (meta-analysis)"
            )
        )

    top_anchor = 0.97
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", color="#0f172a", y=top_anchor)
        top_anchor -= 0.07

    fig.legend(
        handles=legend_items,
        loc="upper center",
        ncol=len(legend_items),
        bbox_to_anchor=(0.5, top_anchor),
        frameon=False,
        fontsize=9,
        handlelength=1.6,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    # Reserve a fixed height in inches for the header (legend + optional suptitle)
    # so the gap stays constant regardless of figure height.
    header_inches = 0.45 + (0.28 if title else 0.0)
    reserved = header_inches / figsize[1]
    # Use fig.tight_layout (not plt.tight_layout) to avoid triggering an extra
    # inline display in Jupyter notebooks via the global pyplot state machine.
    fig.tight_layout(rect=[0, 0, 1, 1.0 - reserved])
    plt.close(fig)
    return fig


def plot_effects(
    results: pd.DataFrame,
    experiment_identifier: str | list[str] | None = None,
    alpha: float = 0.05,
    outcomes: list[str] | str | None = None,
    effect: str = "absolute",
    meta_df: pd.DataFrame | None = None,
    comparison: tuple | list[tuple] | None = None,
    figsize: tuple | None = None,
    title: str | None = None,
    show_zero_line: bool = True,
    sort_by_magnitude: bool = True,
    group_by: str | list[str] | None = None,
    y: str = "experiment",
    panel_titles: str | list | dict | None = None,
    row_labels: dict | None = None,
    show_labels: bool = False,
) -> plt.Figure | dict[str, plt.Figure] | None:
    """
    Cleveland dot plot of treatment effects across experiments.

    Each outcome gets its own panel.  Experiments are shown as rows with
    a dot at the point estimate and a bracketed confidence interval.
    An optional pooled (meta-analysis) row is appended at the bottom.

    Parameters
    ----------
    results : pd.DataFrame
        Output of ``ExperimentAnalyzer.get_effects()``.
    experiment_identifier : str or list[str], optional
        Column(s) used to label each row.  Falls back to
        ``"treatment_group vs control_group"`` when ``None``.
    alpha : float, optional
        Significance level used for the legend label (default 0.05).
    outcomes : str or list[str], optional
        Outcomes to include.  ``None`` shows all outcomes in *results*.
    effect : {"absolute", "relative"}, optional
        Which effect metric to display (default ``"absolute"``).
    meta_df : pd.DataFrame, optional
        Pre-computed ``combine_effects()`` result to append as a pooled row.
        Pass ``None`` (default) to omit the pooled row.
    comparison : tuple or list of tuple, optional
        ``(treatment_group, control_group)`` to restrict the plot to one
        specific comparison, or a list of such tuples to include multiple
        comparisons, e.g.
        ``[('variant_1', 'control'), ('variant_1', 'variant_2')]``.
    figsize : tuple, optional
        ``(width, height)`` in inches.  Auto-sized when ``None``.
    title : str, optional
        Figure-level suptitle.  When *group_by* is set and *title* is ``None``,
        the group value is used as the suptitle automatically.
    show_zero_line : bool, optional
        Vertical reference line at zero (default ``True``).
    sort_by_magnitude : bool, optional
        Sort rows within each panel by effect value descending (default ``True``).
    group_by : str or list[str], optional
        Column(s) to split into separate figures — one figure per unique value.
        Row labels are built from ``experiment_identifier`` minus these columns.
        Returns a ``dict`` keyed by the group value instead of a single figure.
    y : {"experiment", "outcome"}, optional
        What to place on the y-axis (rows).

        - ``"experiment"`` (default) — rows = experiment labels, panels = outcomes.
        - ``"outcome"`` — rows = outcomes, panels = experiment labels.

    panel_titles : str or dict, optional
        Override the auto-generated panel (subplot) titles.

        - ``None`` (default) — use the panel value as the title.
        - ``str`` — use the same string for every panel (``""`` hides all).
        - ``list`` — titles in panel order, e.g. ``["Revenue ($)", "CVR"]``.
        - ``dict`` — map each panel value to a custom display string.

        What counts as a "panel value" depends on *y*:

        - ``y="experiment"`` (default): one panel per **outcome**, so keys
          are outcome names, e.g.
          ``{"revenue": "Revenue ($)", "converted": "Conversion rate"}``.
        - ``y="outcome"``: one panel per **experiment label**, so keys are
          the auto-generated experiment labels (``experiment_identifier``
          column values joined with ``" | "``), e.g.
          ``{"US | email": "US — Email campaign"}``.

    row_labels : dict, optional
        Rename individual row labels on the y-axis.  Keys are the
        auto-generated labels (column values joined with ``" | "``); values
        are the display strings to use instead.
        e.g. ``{"US | email": "Email (US)", "EU | push": "Push (EU)"}``.
        Rows not in the dict keep their auto-generated label.

    show_labels : bool, optional
        Annotate each dot with its effect value (and ``*`` when significant).
        Default ``False``.

    Returns
    -------
    matplotlib.figure.Figure, dict[str, matplotlib.figure.Figure], or None
        Single figure when *group_by* is ``None``, otherwise a dict mapping
        each group value to its figure.
    """
    if results is None or results.empty:
        return None

    data = results.copy()

    # ── filter outcomes ───────────────────────────────────────────────
    if outcomes is not None:
        outcomes_list = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        data = data[data["outcome"].isin(outcomes_list)]
    unique_outcomes = list(data["outcome"].unique())
    if not unique_outcomes:
        return None

    # ── filter comparison ─────────────────────────────────────────────
    if comparison is not None:
        pairs = [comparison] if isinstance(comparison, tuple) else list(comparison)
        mask = pd.Series(False, index=data.index)
        for t_val, c_val in pairs:
            mask |= (data["treatment_group"] == t_val) & (data["control_group"] == c_val)
        data = data[mask]
    if data.empty:
        return None

    # ── effect columns + CI fallback ──────────────────────────────────
    if effect == "relative":
        eff_col, lo_col, hi_col = "relative_effect", "rel_effect_lower", "rel_effect_upper"
        x_label = "Relative Effect"
    else:
        eff_col, lo_col, hi_col = "absolute_effect", "abs_effect_lower", "abs_effect_upper"
        x_label = "Absolute Effect"

    z = stats.norm.ppf(1 - alpha / 2)
    if lo_col not in data.columns or data[lo_col].isna().all():
        data[lo_col] = data[eff_col] - z * data["standard_error"]
        data[hi_col] = data[eff_col] + z * data["standard_error"]

    if meta_df is not None:
        meta_df = meta_df.copy()
        if "_label" not in meta_df.columns:
            meta_df["_label"] = "Pooled"
        if lo_col not in meta_df.columns or meta_df[lo_col].isna().all():
            meta_df[lo_col] = meta_df[eff_col] - z * meta_df["standard_error"]
            meta_df[hi_col] = meta_df[eff_col] + z * meta_df["standard_error"]

    # ── resolve group_by and label columns ───────────────────────────
    group_cols: list[str] = []
    if group_by is not None:
        gc = [group_by] if isinstance(group_by, str) else list(group_by)
        group_cols = [c for c in gc if c in data.columns]

    exp_id = experiment_identifier
    exp_id_list = ([exp_id] if isinstance(exp_id, str) else list(exp_id)) if exp_id else []
    label_cols = [c for c in exp_id_list if c not in group_cols]

    def _build_labels(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        available = [c for c in label_cols if c in df.columns]
        if available:
            df["_label"] = df[available].astype(str).agg(" | ".join, axis=1)
        elif exp_id_list:
            df["_label"] = "Experiment"
        else:
            df["_label"] = df["treatment_group"].astype(str) + " vs " + df["control_group"].astype(str)

        n_comp = df.groupby(["treatment_group", "control_group"]).ngroups
        if n_comp > 1 and available:
            df["_label"] = (
                df["_label"]
                + "\n("
                + df["treatment_group"].astype(str)
                + " vs "
                + df["control_group"].astype(str)
                + ")"
            )
        return df

    # ── render: grouped or single ─────────────────────────────────────
    if y == "outcome":
        panel_col = "_label"
        row_col = "outcome"
    else:
        panel_col = "outcome"
        row_col = "_label"

    render_kw = dict(
        unique_outcomes=unique_outcomes,
        eff_col=eff_col,
        lo_col=lo_col,
        hi_col=hi_col,
        x_label=x_label,
        alpha=alpha,
        meta_df=meta_df,
        figsize=figsize,
        show_zero_line=show_zero_line,
        sort_by_magnitude=sort_by_magnitude,
        panel_col=panel_col,
        row_col=row_col,
        panel_titles=panel_titles,
        row_labels=row_labels,
        show_labels=show_labels,
    )

    if group_cols:
        figures: dict[str, plt.Figure] = {}
        for group_key, group_data in data.groupby(group_cols):
            key_str = " | ".join(str(v) for v in group_key) if isinstance(group_key, tuple) else str(group_key)
            fig_title = title if title is not None else key_str
            labelled = _build_labels(group_data)
            figures[key_str] = _render_effects_figure(labelled, title=fig_title, **render_kw)
        return figures

    labelled = _build_labels(data)
    return _render_effects_figure(labelled, title=title, **render_kw)


# ─────────────────────────────────────────────────────────────────────────────
# plot_power
# ─────────────────────────────────────────────────────────────────────────────


def plot_power(
    data: pd.DataFrame,
    comparisons: list[tuple],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    facet_by: str | None = "comparison",
    hue: str = "effect",
) -> None:
    """
    Plot statistical power by sample size and effect size.

    Parameters
    ----------
    data : pd.DataFrame
        Output of ``PowerSim.grid_sim_power()``.
    comparisons : list of tuple
        List of ``(treatment, control)`` pairs from the ``PowerSim`` instance.
    alpha : float, optional
        Significance level used in the plot title (default 0.05).
    alternative : str, optional
        Hypothesis direction used in the plot title (default ``"two-sided"``).
    facet_by : str or None, optional
        Column whose unique values each get their own subplot
        (default ``"comparison"``).  Pass ``None`` for a single combined plot.
    hue : str, optional
        Column used to colour lines within each subplot (default ``"effect"``).
    """
    from matplotlib.lines import Line2D

    value_vars = [str((i, j)) for i, j in comparisons]

    base_cols = [
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
    ]
    cols = [c for c in base_cols if c in data.columns]

    temp = pd.melt(data, id_vars=cols, var_name="comparison", value_name="power", value_vars=value_vars)

    try:
        temp = temp.sort_values("sample_size")
    except TypeError:
        pass

    facet_values = [None] if facet_by is None else sorted(temp[facet_by].unique(), key=str)

    all_sizes = sorted(temp["sample_size"].unique())
    size_labels = [str(s) for s in all_sizes]

    baselines = sorted(data["baseline"].unique())
    baseline_str = ", ".join(str(b) for b in baselines)

    legend_title = hue.replace("_", " ").title()

    def _fmt_hue(val):
        if hue == "effect":
            try:
                return f"+{float(val):.3f}"
            except (ValueError, TypeError):
                pass
        return str(val)

    for facet_val in facet_values:
        subset = temp if facet_by is None else temp[temp[facet_by] == facet_val]

        plot_data = subset.copy()
        plot_data["sample_size"] = plot_data["sample_size"].astype(str)

        hue_col = f"__{hue}_fmt"
        plot_data[hue_col] = plot_data[hue].apply(_fmt_hue)
        try:
            hue_order_raw = sorted(subset[hue].unique(), key=float)
        except (TypeError, ValueError):
            hue_order_raw = sorted(subset[hue].unique(), key=str)
        hue_order = [_fmt_hue(v) for v in hue_order_raw]

        fig, ax = plt.subplots()
        sns.lineplot(
            x="sample_size",
            y="power",
            hue=hue_col,
            hue_order=hue_order,
            marker="o",
            errorbar=None,
            data=plot_data,
            legend="full",
            ax=ax,
        )

        ax.axhline(y=0.8, linestyle="--", color="gray", linewidth=1)

        ax.set_xticks(range(len(size_labels)))
        ax.set_xticklabels(size_labels, rotation=45, ha="right")

        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="gray", linestyle="--", linewidth=1))
        labels.append("80% threshold")
        ax.legend(
            handles=handles,
            labels=labels,
            title=legend_title,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        title_suffix = "" if facet_by is None else f"  [{facet_by}={facet_val}]"
        ax.set_title(
            f"Power by Sample Size and Effect Size{title_suffix}\n"
            f"(baseline = {baseline_str},  α = {alpha},  {alternative})"
        )

        ax.set_xlabel("Sample size (per group)")
        ax.set_ylabel("Power")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
