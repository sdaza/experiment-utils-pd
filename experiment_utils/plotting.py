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
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# shared palette — used by both the public function and the internal renderer
_CLR_SIG = "#1e40af"  # deep indigo-blue — significant
_CLR_NSIG = "#64748b"  # muted slate — not significant
_CLR_META_BG = "#fef2f2"  # faint rose — pooled row background
_CLR_ZERO = "#475569"  # dark slate zero line
_CLR_GUIDE = "#e2e8f0"  # very light guide lines
_CLR_SPINE = "#cbd5e1"  # spine / tick color


def _fmt_label(value: float, significant: int, eff_col: str, decimals: int = 2, pct_points: bool = False) -> str:
    """Format an effect value as a label, appending '*' when significant."""
    if "relative" in eff_col:
        text = f"{value:+.{decimals}%}"
    elif pct_points:
        text = f"{value:+.{decimals}f}pp"
    else:
        text = f"{value:+.{decimals}f}"
    return text + ("*" if significant else "")


def _resolve_ci_cols(
    lo_col: str, hi_col: str, data: pd.DataFrame, sig_col: str, eff_col: str | None = None
) -> tuple[str, str, pd.DataFrame]:
    """Return effective (lo, hi) columns and (possibly augmented) data.

    When Bonferroni/MCP significance is active but the pre-computed MCP CI
    columns are absent, derive them on-the-fly so the displayed CI always
    matches the significance decision.  Without this, a gray (non-significant)
    dot can have a CI that visually excludes 0 — inconsistent with the legend.
    """
    _mcp_lo = lo_col.replace("_lower", "_lower_mcp").replace("effect_lower", "effect_lower_mcp")
    _mcp_hi = hi_col.replace("_upper", "_upper_mcp").replace("effect_upper", "effect_upper_mcp")

    mcp_cols_exist = _mcp_lo in data.columns and _mcp_hi in data.columns and not data[_mcp_lo].isna().all()

    if sig_col != "stat_significance_mcp":
        return lo_col, hi_col, data

    if mcp_cols_exist:
        return _mcp_lo, _mcp_hi, data

    # mcp CI columns are absent — compute them from pvalue ratio or row count.
    if "standard_error" not in data.columns or "pvalue" not in data.columns:
        return lo_col, hi_col, data

    # estimate n_tests from median pvalue_mcp / pvalue ratio, fall back to row count.
    n_tests = len(data)
    if "pvalue_mcp" in data.columns:
        valid = data[(data["pvalue"] > 0) & (data["pvalue_mcp"] < 1) & data["pvalue_mcp"].notna()]
        if not valid.empty:
            n_tests = max(1, round((valid["pvalue_mcp"] / valid["pvalue"]).median()))

    z_adj = stats.norm.ppf(1 - 0.05 / (2 * n_tests))
    _eff_col = eff_col or lo_col.replace("_lower", "").replace("abs_effect", "absolute_effect").replace(
        "rel_effect", "relative_effect"
    )
    if _eff_col not in data.columns:
        return lo_col, hi_col, data

    data = data.copy()
    data[_mcp_lo] = data[_eff_col] - z_adj * data["standard_error"]
    data[_mcp_hi] = data[_eff_col] + z_adj * data["standard_error"]
    return _mcp_lo, _mcp_hi, data


def _draw_panels_into_axes(
    axes: list,
    data: pd.DataFrame,
    unique_panels: list,
    eff_col: str,
    lo_col: str,
    hi_col: str,
    x_label: str,
    sig_col: str,
    meta_df: pd.DataFrame | None,
    show_zero_line: bool,
    sort_by_magnitude: bool,
    panel_col: str,
    row_col: str,
    panel_titles: str | list | dict | None,
    row_labels: dict | None,
    show_values: bool,
    value_decimals: int = 2,
    show_yticklabels: bool | list[bool] = True,
    row_order: dict | None = None,
    pct_points: bool = False,
    combined_label: bool = False,
) -> None:
    """Draw Cleveland-dot panels into a pre-created list of axes (one ax per panel value)."""
    _lo_col, _hi_col, data = _resolve_ci_cols(lo_col, hi_col, data, sig_col, eff_col)

    for panel_idx, (ax, panel_val) in enumerate(zip(axes, unique_panels, strict=False)):
        show_yticks = show_yticklabels[panel_idx] if isinstance(show_yticklabels, list) else show_yticklabels
        od = data[data[panel_col] == panel_val].copy()

        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        if show_zero_line:
            ax.axvline(0, color=_CLR_ZERO, linestyle="-", linewidth=1.0, alpha=0.55, zorder=1)
        cap_h = 0.06

        if row_order is not None and panel_val in row_order:
            order_map = {label: i for i, label in enumerate(row_order[panel_val])}
            od["_row_order"] = od[row_col].map(order_map).fillna(len(order_map))
            od = od.sort_values("_row_order")
        elif sort_by_magnitude:
            od = od.sort_values(by=eff_col, ascending=False)

        labels = list(od[row_col])
        effs = list(od[eff_col])
        los = list(od[_lo_col])
        his = list(od[_hi_col])
        sigs = list(od[sig_col]) if sig_col in od.columns else [0] * len(od)
        # secondary effect for combined_label: the column that is NOT being plotted
        _secondary_col = "absolute_effect" if "relative" in eff_col else "relative_effect"
        secondary_effs = list(od[_secondary_col]) if _secondary_col in od.columns else [None] * len(od)

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
                meta_secondary = float(mo[_secondary_col].iloc[0]) if _secondary_col in mo.columns else None
                secondary_effs = secondary_effs + [meta_secondary]

        n_rows = len(labels)
        n_exp = n_rows - (1 if has_meta else 0)
        y_pos = list(range(n_rows))

        for i in range(n_exp):
            ax.axhline(i, color=_CLR_GUIDE, linewidth=0.6, linestyle=":", zorder=0)

        if has_meta:
            meta_y = n_rows - 1
            ax.axhspan(meta_y - 0.48, meta_y + 0.48, color=_CLR_META_BG, zorder=0)
            ax.axhline(meta_y - 0.52, color=_CLR_SPINE, linewidth=1.2, linestyle=(0, (6, 3)), zorder=1)

        for i, (label, eff, lo, hi, sig, secondary_eff) in enumerate(
            zip(labels, effs, los, his, sigs, secondary_effs, strict=False)
        ):
            if eff is None or not pd.notna(eff) or not np.isfinite(eff):
                continue
            is_meta_row = has_meta and label == meta_label
            if is_meta_row:
                color = _CLR_SIG if sig == 1 else _CLR_NSIG
                marker, dot_size, ci_lw = "D", 70, 1.8
            elif sig == 1:
                color, marker, dot_size, ci_lw = _CLR_SIG, "o", 45, 1.4
            else:
                color, marker, dot_size, ci_lw = _CLR_NSIG, "o", 35, 1.2

            ci_valid = (
                lo is not None
                and hi is not None
                and pd.notna(lo)
                and pd.notna(hi)
                and np.isfinite(lo)
                and np.isfinite(hi)
            )
            if not ci_valid:
                continue
            ax.hlines(i, lo, hi, color=color, linewidth=ci_lw, alpha=0.75, zorder=3)
            ax.vlines(lo, i - cap_h, i + cap_h, color=color, linewidth=ci_lw * 0.75, alpha=0.75, zorder=3)
            ax.vlines(hi, i - cap_h, i + cap_h, color=color, linewidth=ci_lw * 0.75, alpha=0.75, zorder=3)
            ax.scatter(eff, i, color=color, s=dot_size, marker=marker, zorder=5, edgecolors="white", linewidths=0.7)

            if show_values:
                lbl = _fmt_label(eff, sig, eff_col, decimals=value_decimals, pct_points=pct_points)
                if (
                    combined_label
                    and secondary_eff is not None
                    and pd.notna(secondary_eff)
                    and np.isfinite(secondary_eff)
                ):
                    if "relative" in eff_col:
                        # plotting relative → append absolute (pp or raw)
                        if pct_points:
                            lbl = f"{lbl} ({secondary_eff * 100:+.{value_decimals}f}pp)"
                        else:
                            lbl = f"{lbl} ({secondary_eff:+.{value_decimals}f})"
                    else:
                        # plotting absolute → append relative %
                        lbl = f"{lbl} ({secondary_eff:+.{value_decimals}%})"
                ax.text(eff, i - 0.14, lbl, ha="center", va="bottom", fontsize=7.5, color=color, zorder=6)

        ax.set_yticks(y_pos)
        if show_yticks:
            display_labels = [row_labels.get(lbl, lbl) for lbl in labels] if row_labels else labels
            ax.set_yticklabels(display_labels, fontsize=9.5, color="#334155")
        else:
            ax.set_yticklabels([])

        ax.tick_params(axis="y", length=0, pad=8 if show_yticks else 2)
        ax.tick_params(axis="x", labelsize=8.5, colors="#64748b", pad=4)
        if combined_label:
            _note = "Absolute effect in parentheses" if "relative" in eff_col else "Relative effect in parentheses"
            _xlabel = f"{x_label}\n{_note}"
        else:
            _xlabel = x_label
        ax.set_xlabel(_xlabel, fontsize=9.5, color="#64748b", labelpad=6)
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
            ptitle = row_labels.get(panel_val, str(panel_val)) if row_labels else str(panel_val)
        ax.set_title(ptitle, fontsize=11, fontweight="semibold", color="#1e293b", loc="left", pad=18)

        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune="both"))


def _add_legend_and_title(
    fig: plt.Figure,
    figsize: tuple,
    title: str | None,
    sig_label: str,
    meta_df: pd.DataFrame | None,
    panel_spacing: float | None = None,
) -> None:
    """Add a shared legend and optional suptitle, then call tight_layout."""
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

    header_inches = 0.45 + (0.28 if title else 0.0)
    reserved = header_inches / figsize[1]
    fig.tight_layout(rect=[0, 0, 1, 1.0 - reserved])
    if panel_spacing is not None:
        fig.subplots_adjust(wspace=panel_spacing)
    plt.close(fig)


def _make_pooled_meta(
    data: pd.DataFrame,
    panel_col: str,
    eff_col: str,
    lo_col: str,
    hi_col: str,
    sig_col: str,
    alpha: float,
) -> pd.DataFrame | None:
    """Compute IVW pooled estimate per panel from the visible plot data.

    Always derives the pooled row from the same rows shown in the figure,
    so it can never be inconsistent with external pre-computed meta DataFrames.
    Returns None when no panel has at least two valid experiments.
    """
    if "standard_error" not in data.columns:
        return None
    z = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for panel_val in data[panel_col].unique():
        od = data[data[panel_col] == panel_val]
        valid = (
            od["standard_error"].notna()
            & od["standard_error"].apply(np.isfinite)
            & (od["standard_error"] > 0)
            & od[eff_col].notna()
            & od[eff_col].apply(np.isfinite)
        )
        od = od[valid]
        if len(od) < 2:
            continue
        w = 1.0 / (od["standard_error"] ** 2)
        eff = float((w * od[eff_col]).sum() / w.sum())
        se = float(np.sqrt(1.0 / w.sum()))
        pvalue = float(2 * stats.norm.sf(abs(eff / se)))
        rows.append(
            {
                panel_col: panel_val,
                "_label": "Pooled",
                eff_col: eff,
                lo_col: eff - z * se,
                hi_col: eff + z * se,
                "standard_error": se,
                "pvalue": pvalue,
                sig_col: 1 if pvalue < alpha else 0,
            }
        )
    return pd.DataFrame(rows) if rows else None


def _fill_missing_panel_rows(
    data: pd.DataFrame,
    panel_col: str,
    row_col: str,
) -> pd.DataFrame:
    """Add NaN rows for every panel × row-label combination absent from *data*.

    This guarantees that all panels share the same y-positions so dots in
    different panels are vertically aligned.  Missing entries have NaN for
    every column except *panel_col* and *row_col*; the draw loop already
    skips NaN effect rows while keeping the guide-line and tick-label.
    """
    unique_panels = list(data[panel_col].unique())
    # Preserve first-appearance order across all panels
    all_labels = list(dict.fromkeys(data[row_col].tolist()))
    rows_to_add = []
    for panel_val in unique_panels:
        existing = set(data[data[panel_col] == panel_val][row_col])
        for label in all_labels:
            if label not in existing:
                rows_to_add.append({panel_col: panel_val, row_col: label})
    if rows_to_add:
        data = pd.concat([data, pd.DataFrame(rows_to_add)], ignore_index=True)
    return data


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
    show_values: bool = False,
    value_decimals: int = 2,
    panel_spacing: float | None = None,
    repeat_ylabels: bool = False,
    pct_points: bool = False,
    combined_label: bool = False,
) -> plt.Figure:
    """Build and return a single effects figure for *data* (already labelled)."""
    sig_col = "stat_significance_mcp" if "stat_significance_mcp" in data.columns else "stat_significance"
    mcp_method = data["mcp_method"].iloc[0] if "mcp_method" in data.columns else None
    sig_label = f"Significant ({mcp_method}, α={alpha})" if mcp_method else f"Significant (α={alpha})"

    unique_panels = list(data[panel_col].unique())
    n_panels = len(unique_panels)
    data = _fill_missing_panel_rows(data, panel_col, row_col)
    max_rows = max(data[data[panel_col] == p][row_col].nunique() for p in unique_panels)
    if meta_df is not None and panel_col == "outcome":
        max_rows += 1
    if figsize is None:
        fig_h = max(3.5, 0.65 * max_rows + 2.4)
        panel_w = max(3.2, min(5.5, 14.0 / n_panels))
        fig_w = max(5.5, panel_w * n_panels)
        figsize = (fig_w, fig_h)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)

    # derive a stable row order from the first panel so all panels share the same
    # top-to-bottom sequence regardless of per-panel effect magnitudes.
    row_order: dict | None = None
    if n_panels > 1:
        row_order = {}
        first_panel = unique_panels[0]
        pdata = data[data[panel_col] == first_panel]
        if sort_by_magnitude:
            sorted_rows = pdata.sort_values(by=eff_col, ascending=False)
        else:
            sorted_rows = pdata  # preserve natural order of first panel
        reference_order = list(sorted_rows[row_col])
        for pval in unique_panels:
            row_order[pval] = reference_order

    # by default only the leftmost panel shows y-tick labels; repeat_ylabels shows all.
    show_yticklabels = True if repeat_ylabels else [pi == 0 for pi in range(n_panels)]

    _draw_panels_into_axes(
        list(axes.flatten()),
        data,
        unique_panels,
        eff_col,
        lo_col,
        hi_col,
        x_label,
        sig_col,
        meta_df,
        show_zero_line,
        sort_by_magnitude,
        panel_col,
        row_col,
        panel_titles,
        row_labels,
        show_values,
        value_decimals=value_decimals,
        show_yticklabels=show_yticklabels,
        row_order=row_order,
        pct_points=pct_points,
        combined_label=combined_label,
    )

    _add_legend_and_title(fig, figsize, title, sig_label, meta_df, panel_spacing=panel_spacing)
    return fig


def _render_multi_effect_figure(
    data: pd.DataFrame,
    effect_specs: list[tuple],
    alpha: float,
    meta_df: pd.DataFrame | None,
    figsize: tuple | None,
    title: str | None,
    show_zero_line: bool,
    sort_by_magnitude: bool,
    panel_col: str,
    row_col: str,
    panel_titles: str | list | dict | None,
    row_labels: dict | None,
    show_values: bool,
    value_decimals: int = 2,
    panel_spacing: float | None = None,
    repeat_ylabels: bool = False,
    pct_points: bool = False,
    combined_label: bool = False,
) -> plt.Figure:
    """Build a side-by-side figure with one column group per effect type.

    ``effect_specs`` is a list of ``(eff_col, lo_col, hi_col, x_label)`` tuples,
    one per effect type.  Panels for the same outcome are placed adjacent to each
    other: ``[abs_p0 | rel_p0 | abs_p1 | rel_p1 | ...]``.  Row labels and the
    panel title are shown only on the leftmost column of each panel group.
    """
    sig_col = "stat_significance_mcp" if "stat_significance_mcp" in data.columns else "stat_significance"
    mcp_method = data["mcp_method"].iloc[0] if "mcp_method" in data.columns else None
    sig_label = f"Significant ({mcp_method}, α={alpha})" if mcp_method else f"Significant (α={alpha})"

    unique_panels = list(data[panel_col].unique())
    n_panels = len(unique_panels)
    n_effects = len(effect_specs)
    n_cols = n_panels * n_effects
    data = _fill_missing_panel_rows(data, panel_col, row_col)
    max_rows = max(data[data[panel_col] == p][row_col].nunique() for p in unique_panels)
    if meta_df is not None and panel_col == "outcome":
        max_rows += 1

    if figsize is None:
        fig_h = max(3.5, 0.65 * max_rows + 2.4)
        panel_w = max(2.8, min(4.5, 12.0 / n_cols))
        fig_w = max(5.5, panel_w * n_cols)
        figsize = (fig_w, fig_h)

    fig, all_axes = plt.subplots(1, n_cols, figsize=figsize, squeeze=False)

    # derive a stable row order from the first panel so all panels share the
    # same top-to-bottom sequence regardless of which effect is displayed.
    row_order: dict | None = None
    if n_panels > 1:
        row_order = {}
        first_panel = unique_panels[0]
        pdata = data[data[panel_col] == first_panel]
        if sort_by_magnitude:
            first_ec = effect_specs[0][0]
            sorted_rows = pdata.sort_values(by=first_ec, ascending=False)
        else:
            sorted_rows = pdata  # preserve natural order of first panel
        reference_order = list(sorted_rows[row_col])
        for pval in unique_panels:
            row_order[pval] = reference_order

    for eff_idx, (ec, lc, hc, xl) in enumerate(effect_specs):
        # prepare CI columns for this effect type
        eff_data = data.copy()
        z = stats.norm.ppf(1 - alpha / 2)
        if lc not in eff_data.columns or eff_data[lc].isna().all():
            eff_data[lc] = eff_data[ec] - z * eff_data["standard_error"]
            eff_data[hc] = eff_data[ec] + z * eff_data["standard_error"]
        if meta_df is not None:
            _meta = meta_df.copy()
            if lc not in _meta.columns or _meta[lc].isna().all():
                _meta[lc] = _meta[ec] - z * _meta["standard_error"]
                _meta[hc] = _meta[ec] + z * _meta["standard_error"]
        else:
            _meta = None

        # columns for this effect: positions [pi * n_effects + eff_idx] for each panel pi
        panel_axes = [all_axes[0][pi * n_effects + eff_idx] for pi in range(n_panels)]
        show_yticks = eff_idx == 0 or repeat_ylabels

        _draw_panels_into_axes(
            panel_axes,
            eff_data,
            unique_panels,
            ec,
            lc,
            hc,
            xl,
            sig_col,
            _meta,
            show_zero_line,
            sort_by_magnitude,
            panel_col,
            row_col,
            panel_titles if show_yticks else "",
            row_labels,
            show_values,
            value_decimals=value_decimals,
            show_yticklabels=show_yticks,
            row_order=row_order,
            pct_points=(pct_points and ec == "absolute_effect"),
            combined_label=combined_label,
        )

    _add_legend_and_title(fig, figsize, title, sig_label, meta_df, panel_spacing=panel_spacing)
    return fig


def plot_effects(
    results: pd.DataFrame,
    experiment_identifier: str | list[str] | None = None,
    alpha: float = 0.05,
    outcomes: list[str] | str | None = None,
    effect: str | list[str] = "absolute",
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
    show_values: bool = False,
    value_decimals: int | None = None,
    panel_spacing: float | None = None,
    repeat_ylabels: bool = False,
    pct_points: bool = False,
    combined_label: bool = False,
    save_path: str | None = None,
    **kwargs,
) -> plt.Figure | dict[str, plt.Figure] | None:
    if kwargs:
        raise TypeError(f"plot_effects() got unexpected keyword argument(s): {list(kwargs.keys())}")
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
    effect : {"absolute", "relative"} or list, optional
        Which effect metric(s) to display (default ``"absolute"``).
        Pass a list such as ``["absolute", "relative"]`` to produce a
        side-by-side figure with one column group per effect type.
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

    show_values : bool, optional
        Annotate each dot with its effect value (and ``*`` when significant).
        Default ``False``.
    value_decimals : int, optional
        Number of decimal places for the value labels shown when
        ``show_values=True``.  Defaults to ``1`` when ``pct_points=True``
        or when any relative effect is shown (e.g. ``+3.0pp``, ``+15.4%``),
        and ``2`` otherwise (e.g. ``+0.03``).
    panel_spacing : float, optional
        Horizontal whitespace between panels as a fraction of the average axes
        width (passed to ``subplots_adjust(wspace=...)``).  Larger values add
        more room between panels.  ``None`` (default) uses matplotlib's
        automatic spacing.  Try values like ``0.4``–``0.8`` when panels overlap.
    repeat_ylabels : bool, optional
        When ``True``, show y-axis tick labels on every panel instead of only
        the leftmost one.  Useful when panels are far apart or when saving
        individual panels.  Default ``False``.
    pct_points : bool, optional
        When ``True``, multiply absolute effect values (and their confidence
        intervals and standard errors) by 100 for display, expressing them as
        **percentage points** (pp).  The x-axis label becomes
        ``"Absolute Effect (pp)"`` and ``show_values`` annotations gain a
        ``"pp"`` suffix.  Has no effect on relative effect columns.
        Default ``False``.
    combined_label : bool, optional
        When ``True`` and ``show_values=True``, append the relative effect in
        parentheses to each dot annotation, e.g. ``+3.0pp (+15.4%)``.
        Requires ``relative_effect`` to be present in the data.
        Pairs naturally with ``pct_points=True`` and ``effect="absolute"``.
        Default ``False``.
    save_path : str or path-like, optional
        File path to save the figure.  When ``group_by`` produces multiple
        figures the group key is inserted before the file extension, e.g.
        ``"effects.png"`` becomes ``"effects_US.png"``, ``"effects_EU.png"``.
        Supports any format recognised by matplotlib (``png``, ``pdf``,
        ``svg``, …).  ``None`` (default) skips saving.

    Returns
    -------
    matplotlib.figure.Figure, dict[str, matplotlib.figure.Figure], or None
        Single figure when *group_by* is ``None``, otherwise a dict mapping
        each group value to its figure.
    """
    if results is None or results.empty:
        return None

    data = results.copy()

    if outcomes is not None:
        outcomes_list = [outcomes] if isinstance(outcomes, str) else list(outcomes)
        data = data[data["outcome"].isin(outcomes_list)]
    unique_outcomes = list(data["outcome"].unique())
    if not unique_outcomes:
        return None

    if comparison is not None:
        pairs = [comparison] if isinstance(comparison, tuple) else list(comparison)
        mask = pd.Series(False, index=data.index)
        for t_val, c_val in pairs:
            mask |= (data["treatment_group"] == t_val) & (data["control_group"] == c_val)
        data = data[mask]
    if data.empty:
        return None

    # Suppress rows with degenerate SE (zero, NaN, inf) so the plot is consistent
    # with the pooled estimate, which applies the same validity filter.
    if "standard_error" in data.columns:
        bad_se = ~(data["standard_error"].notna() & np.isfinite(data["standard_error"]) & (data["standard_error"] > 0))
        data.loc[bad_se, "absolute_effect"] = np.nan
        data.loc[bad_se, "relative_effect"] = np.nan

    # Scale absolute effect columns to percentage points for display.
    if pct_points:
        abs_scale_cols = ["absolute_effect", "abs_effect_lower", "abs_effect_upper", "standard_error"]
        for col in abs_scale_cols:
            if col in data.columns:
                data[col] = data[col] * 100
        if meta_df is not None:
            meta_df = meta_df.copy()
            for col in abs_scale_cols:
                if col in meta_df.columns:
                    meta_df[col] = meta_df[col] * 100

    if value_decimals is None:
        has_relative = any(e == "relative" for e in ([effect] if isinstance(effect, str) else list(effect)))
        value_decimals = 1 if (pct_points or has_relative or combined_label) else 2

    effects: list[str] = [effect] if isinstance(effect, str) else list(effect)

    def _effect_cols(e: str) -> tuple[str, str, str, str]:
        if e == "relative":
            return "relative_effect", "rel_effect_lower", "rel_effect_upper", "Relative Effect"
        if pct_points:
            return "absolute_effect", "abs_effect_lower", "abs_effect_upper", "Absolute Effect (pp)"
        return "absolute_effect", "abs_effect_lower", "abs_effect_upper", "Absolute Effect"

    z = stats.norm.ppf(1 - alpha / 2)

    if len(effects) == 1:
        eff_col, lo_col, hi_col, x_label = _effect_cols(effects[0])
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
    else:
        eff_col = lo_col = hi_col = x_label = None
        if meta_df is not None:
            meta_df = meta_df.copy()
            if "_label" not in meta_df.columns:
                meta_df["_label"] = "Pooled"

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

    if y == "outcome":
        panel_col = "_label"
        row_col = "outcome"
    else:
        panel_col = "outcome"
        row_col = "_label"

    shared_kw = dict(
        alpha=alpha,
        meta_df=meta_df,
        figsize=figsize,
        show_zero_line=show_zero_line,
        sort_by_magnitude=sort_by_magnitude,
        panel_col=panel_col,
        row_col=row_col,
        panel_titles=panel_titles,
        row_labels=row_labels,
        show_values=show_values,
        value_decimals=value_decimals,
        panel_spacing=panel_spacing,
        repeat_ylabels=repeat_ylabels,
        pct_points=pct_points,
        combined_label=combined_label,
    )

    def _render(labelled: pd.DataFrame, fig_title: str | None, group_meta: pd.DataFrame | None = None) -> plt.Figure:
        # Always recompute the pooled row from the visible data so it is
        # guaranteed to be the IVW of the individual rows shown, regardless
        # of how (or on what dataset) the external meta_df was computed.
        if group_meta is not None and len(effects) == 1:
            sig_col_local = (
                "stat_significance_mcp" if "stat_significance_mcp" in labelled.columns else "stat_significance"
            )
            effective_meta = _make_pooled_meta(labelled, panel_col, eff_col, lo_col, hi_col, sig_col_local, alpha)
        else:
            effective_meta = group_meta  # None or multi-effect passthrough

        kw = {**shared_kw, "meta_df": effective_meta}
        if len(effects) == 1:
            return _render_effects_figure(
                labelled,
                unique_outcomes=unique_outcomes,
                eff_col=eff_col,
                lo_col=lo_col,
                hi_col=hi_col,
                x_label=x_label,
                title=fig_title,
                **kw,
            )
        effect_specs = [_effect_cols(e) for e in effects]
        return _render_multi_effect_figure(
            labelled,
            effect_specs=effect_specs,
            title=fig_title,
            **kw,
        )

    def _filter_meta_for_group(group_key, cols: list[str]) -> pd.DataFrame | None:
        """Return the slice of meta_df relevant to this group, or None if not filterable."""
        if meta_df is None:
            return None
        matching_cols = [c for c in cols if c in meta_df.columns]
        if not matching_cols:
            return None  # meta_df has no group cols — it's a global pool, suppress it
        key_vals = list(group_key) if isinstance(group_key, tuple) else [group_key]
        mask = pd.Series(True, index=meta_df.index)
        for col, val in zip(matching_cols, key_vals[: len(matching_cols)], strict=False):
            mask &= meta_df[col] == val
        filtered = meta_df[mask]
        return filtered if not filtered.empty else None

    if group_cols:
        figures: dict[str, plt.Figure] = {}
        for group_key, group_data in data.groupby(group_cols):
            key_str = " | ".join(str(v) for v in group_key) if isinstance(group_key, tuple) else str(group_key)
            fig_title = title if title is not None else key_str
            labelled = _build_labels(group_data)
            group_meta = _filter_meta_for_group(group_key, group_cols)
            figures[key_str] = _render(labelled, fig_title, group_meta=group_meta)
        if save_path is not None:
            import os

            base, ext = os.path.splitext(save_path)
            ext = ext or ".png"
            for key_str, fig in figures.items():
                safe_key = key_str.replace(" | ", "_").replace(" ", "_")
                fig.savefig(f"{base}_{safe_key}{ext}", bbox_inches="tight")
        return figures

    labelled = _build_labels(data)
    fig = _render(labelled, title, group_meta=meta_df)
    if save_path is not None and fig is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_power(
    data: pd.DataFrame,
    comparisons: list[tuple],
    alpha: float = 0.05,
    alternative: str = "two-sided",
    facet_by: str | None = "comparison",
    color_by: str = "effect",
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
    color_by : str, optional
        Column used to colour lines within each subplot (default ``"effect"``).
        Any column in the grid output is valid, e.g. ``"baseline"`` or ``"compliance"``.
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

    legend_title = color_by.replace("_", " ").title()

    def _fmt_color(val):
        if color_by == "effect":
            try:
                return f"+{float(val):.3f}"
            except (ValueError, TypeError):
                pass
        return str(val)

    for facet_val in facet_values:
        subset = temp if facet_by is None else temp[temp[facet_by] == facet_val]

        plot_data = subset.copy()
        plot_data["sample_size"] = plot_data["sample_size"].astype(str)

        color_col = f"__{color_by}_fmt"
        plot_data[color_col] = plot_data[color_by].apply(_fmt_color)
        try:
            color_order_raw = sorted(subset[color_by].unique(), key=float)
        except (TypeError, ValueError):
            color_order_raw = sorted(subset[color_by].unique(), key=str)
        color_order = [_fmt_color(v) for v in color_order_raw]

        fig, ax = plt.subplots()
        sns.lineplot(
            x="sample_size",
            y="power",
            hue=color_col,
            hue_order=color_order,
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
