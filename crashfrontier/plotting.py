from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter

from crashfrontier.config import COLORS, LABELS, PALETTE, SET_ORDER
from crashfrontier.data.cross_section import CrossSection
from crashfrontier.frontier import Frontier
from crashfrontier.riskneutral.certificates import DualCertificate

__all__ = [
    "plot_binding_quotes",
    "plot_conditional_depth",
    "plot_frontier",
    "plot_frontier_panels",
    "plot_quotes",
    "plot_report",
    "plot_term_structure",
    "plot_threshold_profile",
    "plot_time_series",
    "save_figure",
]

INK = "#17202A"
MUTED = "#5E6D7C"
GRID_COLOR = "#E8EDF2"
SPINE = "#CBD3DC"
FONT_SIZE = 8.5
TICK_SIZE = 7.5
TITLE_SIZE = 11.0
CAPTION_SIZE = 8.0
PANEL_TITLE_SIZE = 9.0
ANNOTATION_SIZE = 7.0
INNER_LINEWIDTH = 1.3
QUOTE_LINEWIDTH = 1.3
QUOTE_MARKER_SIZE = 8.0
OUTER_LINEWIDTH = 1.05
OUTER_LINESTYLE = (0, (1.6, 1.35))
BOUNDS_LINESTYLE = (0, (0.7, 1.5))
GRID_LINEWIDTH = 0.4
SPINE_WIDTH = 0.6
TICK_WIDTH = 0.5
TICK_LENGTH = 2.4
FILL_ALPHAS = (0.10, 0.14, 0.18)
FILL_ALPHA_MAX = 0.18
BAND_FILL_ALPHA = 0.11
BAND_EDGE_WIDTH = 0.6
BAND_EDGE_ALPHA = 0.4
MIDLINE_WIDTH = 1.35
UNUSED_ALPHA = 0.3
INTRINSIC_COLOR = "#C5CED6"
INTRINSIC_LINEWIDTH = 0.4
THRESHOLD_LINEWIDTH = 0.7
SEPARATOR_COLOR = "#8E99A4"
ANNOTATION_EDGE = "#E8EDF2"
SERIES_COLOR = "#2C4E86"
SHORT_LABELS = {
    "put_pair": "Pair",
    "local_eight": "Eight",
    "put_wing": "Put wing",
    "full_otm": "OTM",
    "full_panel": "All",
}
ROLE_COLORS = {
    "otm_put": "#5C4B9A",
    "otm_call": "#C2512C",
    "itm": "#7E8A96",
}
ROLE_LABELS = {
    "otm_put": "OTM put",
    "otm_call": "OTM call",
    "itm": "ITM",
}
INSET_BOUNDS = (0.08, 0.08, 0.34, 0.34)
DEPTH_ALL_PAD = 1.02
DEPTH_LAST_PAD = 1.28
DEPTH_INSET_RATIO = 1.35
PANELS_SIZE = (7.2, 3.1)
REPORT_SIZE = (9.2, 7.0)
DEFAULT_SIZE = (4.2, 3.3)
DATE_MINTICKS = 3
DATE_MAXTICKS = 7
PROB_SCALE = 100.0
DEPTH_SCALE = 100.0
SAVE_DPI = 300
SAVE_PAD_INCHES = 0.12
HIGHLIGHT_SIZE = 36.0
WEIGHT_DISPLAY_CUTOFF = 1e-8
LAYOUT_PADS = dict(w_pad=0.03, h_pad=0.04, hspace=0.06, wspace=0.06)
COMPOSITE_PADS = dict(w_pad=0.05, h_pad=0.10, hspace=0.16, wspace=0.10)
_KNOWN_IMAGE_SUFFIXES = {".pdf", ".png", ".svg", ".eps", ".jpg", ".jpeg"}


def save_figure(
    fig: Figure,
    path_stem: str | Path,
    formats: Sequence[str] = ("pdf", "png"),
    dpi: int = SAVE_DPI,
) -> list[Path]:
    """Write ``fig`` next to ``path_stem`` in each requested format."""
    stem = Path(path_stem)
    if stem.suffix.lower() in _KNOWN_IMAGE_SUFFIXES:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for raw in formats:
        suffix = str(raw).lower().lstrip(".")
        path = Path(f"{stem}.{suffix}")
        fig.savefig(
            path,
            dpi=int(dpi),
            format=suffix,
            facecolor="white",
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
        )
        written.append(path)
    return written


def plot_frontier(
    fronts: Frontier | Mapping[str, Frontier] | Sequence[Frontier],
    *,
    ax: Axes | None = None,
    fill: bool = True,
    outline_outer: bool = True,
    show_bounds: bool = False,
    loss_unit: str = "bp",
    annotate: bool = True,
) -> Figure:
    """Probability–loss crash frontier, inner fill with optional outer outline."""
    named = _as_frontiers(fronts)
    loss_scale, loss_name = _loss_scale(loss_unit)
    fig, ax = _axes(ax)
    colors = _colors_for(named)
    handles: list[Patch | Line2D] = []
    for index, (name, frontier) in enumerate(named.items()):
        color = colors[name]
        label = LABELS.get(name, name)
        alpha = _fill_alpha(index, len(named))
        inner = _scaled(_cloud(frontier.inner), PROB_SCALE, loss_scale)
        outer = _scaled(_cloud(frontier.outer), PROB_SCALE, loss_scale)
        if show_bounds:
            _draw_bounds(ax, frontier, color, loss_scale)
        handle = _draw_set(ax, inner, color, alpha, label, fill=fill)
        if handle is not None:
            handles.append(handle)
        if outline_outer and _outer_is_distinct(frontier):
            _draw_outline(ax, outer, color, OUTER_LINEWIDTH, linestyle=OUTER_LINESTYLE)
    _style_axes(ax)
    prob_label, loss_label, _depth_label = _axis_labels(named)
    _set_axis_label(ax, "x", f"{prob_label} (%)")
    _set_axis_label(ax, "y", _loss_ylabel(loss_label, loss_name))
    _set_probability_limits(ax, named, log_p=False, full=True)
    if all(int(frontier.gamma) == 0 for frontier in named.values()):
        _pad_y_limits(ax, zero_origin=True)
    else:
        _pad_y_limits(ax, zero_origin=False)
    if handles:
        _place_legend(ax, handles, loc="lower right")
    if annotate:
        _annotate_intervals(ax, named)
    return fig


def plot_conditional_depth(
    fronts: Frontier | Mapping[str, Frontier] | Sequence[Frontier],
    *,
    ax: Axes | None = None,
    log_p: bool | None = None,
    inset: bool = True,
    legend: bool = True,
) -> Figure:
    """Probability–depth image under ``s = loss / probability``.

    A logarithmic probability axis is selected only when the displayed
    upper/lower ratio is at least 12. Dashed curves show the transformed outer
    reconstruction when it is visibly distinct from the inner set.
    """
    named = _as_frontiers(fronts)
    fig, ax = _axes(ax)
    colors = _colors_for(named)
    zoomed, full_max, need_inset = _depth_y_limit(named)
    focus = _depth_focus(named) if need_inset else named
    use_log = _use_log_probability_axis(focus, p0_floor=True) if log_p is None else bool(log_p)
    handles = _draw_depth_sets(ax, named, colors, fill=True, outline_outer=True)
    _style_axes(ax)
    _prob_label, _loss_label, depth_label = _axis_labels(named)
    _set_axis_label(ax, "x", f"{_prob_label} (%)")
    _set_axis_label(ax, "y", depth_label)
    if use_log:
        ax.set_xscale("log")
        _format_log_percent_axis(ax)
        _restrain_ticks(ax)
    _set_probability_limits(ax, focus, log_p=use_log, full=True, p0_floor=True)
    ymin = _depth_y_min(named, zoomed)
    if np.isfinite(zoomed) and zoomed > 0.0:
        ax.set_ylim(ymin, zoomed * DEPTH_SCALE)
    if handles and legend:
        _place_legend(ax, handles, loc="upper right")
    if _probability_floor_active(named):
        floor = max(float(frontier.p0) for frontier in named.values())
        ax.text(
            0.98,
            0.02,
            f"Ratio view: probability ≥ {_pct_compact(floor)}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=MUTED,
            zorder=8,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
                pad=1.2,
            ),
        )
    if inset and need_inset and np.isfinite(full_max):
        inset_ax = ax.inset_axes(list(INSET_BOUNDS))
        inset_ax.set_zorder(5)
        _draw_depth_sets(inset_ax, named, colors, fill=True, outline_outer=False)
        _style_inset(inset_ax)
        inset_log = _use_log_probability_axis(named, p0_floor=True) if log_p is None else bool(log_p)
        if inset_log:
            inset_ax.set_xscale("log")
            _format_log_percent_axis(inset_ax)
        else:
            inset_ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        _restrain_ticks(inset_ax, size=6.5, length=1.8)
        _set_probability_limits(inset_ax, named, log_p=inset_log, full=True, p0_floor=True)
        inset_ax.set_ylim(
            _depth_y_min(named, full_max),
            full_max * DEPTH_ALL_PAD * DEPTH_SCALE,
        )
        inset_ax.set_title("")
        inset_ax.text(
            0.97,
            0.05,
            "Full range",
            transform=inset_ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
            zorder=6,
        )
        inset_ax.set_xlabel("")
        inset_ax.set_ylabel("")
    return fig


def plot_frontier_panels(
    fronts: Frontier | Mapping[str, Frontier] | Sequence[Frontier],
    *,
    title: str | None = None,
) -> Figure:
    """Side-by-side probability–loss and conditional-depth panels."""
    named = _as_frontiers(fronts)
    fig, axes = plt.subplots(1, 2, figsize=PANELS_SIZE, layout="constrained")
    _style_figure(fig)
    _apply_layout_pads(fig, COMPOSITE_PADS)
    _set_panel_letter(axes[0], "(a)")
    _set_panel_letter(axes[1], "(b)")
    plot_frontier(named, ax=axes[0], annotate=False)
    plot_conditional_depth(named, ax=axes[1], legend=False)
    _set_figure_title(fig, title or _default_title(named), named)
    return fig


def plot_quotes(
    cs: CrossSection,
    K,
    *,
    ax: Axes | None = None,
    highlight: DualCertificate | None = None,
    used: pd.DataFrame | None = None,
) -> Figure:
    """Normalized bid–ask call prices against moneyness, with optional binding marks."""
    fig, ax = _axes(ax, figsize=(4.4, 3.2))
    quotes = cs.quotes
    used_ids = None
    if used is not None and len(used):
        used_ids = set(used["quote_id"].astype(str))
    seen_roles: set[str] = set()
    handles: list[Line2D] = []
    xs: list[float] = []
    ys: list[float] = []
    for row in quotes.itertuples(index=False):
        x_value = float(row.x)
        lo = float(row.norm_l)
        hi = float(row.norm_u)
        if not np.isfinite(x_value) or not np.isfinite(lo) or not np.isfinite(hi):
            continue
        role = str(getattr(row, "role", "") or _infer_role(row))
        color = ROLE_COLORS.get(role, ROLE_COLORS["itm"])
        alpha = 1.0
        if used_ids is not None and str(row.quote_id) not in used_ids:
            alpha = UNUSED_ALPHA
        ax.plot(
            [x_value, x_value],
            [lo, hi],
            color=color,
            linewidth=QUOTE_LINEWIDTH,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.scatter(
            [x_value, x_value],
            [lo, hi],
            s=QUOTE_MARKER_SIZE,
            color=color,
            alpha=alpha,
            linewidths=0.0,
            zorder=3,
        )
        if role not in seen_roles:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linewidth=QUOTE_LINEWIDTH,
                    label=ROLE_LABELS.get(role, role),
                )
            )
            seen_roles.add(role)
        xs.append(x_value)
        ys.extend((lo, hi))
    threshold = float(K)
    ax.axvline(
        threshold,
        color=SEPARATOR_COLOR,
        linestyle="--",
        linewidth=THRESHOLD_LINEWIDTH,
        zorder=1,
        label="_nolegend_",
    )
    if xs:
        grid = np.linspace(min(min(xs), threshold), max(max(xs), threshold), 200)
        ax.plot(
            grid,
            np.maximum(1.0 - grid, 0.0),
            color=INTRINSIC_COLOR,
            linewidth=INTRINSIC_LINEWIDTH,
            zorder=1,
        )
        ys.extend(np.maximum(1.0 - grid, 0.0).tolist())
    if highlight is not None and not highlight.quote_positions.empty:
        _mark_binding_quotes(ax, cs, highlight)
    _style_axes(ax)
    _set_axis_label(ax, "x", r"Normalized strike $x$")
    _set_axis_label(ax, "y", r"Normalized call price $c(k)$")
    if xs:
        span = max(max(xs), threshold) - min(min(xs), threshold)
        pad = 0.03 * (span if span > 0.0 else 1.0)
        ax.set_xlim(min(min(xs), threshold) - pad, max(max(xs), threshold) + pad)
    if ys:
        y_hi = max(ys)
        ax.set_ylim(0.0, y_hi * 1.08 if y_hi > 0.0 else 1.0)
    if handles:
        _place_legend(ax, handles, loc="upper right")
    return fig


def plot_binding_quotes(cert: DualCertificate, *, ax: Axes | None = None) -> Figure:
    """Absolute dual weights on binding quotes and structural constraints."""
    fig, ax = _axes(ax, figsize=(4.4, 3.2))
    quotes = cert.quote_positions
    structural = cert.structural
    if not quotes.empty:
        quotes = quotes.loc[quotes["weight"].abs() > WEIGHT_DISPLAY_CUTOFF]
    if not structural.empty:
        structural = structural.loc[structural["weight"].abs() > WEIGHT_DISPLAY_CUTOFF]
    labels: list[str] = []
    weights: list[float] = []
    for row in quotes.itertuples(index=False):
        labels.append(_quote_tick_label(row))
        weights.append(abs(float(row.weight)))
    n_quotes = len(labels)
    for row in structural.itertuples(index=False):
        labels.append(_structural_tick_label(str(row.constraint)))
        weights.append(abs(float(row.weight)))
    positions = np.arange(len(labels), dtype=float)
    finite = [(pos, weight, label) for pos, weight, label in zip(positions, weights, labels) if weight > 0.0]
    if finite:
        values = np.array([item[1] for item in finite], dtype=float)
        ax.set_yscale("log")
        ax.set_ylim(float(np.min(values)) * 0.45, float(np.max(values)) * 5.5)
        for pos, weight, _label in finite:
            ax.vlines(pos, weight * 0.02, weight, colors=INK, linewidth=0.8, zorder=2)
            ax.scatter([pos], [weight], s=16.0, color=INK, zorder=3)
        _annotate_distinct_weights(ax, [(pos, weight) for pos, weight, _label in finite])
    if n_quotes and n_quotes < len(labels):
        ax.axvline(n_quotes - 0.5, color=SEPARATOR_COLOR, linewidth=0.7, zorder=1)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", va="top", fontsize=ANNOTATION_SIZE)
    ax.tick_params(axis="x", labelsize=ANNOTATION_SIZE, pad=2.0)
    _style_axes(ax, grid="y")
    ax.tick_params(axis="x", labelcolor=INK)
    _set_axis_label(ax, "y", r"$|\mathrm{weight}|$")
    ax.set_xlim(-0.6, max(len(labels) - 0.4, 0.6))
    return fig


def plot_report(
    cs: CrossSection,
    fronts: Frontier | Mapping[str, Frontier] | Sequence[Frontier],
    *,
    certificate: DualCertificate | None = None,
    title: str | None = None,
) -> Figure:
    """One-page 2×2 report: quotes, frontier, depth, and certificate or summary text."""
    named = _as_frontiers(fronts)
    fig, axes = plt.subplots(2, 2, figsize=REPORT_SIZE, layout="constrained")
    _style_figure(fig)
    _apply_layout_pads(fig, COMPOSITE_PADS)
    _set_panel_letter(axes[0, 0], "(a)")
    _set_panel_letter(axes[0, 1], "(b)")
    _set_panel_letter(axes[1, 0], "(c)")
    _set_panel_letter(axes[1, 1], "(d)")
    first = next(iter(named.values()))
    largest = _largest_frontier(named)
    plot_quotes(cs, first.K, ax=axes[0, 0], highlight=certificate, used=largest.quotes)
    plot_frontier(named, ax=axes[0, 1], annotate=False)
    plot_conditional_depth(named, ax=axes[1, 0], legend=False)
    if certificate is not None:
        plot_binding_quotes(certificate, ax=axes[1, 1])
    else:
        _draw_summary_panel(axes[1, 1], named)
    _set_panel_letter(axes[0, 0], "(a)")
    _set_panel_letter(axes[0, 1], "(b)")
    _set_panel_letter(axes[1, 0], "(c)")
    _set_panel_letter(axes[1, 1], "(d)")
    _set_figure_title(fig, title or _default_title(named), named)
    return fig


def plot_term_structure(table: pd.DataFrame, *, ax: Axes | None = None) -> Figure:
    """Probability and depth bands against calendar days to expiry."""
    return _interval_panels(
        table,
        x_name="days",
        x_label="Days to expiry",
        ax=ax,
        zero_origin=True,
    )


def plot_threshold_profile(table: pd.DataFrame, *, ax: Axes | None = None) -> Figure:
    """Probability and depth bands against the crash threshold."""
    return _interval_panels(
        table,
        x_name="K",
        x_label=r"Threshold $K$",
        ax=ax,
        zero_origin=False,
    )


def plot_time_series(
    table: pd.DataFrame,
    *,
    value: str = "prob",
    ax: Axes | None = None,
    overlay: pd.Series | None = None,
    overlay_label=None,
) -> Figure:
    """Date-indexed fill band for probability, loss, or depth intervals."""
    if value not in {"prob", "loss", "depth"}:
        raise ValueError("value must be 'prob', 'loss', or 'depth'")
    fig, ax = _axes(ax, figsize=PANELS_SIZE)
    work = table.copy()
    x_values = _time_series_index(work)
    order = np.argsort(np.asarray(x_values), kind="mergesort")
    x_values = x_values[order]
    work = work.iloc[order]
    failed = _error_mask(work)
    value_scale = 1e4 if value == "loss" else PROB_SCALE
    lower = pd.to_numeric(work[f"{value}_lo"], errors="coerce") * value_scale
    upper = pd.to_numeric(work[f"{value}_hi"], errors="coerce") * value_scale
    lower = lower.where(~failed)
    upper = upper.where(~failed)
    finite = np.isfinite(lower.to_numpy(dtype=float)) & np.isfinite(upper.to_numpy(dtype=float))
    _draw_scaled_band(
        ax,
        x_values,
        lower.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        finite=finite,
    )
    _style_axes(ax)
    _set_axis_label(ax, "x", "Date")
    _set_axis_label(ax, "y", _TIME_SERIES_YLABELS[value])
    if overlay is not None:
        right = ax.twinx()
        overlay_x = pd.to_datetime(overlay.index)
        right.plot(
            overlay_x,
            np.asarray(overlay.to_numpy(), dtype=float),
            color=MUTED,
            linewidth=1.0,
            zorder=1,
            solid_capstyle="round",
        )
        right.set_facecolor("none")
        right.spines["top"].set_visible(False)
        right.spines["left"].set_visible(False)
        right.spines["bottom"].set_visible(False)
        right.spines["right"].set_color(SPINE)
        right.spines["right"].set_linewidth(SPINE_WIDTH)
        right.tick_params(
            axis="y",
            labelsize=TICK_SIZE,
            width=TICK_WIDTH,
            length=TICK_LENGTH,
            colors=MUTED,
            labelcolor=MUTED,
        )
        right.grid(False)
        if overlay_label:
            right.set_ylabel(str(overlay_label), fontsize=FONT_SIZE, color=MUTED)
    _format_date_axis(ax)
    return fig


_TIME_SERIES_YLABELS = {
    "prob": "Crash probability (%)",
    "loss": "Expected crash loss (basis points)",
    "depth": "Conditional depth (%)",
}


def _format_date_axis(ax: Axes) -> None:
    locator = AutoDateLocator(minticks=DATE_MINTICKS, maxticks=DATE_MAXTICKS)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))


def _interval_panels(
    table: pd.DataFrame,
    *,
    x_name: str,
    x_label: str,
    ax: Axes | None,
    zero_origin: bool,
) -> Figure:
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=PANELS_SIZE, layout="constrained")
        _style_figure(fig)
        _apply_layout_pads(fig, COMPOSITE_PADS)
    else:
        fig = ax.figure
        _style_figure(fig)
        if len(fig.axes) >= 2:
            axes = (fig.axes[0], fig.axes[1])
        else:
            fig, axes = plt.subplots(1, 2, figsize=PANELS_SIZE, layout="constrained")
            _style_figure(fig)
            _apply_layout_pads(fig, COMPOSITE_PADS)
    ok = table.loc[~_error_mask(table)].copy()
    if not ok.empty and x_name in ok.columns:
        ok = ok.sort_values(x_name, kind="mergesort")
        x_values = pd.to_numeric(ok[x_name], errors="coerce").to_numpy(dtype=float)
        _draw_interval_band(axes[0], x_values, ok["prob_lo"], ok["prob_hi"], midpoint=True)
        _draw_interval_band(axes[1], x_values, ok["depth_lo"], ok["depth_hi"], midpoint=True)
    for axis, ylabel, letter in zip(
        axes,
        ("Crash probability (%)", "Conditional depth (%)"),
        ("(a)", "(b)"),
    ):
        _style_axes(axis)
        _set_axis_label(axis, "x", x_label)
        _set_axis_label(axis, "y", ylabel)
        _set_panel_letter(axis, letter)
        _pad_limits(axis, zero_origin=zero_origin)
    return fig


def _draw_interval_band(ax: Axes, x_values: np.ndarray, lo, hi, *, midpoint: bool) -> None:
    lower = pd.to_numeric(lo, errors="coerce").to_numpy(dtype=float) * PROB_SCALE
    upper = pd.to_numeric(hi, errors="coerce").to_numpy(dtype=float) * PROB_SCALE
    finite = np.isfinite(x_values) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(finite):
        return
    _draw_scaled_band(
        ax,
        x_values,
        lower,
        upper,
        finite=finite,
        midpoint=midpoint,
    )


def _error_mask(table: pd.DataFrame) -> pd.Series:
    if "error" not in table.columns:
        return pd.Series(False, index=table.index)
    text = table["error"].fillna("").astype(str).str.strip()
    return text.ne("") & text.str.lower().ne("nan")


def _time_series_index(table: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" in table.columns:
        stamps = pd.to_datetime(table["date"], errors="coerce")
    else:
        stamps = pd.Series(pd.NaT, index=table.index)
    if "requested_date" in table.columns:
        stamps = stamps.fillna(pd.to_datetime(table["requested_date"], errors="coerce"))
    return pd.DatetimeIndex(stamps)


def _as_frontiers(
    fronts: Frontier | Mapping[str, Frontier] | Sequence[Frontier],
) -> dict[str, Frontier]:
    if isinstance(fronts, Frontier):
        return {str(fronts.quote_set): fronts}
    if isinstance(fronts, Mapping):
        named = {str(name): frontier for name, frontier in fronts.items()}
        if not named:
            raise ValueError("fronts is empty")
        return named
    items = list(fronts)
    if not items:
        raise ValueError("fronts is empty")
    parts = [_key_parts(item) for item in items]
    differ = [any(row[index] != parts[0][index] for row in parts) for index in range(4)]
    if not any(differ):
        differ = [True, False, False, False]
    named: dict[str, Frontier] = {}
    for item, part in zip(items, parts):
        key = " ".join(token for token, keep in zip(part, differ) if keep)
        if key in named:
            suffix = 2
            while f"{key} ({suffix})" in named:
                suffix += 1
            key = f"{key} ({suffix})"
        named[key] = item
    return named


def _key_parts(frontier: Frontier) -> tuple[str, str, str, str]:
    return (
        str(frontier.quote_set),
        f"γ={int(frontier.gamma)}",
        f"K={float(frontier.K):.2f}",
        pd.Timestamp(frontier.date).strftime("%Y-%m-%d"),
    )


def _colors_for(named: Mapping[str, Frontier]) -> dict[str, str]:
    colors: dict[str, str] = {}
    palette_index = 0
    for key in named:
        if key in COLORS:
            colors[key] = COLORS[key]
            continue
        colors[key] = PALETTE[palette_index % len(PALETTE)]
        palette_index += 1
    return colors


def _fill_alpha(index: int, count: int) -> float:
    if count <= 1:
        return FILL_ALPHA_MAX
    position = min(max(index, 0), count - 1) / max(count - 1, 1)
    return float(FILL_ALPHAS[0] + position * (FILL_ALPHA_MAX - FILL_ALPHAS[0]))


def _axis_labels(named: Mapping[str, Frontier]) -> tuple[str, str, str]:
    gammas = {int(frontier.gamma) for frontier in named.values()}
    if len(gammas) != 1:
        return "Crash probability", "Expected crash loss", "Conditional depth (%)"
    gamma = next(iter(gammas))
    if gamma == 0:
        return (
            r"Crash probability $p_K$",
            r"Expected crash loss $L_K$",
            "Conditional depth\n$s_K = L_K/p_K$ (%)",
        )
    return (
        rf"$q_{{{gamma},K}}$",
        rf"$\ell_{{{gamma},K}}$",
        rf"Conditional depth" "\n"
        rf"$s_{{{gamma},K}}=\ell_{{{gamma},K}}/q_{{{gamma},K}}$ (%)",
    )


def _loss_scale(loss_unit: str) -> tuple[float, str]:
    if loss_unit == "bp":
        return 1e4, "bp"
    if loss_unit == "percent":
        return 100.0, "%"
    raise ValueError("loss_unit must be 'bp' or 'percent'")


def _loss_ylabel(base: str, unit_name: str) -> str:
    if unit_name == "bp":
        return f"{base} (basis points)"
    if unit_name == "%":
        return f"{base} (%)"
    return f"{base} ({unit_name})"


def _style_figure(fig: Figure) -> None:
    fig.set_facecolor("white")
    fig.set_edgecolor("none")
    fig.patch.set_linewidth(0.0)


def _set_axis_label(ax: Axes, axis: str, text: str, *, color: str = INK) -> None:
    setter = ax.set_xlabel if axis == "x" else ax.set_ylabel
    setter(text, fontsize=FONT_SIZE, color=color, labelpad=5.0 if axis == "x" else 7.0)


def _apply_layout_pads(fig: Figure, pads: dict[str, float] | None = None) -> None:
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(**(pads or LAYOUT_PADS))


def _axes(ax: Axes | None, *, figsize: tuple[float, float] = DEFAULT_SIZE) -> tuple[Figure, Axes]:
    if ax is not None:
        _style_figure(ax.figure)
        ax.set_facecolor("white")
        return ax.figure, ax
    fig, created = plt.subplots(figsize=figsize, layout="constrained")
    _style_figure(fig)
    created.set_facecolor("white")
    _apply_layout_pads(fig)
    return fig, created


def _style_axes(ax: Axes, *, grid: str = "both") -> None:
    _style_figure(ax.figure)
    ax.set_facecolor("white")
    _restrain_ticks(ax)
    ax.xaxis.label.set_size(FONT_SIZE)
    ax.yaxis.label.set_size(FONT_SIZE)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(SPINE_WIDTH)
    ax.grid(False, which="both")
    ax.set_axisbelow(True)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    if grid == "both":
        ax.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
    elif grid == "y":
        ax.yaxis.grid(True, which="major", color=GRID_COLOR, linewidth=GRID_LINEWIDTH)
        ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _restrain_ticks(ax: Axes, *, size: float = TICK_SIZE, length: float = TICK_LENGTH) -> None:
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=size,
        width=TICK_WIDTH,
        length=length,
        colors=MUTED,
        labelcolor=MUTED,
        pad=2.0,
    )
    if ax.get_xscale() == "log" or ax.get_yscale() == "log":
        ax.tick_params(which="minor", length=max(length * 0.55, 1.2), width=0.4, colors=SPINE)
        ax.grid(False, which="minor")
    else:
        ax.tick_params(which="minor", length=0.0)


def _style_inset(ax: Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(False, which="both")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.55)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=6.5,
        width=0.4,
        length=1.8,
        colors=MUTED,
        labelcolor=MUTED,
        pad=1.0,
    )
    ax.tick_params(which="minor", length=0.0)
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    ax.set_axisbelow(True)


def _set_panel_letter(ax: Axes, letter: str) -> None:
    artist = ax.set_title(
        letter,
        loc="left",
        fontsize=PANEL_TITLE_SIZE,
        pad=1.5,
        color=INK,
        fontweight="medium",
    )
    artist.set_position((-0.035, 1.0))


def _place_legend(ax: Axes, handles, *, loc: str) -> None:
    # Kept for callers. The legend sits above the axes, on the right, so a
    # left-hand panel letter and the data both stay clear of it.
    del loc
    count = len(handles)
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=min(3, count),
        fontsize=TICK_SIZE,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.55,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.25,
        borderpad=0.1,
    )
    for text in legend.get_texts():
        text.set_color(INK)
    legend.set_zorder(6)


def _cloud(points) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    return np.atleast_2d(array).reshape(-1, 2)


def _closed(points: np.ndarray) -> np.ndarray:
    cloud = _cloud(points)
    if len(cloud) <= 1:
        return cloud
    if np.allclose(cloud[0], cloud[-1]):
        return cloud
    return np.vstack([cloud, cloud[0]])


def _scaled(points: np.ndarray, x_scale: float, y_scale: float) -> np.ndarray:
    if len(points) == 0:
        return points
    out = np.array(points, dtype=float, copy=True)
    out[:, 0] *= x_scale
    out[:, 1] *= y_scale
    return out


def _draw_set(
    ax: Axes,
    points: np.ndarray,
    color: str,
    alpha: float,
    label: str,
    *,
    fill: bool,
) -> Patch | Line2D | None:
    raw = _cloud(points)
    if len(raw) == 0:
        return None
    if len(raw) == 1:
        ax.scatter(raw[:, 0], raw[:, 1], color=color, s=HIGHLIGHT_SIZE, zorder=3)
        return Line2D([0], [0], color=color, linewidth=INNER_LINEWIDTH, label=label)
    cloud = _closed(raw)
    unique = len(raw) - 1 if len(raw) >= 2 and np.allclose(raw[0], raw[-1]) else len(raw)
    if unique < 3 or not fill:
        ax.plot(
            cloud[:, 0],
            cloud[:, 1],
            color=color,
            linewidth=INNER_LINEWIDTH,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        return Line2D([0], [0], color=color, linewidth=INNER_LINEWIDTH, label=label)
    rgb = to_rgb(color)
    ax.fill(
        cloud[:, 0],
        cloud[:, 1],
        facecolor=(*rgb, alpha),
        edgecolor="none",
        zorder=2,
    )
    ax.plot(
        cloud[:, 0],
        cloud[:, 1],
        color=color,
        linewidth=INNER_LINEWIDTH,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    return Patch(
        facecolor=(*rgb, min(alpha, 0.22)),
        edgecolor=color,
        linewidth=INNER_LINEWIDTH,
        label=label,
    )


def _draw_outline(ax: Axes, points: np.ndarray, color: str, linewidth: float, *, linestyle: str) -> None:
    cloud = _closed(points)
    if len(cloud) == 0:
        return
    if len(cloud) == 1:
        ax.scatter(cloud[:, 0], cloud[:, 1], color=color, s=12.0, zorder=3)
        return
    ax.plot(
        cloud[:, 0],
        cloud[:, 1],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        dash_capstyle="butt",
        solid_joinstyle="round",
        zorder=2.6,
    )


def _draw_bounds(ax: Axes, frontier: Frontier, color: str, loss_scale: float) -> None:
    lo = np.asarray(frontier.lo, dtype=float).reshape(2)
    hi = np.asarray(frontier.hi, dtype=float).reshape(2)
    width = (hi[0] - lo[0]) * PROB_SCALE
    height = (hi[1] - lo[1]) * loss_scale
    if width <= 0.0 or height <= 0.0:
        return
    ax.add_patch(
        Rectangle(
            (lo[0] * PROB_SCALE, lo[1] * loss_scale),
            width,
            height,
            fill=False,
            linestyle=BOUNDS_LINESTYLE,
            linewidth=0.7,
            edgecolor=color,
            zorder=1,
        )
    )


def _short_label(key: str) -> str:
    if key in SHORT_LABELS:
        return SHORT_LABELS[key]
    return LABELS.get(key, key)


def _annotation_lines(named: Mapping[str, Frontier]) -> list[str]:
    lines: list[str] = []
    for name, frontier in named.items():
        prob_symbol, depth_symbol = _interval_symbols(frontier.gamma)
        lines.append(
            f"{_short_label(name)}: {prob_symbol} "
            f"{_dash_percent(float(frontier.lo[0]), float(frontier.hi[0]))}, "
            f"{depth_symbol} "
            f"{_dash_percent(float(frontier.depth_lo), float(frontier.depth_hi))}"
        )
    return lines


def _interval_symbols(gamma: int) -> tuple[str, str]:
    if int(gamma) == 0:
        return "p", "s"
    subscript = {1: "₁", 2: "₂"}.get(int(gamma), str(int(gamma)))
    return f"q{subscript}", f"s{subscript}"


def _caption_intervals(named: Mapping[str, Frontier]) -> str:
    parts = _annotation_lines(named)
    lines: list[str] = []
    current = ""
    for part in parts:
        trial = part if not current else f"{current}    {part}"
        if current and len(trial) > 88:
            lines.append(current)
            current = part
        else:
            current = trial
    if current:
        lines.append(current)
    return "\n".join(lines)


def _font_height_frac(fig: Figure, fontsize: float, *, lines: int = 1, linespacing: float = 1.35) -> float:
    height_in = max(float(fig.get_size_inches()[1]), 0.1)
    return lines * (fontsize / 72.0) * linespacing / height_in


def _set_figure_title(fig: Figure, title: str, named: Mapping[str, Frontier]) -> None:
    """Place the title above a reserved band so it stays clear of legends."""
    caption = _caption_intervals(named)
    height_in = max(float(fig.get_size_inches()[1]), 0.1)
    caption_lines = 0 if not caption else caption.count("\n") + 1
    top_in = (TITLE_SIZE / 72.0) * 1.35 + caption_lines * (CAPTION_SIZE / 72.0) * 1.45 + 0.06
    top_frac = min(0.40, top_in / height_in)
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(rect=(0.0, 0.0, 1.0, 1.0 - top_frac))
    fig.suptitle(
        title,
        fontsize=TITLE_SIZE,
        color=INK,
        fontweight="medium",
        ha="left",
        va="top",
        x=0.012,
        y=0.995,
    )
    if not caption:
        return
    fig.text(
        0.012,
        0.995 - _font_height_frac(fig, TITLE_SIZE, linespacing=1.25),
        caption,
        fontsize=CAPTION_SIZE,
        color=MUTED,
        ha="left",
        va="top",
        linespacing=1.35,
        transform=fig.transFigure,
    )


def _annotate_intervals(ax: Axes, named: Mapping[str, Frontier]) -> None:
    ax.text(
        0.98,
        0.98,
        "\n".join(_annotation_lines(named)),
        transform=ax.transAxes,
        ha="right",
        va="top",
        multialignment="left",
        fontsize=ANNOTATION_SIZE,
        color=INK,
        linespacing=1.2,
        zorder=6,
        clip_on=True,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor=ANNOTATION_EDGE,
            linewidth=0.4,
            alpha=0.94,
        ),
    )


def _dash_percent(lo: float, hi: float) -> str:
    return f"{_pct_compact(lo)}–{_pct_compact(hi)}%"


def _pct_compact(value: float) -> str:
    scaled = PROB_SCALE * float(value)
    if not np.isfinite(scaled):
        return "nan"
    if abs(scaled) >= 10.0:
        return f"{scaled:.1f}"
    if abs(scaled) >= 1.0:
        return f"{scaled:.2f}"
    return f"{scaled:.3g}"


def _depth_points(frontier: Frontier, *, outer: bool = False) -> np.ndarray:
    points = frontier.outer if outer else frontier.inner
    boundary = _cloud(_sample_depth_boundary(points, frontier.p0))
    if len(boundary) == 0:
        return boundary
    return _scaled(boundary, PROB_SCALE, DEPTH_SCALE)


def _depth_curve(frontier: Frontier, *, outer: bool) -> np.ndarray:
    points = frontier.outer if outer else frontier.inner
    return _cloud(_sample_depth_boundary(points, float(frontier.p0)))


def _depth_curve_values(frontier: Frontier) -> np.ndarray:
    samples = [_depth_curve(frontier, outer=False), _depth_curve(frontier, outer=True)]
    finite: list[np.ndarray] = []
    for boundary in samples:
        if len(boundary) == 0:
            continue
        mask = np.isfinite(boundary[:, 1])
        if np.any(mask):
            finite.append(boundary[mask, 1])
    if not finite:
        return np.empty(0, dtype=float)
    return np.concatenate(finite)


def _depth_max(frontier: Frontier) -> float:
    values = _depth_curve_values(frontier)
    if len(values) == 0:
        return float("nan")
    return float(np.max(values))


def _depth_min(frontier: Frontier) -> float:
    values = _depth_curve_values(frontier)
    if len(values) == 0:
        return 0.0
    return float(np.min(values))


def _depth_y_limit(named: Mapping[str, Frontier]) -> tuple[float, float, bool]:
    maxima = [_depth_max(frontier) for frontier in named.values()]
    finite = [value for value in maxima if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan"), False
    all_max = max(finite)
    last = next(reversed(named.values()))
    last_max = _depth_max(last)
    if not np.isfinite(last_max) or last_max <= 0.0:
        last_max = all_max
    zoomed = min(all_max * DEPTH_ALL_PAD, last_max * DEPTH_LAST_PAD)
    if zoomed <= 0.0:
        zoomed = all_max * DEPTH_ALL_PAD
    need_inset = bool(all_max > zoomed * DEPTH_INSET_RATIO)
    return zoomed, all_max, need_inset


def _depth_focus(named: Mapping[str, Frontier]) -> dict[str, Frontier]:
    items = list(named.items())
    return dict(items[-2:]) if len(items) > 1 else dict(items)


def _depth_y_min(named: Mapping[str, Frontier], ceiling: float) -> float:
    minima = [_depth_min(frontier) for frontier in named.values()]
    finite = [value for value in minima if np.isfinite(value)]
    if not finite:
        return 0.0
    low = min(finite) * DEPTH_SCALE
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        return min(0.0, low)
    if low <= 0.0:
        return 0.0
    return max(0.0, low * 0.92)


def _draw_depth_sets(
    ax: Axes,
    named: Mapping[str, Frontier],
    colors: Mapping[str, str],
    *,
    fill: bool,
    outline_outer: bool,
) -> list[Patch | Line2D]:
    handles: list[Patch | Line2D] = []
    for index, (name, frontier) in enumerate(named.items()):
        handle = _draw_set(
            ax,
            _depth_points(frontier),
            colors[name],
            _fill_alpha(index, len(named)),
            LABELS.get(name, name),
            fill=fill,
        )
        if handle is not None:
            handles.append(handle)
        if outline_outer and _outer_is_distinct(frontier):
            _draw_outline(
                ax,
                _depth_points(frontier, outer=True),
                colors[name],
                OUTER_LINEWIDTH,
                linestyle=OUTER_LINESTYLE,
            )
    return handles


def _sample_depth_boundary(points: np.ndarray, p0: float) -> np.ndarray:
    from crashfrontier.geometry.depth import depth_boundary

    return depth_boundary(points, samples_per_edge=160, p0=p0)


def _outer_is_distinct(frontier: Frontier) -> bool:
    inner_area = float(frontier.area)
    outer_area = float(frontier.area_outer)
    if not np.isfinite(inner_area) or not np.isfinite(outer_area):
        return False
    scale = max(abs(inner_area), abs(outer_area), 1e-12)
    return outer_area - inner_area > max(1e-12, 0.002 * scale)


def _probability_extent(
    named: Mapping[str, Frontier], *, outer: bool, p0_floor: bool = False
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for frontier in named.values():
        if p0_floor:
            cloud = _depth_curve(frontier, outer=outer)
            if len(cloud):
                finite = cloud[:, 0][np.isfinite(cloud[:, 0])]
                if len(finite):
                    values.append(finite * PROB_SCALE)
            continue
        cloud = _cloud(frontier.outer if outer else frontier.inner)
        if len(cloud):
            finite = cloud[:, 0][np.isfinite(cloud[:, 0]) & (cloud[:, 0] > 0.0)]
            if len(finite):
                values.append(finite * PROB_SCALE)
    if not values:
        return float("nan"), float("nan")
    joined = np.concatenate(values)
    return float(np.min(joined)), float(np.max(joined))


def _probability_floor_active(named: Mapping[str, Frontier]) -> bool:
    for frontier in named.values():
        cloud = _cloud(frontier.inner)
        if len(cloud) and np.nanmin(cloud[:, 0]) < float(frontier.p0):
            return True
    return False


def _use_log_probability_axis(named: Mapping[str, Frontier], *, p0_floor: bool = False) -> bool:
    lower, upper = _probability_extent(named, outer=True, p0_floor=p0_floor)
    return bool(
        np.isfinite(lower)
        and np.isfinite(upper)
        and lower > 0.0
        and upper / lower >= 12.0
    )


def _set_probability_limits(
    ax: Axes,
    named: Mapping[str, Frontier],
    *,
    log_p: bool,
    full: bool = False,
    p0_floor: bool = False,
) -> None:
    lower, upper = _probability_extent(named, outer=full, p0_floor=p0_floor)
    if not np.isfinite(lower) or not np.isfinite(upper):
        _pad_limits(ax, zero_origin=not log_p)
        return
    span = max(upper - lower, upper * 0.02, 1e-9)
    if log_p:
        ratio = upper / lower
        factor = max(ratio ** 0.035, 1.04)
        ax.set_xlim(lower / factor, upper * factor)
        return
    left = 0.0 if lower <= 0.25 * upper else max(0.0, lower - 0.08 * span)
    ax.set_xlim(left, upper + 0.08 * span)


def _format_log_percent_axis(ax: Axes) -> None:
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _pad_limits(ax: Axes, *, zero_origin: bool, pad_y: bool = True) -> None:
    ax.margins(x=0.04, y=0.06 if pad_y else 0.0)
    if not zero_origin:
        return
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()
    ax.set_xlim(min(0.0, left), right)
    if pad_y:
        ax.set_ylim(min(0.0, bottom), top)


def _pad_y_limits(ax: Axes, *, zero_origin: bool) -> None:
    data_bottom, data_top = (float(value) for value in ax.dataLim.intervaly)
    if not np.isfinite(data_bottom) or not np.isfinite(data_top):
        return
    span = max(data_top - data_bottom, abs(data_top) * 0.02, 1e-9)
    use_zero = zero_origin and data_bottom <= 0.45 * max(data_top, 1e-12)
    lower = 0.0 if use_zero else max(0.0, data_bottom - 0.10 * span)
    ax.set_ylim(lower, data_top + 0.08 * span)


def _default_title(named: Mapping[str, Frontier]) -> str:
    first = next(iter(named.values()))
    date = pd.Timestamp(first.date).strftime("%Y-%m-%d")
    expiry = pd.Timestamp(first.expiry).strftime("%Y-%m-%d")
    return f"{date} → {expiry} · {int(first.days)} days · K = {float(first.K):.2f}"


def _largest_frontier(named: Mapping[str, Frontier]) -> Frontier:
    def rank(item: tuple[str, Frontier]) -> tuple[int, int]:
        name, frontier = item
        order = SET_ORDER.index(name) if name in SET_ORDER else -1
        return (int(frontier.n_quotes), order)

    return max(named.items(), key=rank)[1]


def _infer_role(row) -> str:
    flag = str(getattr(row, "cp_flag", "")).upper()
    x_value = float(getattr(row, "x", float("nan")))
    if flag == "P" and x_value <= 1.0:
        return "otm_put"
    if flag == "C" and x_value >= 1.0:
        return "otm_call"
    return "itm"


def _mark_binding_quotes(ax: Axes, cs: CrossSection, cert: DualCertificate) -> None:
    by_id = {str(row.quote_id): row for row in cs.quotes.itertuples(index=False)}
    forward = float(cs.forward)
    for row in cert.quote_positions.itertuples(index=False):
        if abs(float(row.weight)) <= WEIGHT_DISPLAY_CUTOFF:
            continue
        quote = by_id.get(str(row.quote_id))
        if quote is not None:
            x_value = float(quote.x)
        elif np.isfinite(forward) and forward != 0.0:
            x_value = float(row.strike) / forward
        else:
            continue
        y_value = float(row.norm_price)
        if not np.isfinite(x_value) or not np.isfinite(y_value):
            continue
        ax.scatter(
            [x_value],
            [y_value],
            s=HIGHLIGHT_SIZE,
            color=INK,
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            clip_on=True,
        )
        ax.annotate(
            str(row.side),
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=ANNOTATION_SIZE,
            color=INK,
            clip_on=True,
            zorder=7,
        )


def _structural_tick_label(name: str) -> str:
    if name.startswith("cell_") and "_moment_" in name:
        body = name[len("cell_") :]
        index, _, side = body.partition("_moment_")
        if index and side:
            return f"cell {index} {side}"
    if name == "threshold_atom_mass":
        return "atom mass"
    if name.startswith("threshold_atom"):
        return "atom residual"
    return name.replace("_", " ")


def _quote_tick_label(row) -> str:
    flag = str(row.cp_flag).upper()
    kind = "Put" if flag.startswith("P") else "Call" if flag.startswith("C") else "Quote"
    return f"{kind} {_format_strike(float(row.strike))} · {row.side} {float(row.price):.2f}"


def _format_strike(strike: float) -> str:
    if not np.isfinite(strike):
        return "nan"
    if abs(strike) >= 10.0:
        return f"{strike:,.0f}"
    return f"{strike:g}"


def _compact(value: float) -> str:
    return f"{value:.3g}"


def _draw_summary_panel(ax: Axes, named: Mapping[str, Frontier]) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.text(
        0.04,
        0.95,
        _summary_text(named),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        color=INK,
        family="sans-serif",
        linespacing=1.35,
        wrap=False,
        clip_on=False,
    )


def _summary_text(named: Mapping[str, Frontier]) -> str:
    first = next(iter(named.values()))
    summary0 = first.summary()
    measure = (
        "Risk-neutral measure"
        if int(first.gamma) == 0
        else f"Physical measure · γ={int(first.gamma)}"
    )
    support = (
        "unbounded"
        if first.support_cap is None
        else f"[0, {float(first.support_cap):g}]"
    )
    lines = [
        measure,
        (
            f"{_fmt_field(summary0['date'])} → {_fmt_field(summary0['expiry'])} "
            f"· {_fmt_field(summary0['days'])} days"
        ),
        f"Threshold K={float(summary0['K']):.2f} · Support {support}",
        "Loss in basis points",
        "",
    ]
    for name, frontier in named.items():
        summary = frontier.summary()
        prob_symbol, depth_symbol = _interval_symbols(frontier.gamma)
        subscript = {1: "₁", 2: "₂"}.get(int(frontier.gamma), str(int(frontier.gamma)))
        loss_symbol = "L" if int(frontier.gamma) == 0 else f"ℓ{subscript}"
        lines.append(LABELS.get(name, name))
        lines.append(
            f"  {int(summary['n_quotes'])} quotes · "
            f"{int(summary['n_strikes'])} strikes · "
            f"{_status_label(summary['status'])}"
        )
        lines.append(
            f"  {prob_symbol}  {_dash_percent(float(summary['prob_lo']), float(summary['prob_hi']))}"
            f"    {depth_symbol}  "
            f"{_dash_percent(float(summary['depth_lo']), float(summary['depth_hi']))}"
        )
        lines.append(
            f"  {loss_symbol}  "
            f"{_bp_interval(float(summary['loss_lo']), float(summary['loss_hi']))}"
            f"    reconstruction gap  {_fmt_sci(float(summary['max_gap']))}"
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _status_label(value) -> str:
    text = str(value).replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else "Unknown"


def _fmt_sci(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.3g}"


def _bp_interval(lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return "[nan, nan]"
    return f"[{1e4 * lo:.3g}, {1e4 * hi:.3g}]"


def _annotate_distinct_weights(ax: Axes, items: list[tuple[float, float]]) -> None:
    """Label each displayed weight once, on the middle mark of that value."""
    grouped: dict[str, list[tuple[float, float]]] = {}
    for pos, weight in items:
        grouped.setdefault(_compact(weight), []).append((pos, weight))
    for text, group in grouped.items():
        pos, weight = group[len(group) // 2]
        ax.annotate(
            text,
            (pos, weight),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            color=MUTED,
            clip_on=True,
        )


def _draw_scaled_band(
    ax: Axes,
    x_values,
    lower,
    upper,
    *,
    finite: np.ndarray | None = None,
    midpoint: bool = True,
) -> None:
    lower_values = np.asarray(lower, dtype=float)
    upper_values = np.asarray(upper, dtype=float)
    if finite is None:
        finite = np.isfinite(lower_values) & np.isfinite(upper_values)
    if not np.any(finite):
        return
    color = SERIES_COLOR
    ax.fill_between(
        x_values,
        lower_values,
        upper_values,
        where=finite,
        interpolate=False,
        color=color,
        alpha=BAND_FILL_ALPHA,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        np.asarray(x_values)[finite],
        lower_values[finite],
        color=color,
        linewidth=BAND_EDGE_WIDTH,
        alpha=BAND_EDGE_ALPHA,
        solid_capstyle="round",
        zorder=3,
    )
    ax.plot(
        np.asarray(x_values)[finite],
        upper_values[finite],
        color=color,
        linewidth=BAND_EDGE_WIDTH,
        alpha=BAND_EDGE_ALPHA,
        solid_capstyle="round",
        zorder=3,
    )
    if midpoint:
        ax.plot(
            np.asarray(x_values)[finite],
            0.5 * (lower_values[finite] + upper_values[finite]),
            color=color,
            linewidth=MIDLINE_WIDTH,
            marker="o",
            markersize=2.6,
            markeredgewidth=0.0,
            solid_capstyle="round",
            zorder=4,
        )


def _fmt_field(value) -> str:
    if value is None:
        return "none"
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return "nan"
        return f"{number:.6g}"
    return str(value)
