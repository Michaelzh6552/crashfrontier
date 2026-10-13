from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crashfrontier.config import COLORS, PALETTE
from crashfrontier.frontier import binding_quotes, nested_frontiers
from crashfrontier.plotting import (
    ROLE_COLORS,
    plot_binding_quotes,
    plot_conditional_depth,
    plot_frontier,
    plot_frontier_panels,
    plot_quotes,
    plot_report,
    plot_term_structure,
    plot_threshold_profile,
    plot_time_series,
    save_figure,
)
from crashfrontier.riskneutral.certificates import DualCertificate
from tests.test_frontier import THRESHOLD, _quotes, _section


@pytest.fixture(scope="module")
def bundle():
    quotes = _quotes()
    section = _section(quotes)
    nested = nested_frontiers(section, THRESHOLD)
    wing = nested["put_wing"]
    certificate = binding_quotes(section, THRESHOLD, quote_set="put_wing")
    return section, nested, wing, certificate


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _nonempty(path: Path) -> None:
    assert path.is_file()
    assert path.stat().st_size > 0


def test_plot_functions_return_figure_and_savefig(bundle, tmp_path: Path) -> None:
    section, nested, wing, certificate = bundle
    figures = {
        "frontier": plot_frontier(nested),
        "depth": plot_conditional_depth(nested),
        "panels": plot_frontier_panels(nested),
        "quotes": plot_quotes(section, THRESHOLD, used=wing.quotes, highlight=certificate),
        "binding": plot_binding_quotes(certificate),
        "report": plot_report(section, nested, certificate=certificate),
        "term": plot_term_structure(_band_table()),
        "threshold": plot_threshold_profile(_band_table()),
        "series": plot_time_series(_band_table(), overlay=_overlay_series()),
    }
    for name, fig in figures.items():
        assert isinstance(fig, Figure)
        path = tmp_path / f"{name}.png"
        fig.savefig(path)
        _nonempty(path)


def test_save_figure_writes_pdf_and_png(bundle, tmp_path: Path) -> None:
    _section, nested, _wing, _certificate = bundle
    fig = plot_frontier(nested)
    written = save_figure(fig, tmp_path / "stem", formats=("pdf", "png"), dpi=120)
    names = {path.name for path in written}
    assert names == {"stem.pdf", "stem.png"}
    for path in written:
        _nonempty(path)


def test_fronts_accept_single_dict_and_sequence(bundle) -> None:
    _section, nested, wing, _certificate = bundle
    single = plot_frontier(wing)
    as_dict = plot_frontier(nested)
    as_list = plot_frontier(list(nested.values()))
    for fig in (single, as_dict, as_list):
        assert isinstance(fig, Figure)
        assert fig.axes
    labels = [text.get_text() for text in as_dict.axes[0].get_legend().get_texts()]
    assert "Put pair" in labels
    assert "Local eight" in labels
    assert "Complete put wing" in labels
    sequence_labels = [text.get_text() for text in as_list.axes[0].get_legend().get_texts()]
    assert sequence_labels == labels


def test_sequence_keys_keep_parts_that_differ(bundle) -> None:
    _section, _nested, wing, _certificate = bundle
    other = replace(wing, gamma=2)
    fig = plot_frontier([wing, other])
    labels = [text.get_text() for text in fig.axes[0].get_legend().get_texts()]
    assert labels == ["γ=0", "γ=2"]
    assert fig.axes[0].get_xlabel() == "Crash probability (%)"


def test_gamma0_axis_labels(bundle) -> None:
    _section, nested, _wing, _certificate = bundle
    fig = plot_frontier(nested)
    assert "$p_K$" in fig.axes[0].get_xlabel()
    assert "$L_K$" in fig.axes[0].get_ylabel()
    depth = plot_conditional_depth(nested)
    assert "s_K" in depth.axes[0].get_ylabel()


def test_gamma_positive_axis_labels(bundle) -> None:
    _section, _nested, wing, _certificate = bundle
    physical = replace(wing, gamma=2)
    fig = plot_frontier(physical)
    assert fig.axes[0].get_xlabel() == r"$q_{2,K}$ (%)"
    assert r"$\ell_{2,K}$" in fig.axes[0].get_ylabel()
    annotation = "\n".join(text.get_text() for text in fig.axes[0].texts)
    assert "q₂ " in annotation
    assert "p " not in annotation
    depth = plot_conditional_depth(physical)
    assert r"$s_{2,K}=" in depth.axes[0].get_ylabel()
    assert r"\ell_{2,K}/q_{2,K}" in depth.axes[0].get_ylabel()
    assert depth.axes[0].get_xscale() == "linear"


def test_plot_frontier_bounds_annotate_and_units(bundle) -> None:
    _section, nested, _wing, _certificate = bundle
    fig = plot_frontier(nested, show_bounds=True, annotate=True, loss_unit="percent")
    ax = fig.axes[0]
    assert ax.get_ylabel().endswith("(%)")
    assert any("Pair:" in text.get_text() for text in ax.texts)
    assert any("p " in text.get_text() and "–" in text.get_text() for text in ax.texts)
    assert not any("Complete put wing: p ∈" in text.get_text() for text in ax.texts)
    assert not any("p ∈" in text.get_text() for text in ax.texts)
    assert len(ax.patches) >= 1
    with pytest.raises(ValueError, match="loss_unit"):
        plot_frontier(nested, loss_unit="raw")


def test_empty_fronts_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_frontier({})
    with pytest.raises(ValueError, match="empty"):
        plot_frontier([])


def test_conditional_depth_axis_respects_p0(bundle) -> None:
    _section, _nested, wing, _certificate = bundle
    cloud = np.array(
        [[0.01, 0.002], [0.40, 0.002], [0.40, 0.05], [0.01, 0.02]],
        dtype=float,
    )
    floored = replace(wing, inner=cloud, outer=cloud, p0=0.20)
    fig = plot_conditional_depth(floored, inset=False)
    lower, _upper = fig.axes[0].get_xlim()
    assert lower > 10.0
    plt.close(fig)


def test_conditional_depth_inset_when_last_set_is_much_smaller(bundle) -> None:
    _section, nested, _wing, _certificate = bundle
    pair = nested["put_pair"]
    wing = nested["put_wing"]
    scaled_inner = np.asarray(wing.inner, dtype=float).copy()
    scaled_inner[:, 1] *= 0.04
    tiny = replace(
        wing,
        inner=scaled_inner,
        outer=scaled_inner,
        depth_lo=float(wing.depth_lo) * 0.04,
        depth_hi=float(wing.depth_hi) * 0.04,
    )
    fig = plot_conditional_depth({"put_pair": pair, "put_wing": tiny}, inset=True)
    assert len(fig.axes[0].child_axes) == 1
    off = plot_conditional_depth({"put_pair": pair, "put_wing": tiny}, inset=False)
    assert len(off.axes[0].child_axes) == 0


def test_plot_frontier_panels_size_and_title(bundle) -> None:
    _section, nested, _wing, _certificate = bundle
    fig = plot_frontier_panels(nested)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(7.2)
    assert height == pytest.approx(3.1)
    assert "2021-07-21" in fig.get_suptitle()
    assert "(a)" in fig.axes[0].get_title(loc="left")
    assert "(b)" in fig.axes[1].get_title(loc="left")


def test_plot_report_with_certificate(bundle) -> None:
    section, nested, _wing, certificate = bundle
    fig = plot_report(section, nested, certificate=certificate)
    width, height = fig.get_size_inches()
    assert width == pytest.approx(9.2)
    assert height == pytest.approx(7.0)
    assert len(fig.axes) >= 4
    letters = [ax.get_title(loc="left") for ax in fig.axes[:4]]
    assert letters == ["(a)", "(b)", "(c)", "(d)"]
    assert fig.axes[3].yaxis.get_scale() == "log"


def test_plot_report_text_panel_without_certificate(bundle) -> None:
    section, nested, _wing, _certificate = bundle
    fig = plot_report(section, nested, certificate=None)
    panel = fig.axes[3]
    blob = "\n".join(text.get_text() for text in panel.texts)
    assert "quotes" in blob
    assert "Put pair" in blob
    assert "Complete put wing" in blob
    assert "p  " in blob
    assert not panel.axison


def test_plot_quotes_highlights_binding_side_labels(bundle) -> None:
    section, _nested, wing, certificate = bundle
    fig = plot_quotes(section, THRESHOLD, highlight=certificate, used=wing.quotes)
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert "bid" in labels or "ask" in labels


def test_plot_binding_quotes_has_quote_tick_labels(bundle) -> None:
    _section, _nested, _wing, certificate = bundle
    fig = plot_binding_quotes(certificate)
    ticks = [text.get_text() for text in fig.axes[0].get_xticklabels()]
    assert ticks
    assert any("bid" in tick or "ask" in tick for tick in ticks)


def _axes_text_blob(fig: Figure) -> str:
    return "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)


def _figure_text_blob(fig: Figure) -> str:
    parts = [fig.get_suptitle() or ""]
    parts.extend(text.get_text() for text in fig.texts)
    return "\n".join(parts)


def test_annotation_text_is_compact_and_inside_axes(bundle) -> None:
    _section, nested, _wing, _certificate = bundle
    fig = plot_frontier(nested, annotate=True)
    ax = fig.axes[0]
    blob = "\n".join(text.get_text() for text in ax.texts)
    assert "Pair:" in blob
    assert "Eight:" in blob
    assert "Put wing:" in blob
    assert "Complete put wing: p ∈" not in blob
    assert "p ∈" not in blob
    for text in ax.texts:
        if text.get_transform() != ax.transAxes:
            continue
        x_pos, y_pos = text.get_position()
        assert 0.0 <= x_pos <= 1.0
        assert 0.0 <= y_pos <= 1.0
        assert text.get_clip_on()
        assert text.get_fontsize() <= 7.5
        assert text.get_ha() == "right"
        assert text.get_va() == "top"


def test_panels_and_report_keep_interval_caption_off_axes(bundle) -> None:
    section, nested, _wing, certificate = bundle
    panels = plot_frontier_panels(nested)
    report = plot_report(section, nested, certificate=certificate)
    nocert = plot_report(section, nested, certificate=None)
    for fig in (panels, report, nocert):
        ax_blob = _axes_text_blob(fig)
        assert "Complete put wing: p ∈" not in ax_blob
        fig_blob = _figure_text_blob(fig)
        assert "Pair:" in fig_blob
        assert "Put wing:" in fig_blob
        for ax in fig.axes:
            for text in ax.texts:
                if text.get_transform() != ax.transAxes:
                    continue
                x_pos, y_pos = text.get_position()
                assert 0.0 <= x_pos <= 1.0
                assert 0.0 <= y_pos <= 1.0


def test_plot_quotes_segments_are_visible(bundle) -> None:
    section, _nested, _wing, _certificate = bundle
    fig = plot_quotes(section, THRESHOLD)
    ax = fig.axes[0]
    quote_lines = [line for line in ax.lines if line.get_linewidth() >= 1.2]
    intrinsic = [line for line in ax.lines if 0.0 < line.get_linewidth() < 0.55]
    assert quote_lines
    assert intrinsic
    assert any(len(coll.get_offsets()) >= 2 for coll in ax.collections)


def test_binding_structural_labels_are_short(bundle) -> None:
    _section, _nested, _wing, certificate = bundle
    fig = plot_binding_quotes(certificate)
    ticks = [text.get_text() for text in fig.axes[0].get_xticklabels()]
    assert not any("moment_upper" in tick or "moment_lower" in tick for tick in ticks)
    if (not certificate.structural.empty) and bool(
        certificate.structural["weight"].abs().gt(1e-8).any()
    ):
        assert any(tick.startswith("cell ") or tick.startswith("atom ") for tick in ticks)


def _band_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-07-21", "2021-07-22", "2021-07-28"]),
            "requested_date": pd.to_datetime(["2021-07-21", "2021-07-22", "2021-07-28"]),
            "days": [7, 14, 30],
            "K": [0.80, 0.90, 0.95],
            "prob_lo": [0.01, 0.02, 0.03],
            "prob_hi": [0.04, 0.05, 0.06],
            "loss_lo": [0.001, 0.002, 0.003],
            "loss_hi": [0.004, 0.005, 0.006],
            "depth_lo": [0.10, 0.12, 0.14],
            "depth_hi": [0.20, 0.22, 0.24],
            "error": ["", "failed", ""],
        }
    )


def _overlay_series() -> pd.Series:
    return pd.Series(
        [4300.0, 4310.0, 4290.0],
        index=pd.to_datetime(["2021-07-21", "2021-07-22", "2021-07-28"]),
    )


def test_term_threshold_and_time_series_plots(tmp_path: Path) -> None:
    table = _band_table()
    term = plot_term_structure(table)
    assert isinstance(term, Figure)
    assert len(term.axes) == 2
    assert "(a)" in term.axes[0].get_title(loc="left")
    threshold = plot_threshold_profile(table)
    assert isinstance(threshold, Figure)
    assert "$K$" in threshold.axes[0].get_xlabel()
    series = plot_time_series(table, value="prob", overlay=_overlay_series(), overlay_label="SPX")
    assert isinstance(series, Figure)
    assert len(series.axes) == 2
    with pytest.raises(ValueError, match="value"):
        plot_time_series(table, value="area")
    for name, fig in ("term", term), ("threshold", threshold), ("series", series):
        path = tmp_path / f"{name}.png"
        fig.savefig(path)
        _nonempty(path)


def test_time_series_date_axis_locator_and_formatter() -> None:
    short = plot_time_series(_band_table(), value="prob")
    long_table = _band_table().copy()
    long_table["date"] = pd.to_datetime(["2013-01-02", "2019-06-15", "2025-08-29"])
    long_table["requested_date"] = long_table["date"]
    long_fig = plot_time_series(long_table, value="depth")
    boundary = _band_table().copy()
    boundary["date"] = pd.to_datetime(["2017-01-25", "2017-01-29", "2017-02-01"])
    boundary["requested_date"] = boundary["date"]
    month_edge = plot_time_series(boundary, value="loss", overlay=_overlay_series())
    assert "basis points" in month_edge.axes[0].get_ylabel()
    for fig in (short, long_fig, month_edge):
        axis = fig.axes[0]
        locator = axis.xaxis.get_major_locator()
        formatter = axis.xaxis.get_major_formatter()
        assert isinstance(locator, AutoDateLocator)
        assert isinstance(formatter, ConciseDateFormatter)
        assert locator.minticks == 3
        assert set(locator.maxticks.values()) == {7}
        fig.canvas.draw()
        labels = [tick for tick in axis.get_xticklabels() if tick.get_text() and tick.get_visible()]
        assert labels
        assert all(abs(tick.get_rotation()) < 45.0 for tick in labels)
        boxes = [tick.get_window_extent(renderer=fig.canvas.get_renderer()) for tick in labels]
        for left, right in zip(boxes, boxes[1:]):
            assert left.x1 <= right.x0 + 0.5


def test_term_and_threshold_keep_numeric_date_free_axes() -> None:
    table = _band_table()
    term = plot_term_structure(table)
    threshold = plot_threshold_profile(table)
    for fig in (term, threshold):
        for axis in fig.axes[:2]:
            assert not isinstance(axis.xaxis.get_major_locator(), AutoDateLocator)
            assert not isinstance(axis.xaxis.get_major_formatter(), ConciseDateFormatter)


def _hex(color: object) -> str:
    return to_hex(color).lower()


def _solid_data_lines(ax):
    return [
        line
        for line in ax.lines
        if line.get_linestyle() in ("-", "solid") and line.get_linewidth() >= 1.2
    ]


def _is_short_dash(line) -> bool:
    pattern = getattr(line, "_unscaled_dash_pattern", None)
    if not pattern or pattern[1] is None:
        return False
    dashes = pattern[1]
    return bool(dashes) and max(dashes[0::2]) <= 2.2 and min(dashes[1::2]) > 0


def _known_polygon() -> np.ndarray:
    return np.array(
        [[0.02, 0.0002], [0.10, 0.0002], [0.10, 0.0010], [0.02, 0.0008]],
        dtype=float,
    )


def _line_extrema(ax, *, minimum_width: float = 1.2) -> tuple[float, float, float, float]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for line in ax.lines:
        if line.get_linewidth() < minimum_width:
            continue
        xs.append(np.asarray(line.get_xdata(), dtype=float))
        ys.append(np.asarray(line.get_ydata(), dtype=float))
    assert xs and ys
    x_values = np.concatenate(xs)
    y_values = np.concatenate(ys)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    assert np.any(finite)
    return (
        float(np.min(x_values[finite])),
        float(np.max(x_values[finite])),
        float(np.min(y_values[finite])),
        float(np.max(y_values[finite])),
    )


def test_public_plot_signatures_keep_names_defaults_and_returns() -> None:
    def params(func):
        return inspect.signature(func).parameters

    frontier = params(plot_frontier)
    assert list(frontier) == [
        "fronts",
        "ax",
        "fill",
        "outline_outer",
        "show_bounds",
        "loss_unit",
        "annotate",
    ]
    assert frontier["fronts"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("ax", "fill", "outline_outer", "show_bounds", "loss_unit", "annotate"):
        assert frontier[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert frontier["ax"].default is None
    assert frontier["fill"].default is True
    assert frontier["outline_outer"].default is True
    assert frontier["show_bounds"].default is False
    assert frontier["loss_unit"].default == "bp"
    assert frontier["annotate"].default is True
    assert inspect.signature(plot_frontier).return_annotation == "Figure"

    depth = params(plot_conditional_depth)
    assert list(depth) == ["fronts", "ax", "log_p", "inset", "legend"]
    assert depth["log_p"].default is None
    assert depth["inset"].default is True
    assert depth["legend"].default is True
    assert inspect.signature(plot_conditional_depth).return_annotation == "Figure"

    panels = params(plot_frontier_panels)
    assert list(panels) == ["fronts", "title"]
    assert panels["title"].default is None
    assert inspect.signature(plot_frontier_panels).return_annotation == "Figure"

    quotes = params(plot_quotes)
    assert list(quotes) == ["cs", "K", "ax", "highlight", "used"]
    assert quotes["K"].default is inspect.Parameter.empty
    assert quotes["highlight"].default is None
    assert quotes["used"].default is None
    assert inspect.signature(plot_quotes).return_annotation == "Figure"

    binding = params(plot_binding_quotes)
    assert list(binding) == ["cert", "ax"]
    assert binding["ax"].default is None
    assert inspect.signature(plot_binding_quotes).return_annotation == "Figure"

    report = params(plot_report)
    assert list(report) == ["cs", "fronts", "certificate", "title"]
    assert report["certificate"].default is None
    assert report["title"].default is None
    assert inspect.signature(plot_report).return_annotation == "Figure"

    assert list(params(plot_term_structure)) == ["table", "ax"]
    assert list(params(plot_threshold_profile)) == ["table", "ax"]
    series = params(plot_time_series)
    assert list(series) == ["table", "value", "ax", "overlay", "overlay_label"]
    assert series["value"].default == "prob"
    assert series["overlay"].default is None
    assert series["overlay_label"].default is None
    assert inspect.signature(plot_time_series).return_annotation == "Figure"

    saved = params(save_figure)
    assert list(saved) == ["fig", "path_stem", "formats", "dpi"]
    assert saved["formats"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert saved["formats"].default == ("pdf", "png")
    assert saved["dpi"].default == 300
    assert inspect.signature(save_figure).return_annotation == "list[Path]"


def test_default_loss_axes_spell_out_basis_points(bundle) -> None:
    _section, nested, wing, _certificate = bundle
    frontier = plot_frontier(nested)
    ylabel = frontier.axes[0].get_ylabel()
    assert "basis points" in ylabel
    assert "$L_K$" in ylabel
    assert not ylabel.endswith("(bp)")
    percent = plot_frontier(nested, loss_unit="percent")
    percent_label = percent.axes[0].get_ylabel()
    assert percent_label.endswith("(%)")
    assert "basis points" not in percent_label
    assert "1 bp = 0.01%" not in percent_label
    physical = plot_frontier(replace(wing, gamma=2))
    assert r"$\ell_{2,K}$" in physical.axes[0].get_ylabel()
    assert "basis points" in physical.axes[0].get_ylabel()
    series = plot_time_series(_band_table(), value="loss")
    series_label = series.axes[0].get_ylabel()
    assert series_label.startswith("Expected crash loss")
    assert "basis points" in series_label


def test_displayed_coordinates_keep_probability_and_loss_scales(bundle) -> None:
    _section, _nested, wing, _certificate = bundle
    cloud = _known_polygon()
    styled = replace(wing, inner=cloud, outer=cloud.copy(), area=1.0, area_outer=1.0, gamma=0)
    basis = plot_frontier(styled, annotate=False, outline_outer=False)
    x_lo, x_hi, y_lo, y_hi = _line_extrema(basis.axes[0])
    assert x_lo == pytest.approx(2.0)
    assert x_hi == pytest.approx(10.0)
    assert y_lo == pytest.approx(2.0)
    assert y_hi == pytest.approx(10.0)
    percent = plot_frontier(styled, annotate=False, outline_outer=False, loss_unit="percent")
    _x_lo, _x_hi, y_lo, y_hi = _line_extrema(percent.axes[0])
    assert y_lo == pytest.approx(0.02)
    assert y_hi == pytest.approx(0.10)
    series = plot_time_series(_band_table(), value="loss")
    _x_lo, _x_hi, y_lo, y_hi = _line_extrema(series.axes[0], minimum_width=0.5)
    assert y_lo == pytest.approx(10.0)
    assert y_hi == pytest.approx(60.0)
    probability = plot_time_series(_band_table(), value="prob")
    _x_lo, _x_hi, y_lo, y_hi = _line_extrema(probability.axes[0], minimum_width=0.5)
    assert y_lo == pytest.approx(1.0)
    assert y_hi == pytest.approx(6.0)


def test_legend_sits_above_axes_and_not_on_data(bundle) -> None:
    section, nested, wing, certificate = bundle
    figures = (
        plot_frontier(nested),
        plot_conditional_depth(nested, inset=False),
        plot_quotes(section, THRESHOLD, used=wing.quotes, highlight=certificate),
        plot_frontier_panels(nested),
    )
    checked = 0
    for fig in figures:
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer)
            axes_box = ax.get_window_extent(renderer)
            assert legend_box.y0 >= axes_box.y1 - 1.5
            for artist in (*ax.patches, *ax.lines, *ax.collections):
                box = artist.get_window_extent(renderer)
                if box.width <= 0.0 or box.height <= 0.0:
                    continue
                if not np.isfinite([box.x0, box.y0, box.x1, box.y1]).all():
                    continue
                shrunk = legend_box.expanded(-1.0, -1.0)
                assert not shrunk.overlaps(box)
            checked += 1
    assert checked >= 3


def test_inner_outer_color_and_linestyle(bundle) -> None:
    _section, nested, wing, _certificate = bundle
    fig = plot_frontier(nested, annotate=False)
    solid = _solid_data_lines(fig.axes[0])
    assert [_hex(line.get_color()) for line in solid] == [_hex(COLORS[name]) for name in nested]
    assert all(1.2 <= line.get_linewidth() <= 1.4 for line in solid)
    dashed = [line for line in fig.axes[0].lines if _is_short_dash(line)]
    distinct = []
    for name, frontier in nested.items():
        scale = max(abs(float(frontier.area)), abs(float(frontier.area_outer)), 1e-12)
        if float(frontier.area_outer) - float(frontier.area) > max(1e-12, 0.002 * scale):
            distinct.append(name)
    assert [_hex(line.get_color()) for line in dashed] == [_hex(COLORS[name]) for name in distinct]
    identical = replace(wing, area=1.0, area_outer=1.0)
    quiet = plot_frontier(identical, annotate=False)
    assert not any(_is_short_dash(line) for line in quiet.axes[0].lines)
    widened = replace(wing, area=1.0, area_outer=1.05)
    marked = plot_frontier(widened, annotate=False)
    assert any(_is_short_dash(line) for line in marked.axes[0].lines)
    unknown = plot_frontier({"custom-layer": wing}, annotate=False, outline_outer=False)
    unknown_lines = _solid_data_lines(unknown.axes[0])
    assert len(unknown_lines) == 1
    assert _hex(unknown_lines[0].get_color()) == PALETTE[0].lower()
    assert _hex(unknown_lines[0].get_color()) not in {_hex(color) for color in ROLE_COLORS.values()}


def test_quote_role_colors_differ_from_information_sets(bundle) -> None:
    section, _nested, _wing, _certificate = bundle
    assert set(map(_hex, COLORS.values())).isdisjoint(map(_hex, ROLE_COLORS.values()))
    assert set(map(_hex, PALETTE)).isdisjoint(map(_hex, ROLE_COLORS.values()))
    fig = plot_quotes(section, THRESHOLD)
    role_lines = [line for line in fig.axes[0].lines if line.get_linewidth() >= 1.2]
    assert role_lines
    assert {_hex(line.get_color()) for line in role_lines} <= {_hex(color) for color in ROLE_COLORS.values()}
    assert {_hex(line.get_color()) for line in role_lines}.isdisjoint({_hex(color) for color in COLORS.values()})


def test_interval_band_is_lighter_than_its_midline() -> None:
    fig = plot_term_structure(_band_table())
    ax = fig.axes[0]
    assert ax.collections
    assert ax.collections[0].get_alpha() <= 0.14
    widths = [line.get_linewidth() for line in ax.lines]
    assert max(widths) >= 1.2
    assert max(widths) > min(widths) + 0.4
    assert _hex(ax.spines["left"].get_edgecolor()) == "#cbd3dc"
    fig.canvas.draw()
    gridlines = [line for line in ax.xaxis.get_gridlines() if line.get_visible()]
    assert gridlines
    assert _hex(gridlines[0].get_color()) == "#e8edf2"
    assert gridlines[0].get_linewidth() <= 0.5


def test_binding_labels_do_not_repeat_the_same_weight(bundle) -> None:
    _section, _nested, _wing, certificate = bundle
    live = plot_binding_quotes(certificate)
    live_labels = [text.get_text() for text in live.axes[0].texts]
    assert len(live_labels) == len(set(live_labels))
    repeated = DualCertificate(
        direction=np.array([1.0, 0.0]),
        support_value=0.0,
        upper_value=0.0,
        duality_gap=0.0,
        point=np.array([0.1, 0.001]),
        primal_dual_gap=0.0,
        max_stationarity_residual=0.0,
        max_complementarity_residual=0.0,
        cash=0.0,
        forward=0.0,
        quote_positions=pd.DataFrame(
            {
                "quote_id": [str(index) for index in range(9)],
                "cp_flag": ["P"] * 9,
                "strike": [1000.0 + index for index in range(9)],
                "side": ["bid"] * 9,
                "weight": [0.012345] * 9,
                "price": [1.25] * 9,
                "norm_price": [0.1] * 9,
            }
        ),
        structural=pd.DataFrame({"constraint": ["threshold_atom_mass"], "weight": [0.2]}),
        threshold_atom=0.0,
    )
    fig = plot_binding_quotes(repeated)
    labels = [text.get_text() for text in fig.axes[0].texts]
    assert labels.count(f"{0.012345:.3g}") == 1
    assert labels.count(f"{0.2:.3g}") == 1
    assert len(labels) == 2


def test_plotting_does_not_mutate_global_rcparams(bundle) -> None:
    import matplotlib as mpl

    section, nested, _wing, certificate = bundle
    before = {key: mpl.rcParams[key] for key in mpl.rcParams}
    plot_frontier(nested)
    plot_report(section, nested, certificate=certificate)
    plot_time_series(_band_table(), value="loss", overlay=_overlay_series())
    after = {key: mpl.rcParams[key] for key in mpl.rcParams}
    changed = [key for key in before if before[key] != after[key]]
    assert changed == []
    assert _hex(plot_frontier(nested).axes[0].get_facecolor()) == "#ffffff"
