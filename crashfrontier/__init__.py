from __future__ import annotations

from crashfrontier.config import ConicSettings, GridSettings, LPSettings
from crashfrontier.data.cross_section import CrossSection
from crashfrontier.data.dataset import Dataset
from crashfrontier.data.filters import QuoteFilter
from crashfrontier.data.schema import ColumnMap
from crashfrontier.export import write_frontier
from crashfrontier.frontier import (
    Frontier,
    binding_quotes,
    compute_frontier,
    nested_frontiers,
    portfolio_bound,
    scenario_check,
)
from crashfrontier.plotting import (
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
from crashfrontier.series import frontier_series, term_structure, threshold_profile

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "CrossSection",
    "QuoteFilter",
    "ColumnMap",
    "Frontier",
    "compute_frontier",
    "nested_frontiers",
    "scenario_check",
    "portfolio_bound",
    "binding_quotes",
    "term_structure",
    "threshold_profile",
    "frontier_series",
    "plot_frontier",
    "plot_conditional_depth",
    "plot_frontier_panels",
    "plot_quotes",
    "plot_binding_quotes",
    "plot_report",
    "plot_term_structure",
    "plot_threshold_profile",
    "plot_time_series",
    "save_figure",
    "write_frontier",
    "LPSettings",
    "ConicSettings",
    "GridSettings",
    "__version__",
]
