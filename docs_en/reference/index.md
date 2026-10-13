# Reference

Lookup for public Python names, result schemas, exceptions, solver settings,
and CLI flags. Behavior matches `crashfrontier` `0.1.0`.

| Page | Use when you need |
|---|---|
| [Python API](api.md) | Signatures, defaults, return types, and exceptions for every name in `crashfrontier.__all__` |
| [Results and schemas](results.md) | `Frontier`, `DualCertificate`, tables, units, statuses, JSON/CSV |
| [Exceptions](errors.md) | Hierarchy, attributes, import paths, and error-to-action |
| [Configuration](configuration.md) | `LPSettings`, `GridSettings`, `ConicSettings`, and packaged defaults |
| [Command-line appendix](cli.md) | `crashfrontier` subcommands and flags |
| [Papers](papers.md) | arXiv citations for the risk-neutral and physical papers |
| [Data guide](../data-guide/index.md) | File formats, column aliases, forwards, rates, filters |

## Package-root names

`import crashfrontier` re-exports exactly `crashfrontier.__all__`.

| Group | Names | Details |
|---|---|---|
| Version | `__version__` | [API](api.md#version) |
| Data | `Dataset`, `CrossSection`, `QuoteFilter`, `ColumnMap` | [API](api.md#data) |
| Compute | `compute_frontier`, `nested_frontiers`, `scenario_check`, `portfolio_bound`, `binding_quotes`, `Frontier` | [API](api.md#compute), [results](results.md) |
| Batch | `term_structure`, `threshold_profile`, `frontier_series` | [API](api.md#series) |
| Plot / export | `plot_*`, `save_figure`, `write_frontier` | [API](api.md#plotting), [results](results.md#export) |
| Settings | `LPSettings`, `GridSettings`, `ConicSettings` | [configuration](configuration.md) |

Names that are **not** in `__all__` are not package-root public names:

```python
from crashfrontier.errors import (
    CrashFrontierError,
    InfeasibleCrossSectionError,
    MissingForwardError,
    NoEligibleExpiryError,
    NoSuchDateError,
    SchemaError,
    SolverFailure,
)
from crashfrontier.riskneutral import DualCertificate
```

`pick_dates` lives on `crashfrontier.series`. `frontier_frames` lives on
`crashfrontier.export`.

## Units

`Frontier.summary()`, `scenario_check`, and exported CSV/JSON store
probability, loss, and depth as **decimals relative to the forward**.
Figures scale probability and depth to percent and loss to basis points (bp)
by default. 1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%.
See [Results and schemas](results.md#units).

## Typical call order

```python
from crashfrontier import Dataset, compute_frontier

ds = Dataset.from_directory("/path/to/market-data")
date = ds.available_dates()[-1]
cs = ds.cross_section(date, target_days=30)
frontier = compute_frontier(cs, K=0.90)
print(frontier.describe())
```
