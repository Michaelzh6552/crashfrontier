# crashfrontier

`crashfrontier` computes joint bounds on crash probability, expected crash
loss, and conditional depth from option bid-ask quotes.

- `gamma=0`: risk-neutral frontier via linear programming
- `gamma=1` or `gamma=2`: physical-measure frontier via second-order cone
  programming

Python 3.10 or later is required.

## Install

```bash
python -m pip install crashfrontier
```

The base install reads CSV and pandas tables and solves with CLARABEL.
Install only the extras required by your workflow:

```bash
python -m pip install "crashfrontier[parquet]"  # PyArrow
python -m pip install "crashfrontier[excel]"    # XLSX, XLS, XLSB engines
python -m pip install "crashfrontier[solvers]"  # ECOS and SCS fallbacks
```

For development, run this in the source checkout containing `pyproject.toml`:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

This self-contained example constructs synthetic put quotes in memory:

```python
import numpy as np
import pandas as pd

from crashfrontier import CrossSection, compute_frontier

atoms = np.array([0.60, 0.75, 0.88, 1.00, 1.12, 1.25, 1.40])
mass = np.array([0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05])
strikes = np.round(np.linspace(0.71, 1.05, 18), 4)
mid = np.array(
    [np.sum(mass * np.maximum(strike - atoms, 0.0)) for strike in strikes]
)
spread = 0.001 + 0.04 * mid
quotes = pd.DataFrame(
    {
        "strike": strikes,
        "cp_flag": "P",
        "bid": np.maximum(mid - spread, 0.0),
        "ask": mid + spread,
        "volume": 10,
    }
)

section = CrossSection.from_quotes(
    quotes,
    date="2024-01-17",
    expiry="2024-01-31",
    forward=1.0,
    rate=0.0,
)
frontier = compute_frontier(section, K=0.90)
print(frontier.describe())
```

Use `plot_frontier_panels`, `save_figure`, and `write_frontier` to export
figures, CSV tables, and strict JSON.

## Market data

The distribution contains no market data and does not download data. Supply
your own licensed vendor extract or another option quote source. A standard
option table needs `date`, `expiry`, `cp_flag`, `strike`, `bid`, and `ask`;
common vendor aliases are recognized.

The papers use licensed OptionMetrics SPX data. Obtain vendor data under your
own institutional, WRDS, or direct OptionMetrics agreement.

```python
from crashfrontier import Dataset

dataset = Dataset.from_directory("/path/to/market-data")
date = dataset.available_dates()[-1]
section = dataset.cross_section(date, target_days=30)
```

Supported inputs include pandas objects, CSV/TXT/TSV, compressed text,
Parquet, Excel, SAS, and Stata files. Parquet and Excel require their
respective extras.

## Principal API

```python
from crashfrontier import (
    Dataset,
    CrossSection,
    QuoteFilter,
    ColumnMap,
    Frontier,
    compute_frontier,
    nested_frontiers,
    scenario_check,
    portfolio_bound,
    binding_quotes,
    term_structure,
    threshold_profile,
    frontier_series,
    plot_frontier_panels,
    save_figure,
    write_frontier,
    LPSettings,
    ConicSettings,
    GridSettings,
)
```

The installed command-line entry point is `crashfrontier`; run
`crashfrontier --help` for its subcommands. The Python API is the primary
interface.

## Papers

- Jirong Zhuang,
  [*The Risk-Neutral Crash Frontier: Sharp Joint Bounds on Crash Probability and Conditional Depth from Option Bid-Ask Quotes*](https://arxiv.org/abs/2607.25353),
  arXiv:2607.25353.
- Jirong Zhuang,
  [*The Physical Crash Frontier: What Finite Option Quotes Can and Cannot Reveal*](https://arxiv.org/abs/2608.23274),
  arXiv:2608.23274.

## Documentation

In a source checkout, documentation lives in `docs_en/`. The API and CLI
references are `docs_en/reference/api.md` and `docs_en/reference/cli.md`;
troubleshooting is `docs_en/user-guide/troubleshooting.md`. The prebuilt
offline site is `site_en/index.html`.

Build the site from the repository root:

```bash
python -m pip install -r docs_en/requirements.txt
python -m mkdocs build --strict
```

## License

MIT. The distribution includes the full license text.
