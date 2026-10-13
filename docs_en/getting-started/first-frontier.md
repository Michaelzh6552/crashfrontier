# First crash frontier

This example runs after `pip install crashfrontier`. It creates a small
arbitrage-free option cross-section in memory, so it does not require
OptionMetrics, a repository checkout, or a `DATA/` directory.

The synthetic distribution is used only to produce internally consistent put
quotes. Replace it with your own market data after checking the workflow.

## Complete script

Save as `first_frontier.py` in any directory:

```python
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crashfrontier import (
    CrossSection,
    compute_frontier,
    plot_frontier_panels,
    save_figure,
    write_frontier,
)

# A discrete risk-neutral distribution for X = S_T / F.
atoms = np.array([0.60, 0.75, 0.88, 1.00, 1.12, 1.25, 1.40])
mass = np.array([0.05, 0.10, 0.15, 0.40, 0.15, 0.10, 0.05])
assert np.isclose(mass.sum(), 1.0)
assert np.isclose(float(mass @ atoms), 1.0)

# Put mids E[(k-X)+], surrounded by positive bid-ask spreads.
strikes = np.round(np.linspace(0.71, 1.05, 18), 4)
mid = np.array(
    [float(np.sum(mass * np.maximum(k - atoms, 0.0))) for k in strikes]
)
half_spread = 0.001 + 0.04 * mid

quotes = pd.DataFrame(
    {
        "strike": strikes,
        "cp_flag": "P",
        "bid": np.maximum(mid - half_spread, 0.0),
        "ask": mid + half_spread,
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

output = Path("outputs/first_frontier")
figure = plot_frontier_panels(frontier)
written = save_figure(figure, output)
plt.close(figure)
written.extend(write_frontier(frontier, output))

for path in written:
    print("wrote", path)
```

Run:

```bash
python first_frontier.py
```

The script creates `outputs/` relative to the directory where it is run.

Expected terminal output:

```text
2024-01-17 → 2024-01-31 (14d)  K=0.90  γ=0
quote_set=put_wing  n_quotes=18  n_strikes=18
F=1 (user)  D=1  support_cap=none
mode=hard  method=hull  status=edges_certified  dimension=2
prob=[0.1402, 0.373]  loss=[0.02983, 0.03532]  depth=[0.0799732, 0.212767]
area=0.00349424  area_outer=0.00349424  area_ratio=0.650453  interior_ok=True  interior_margin=0.5
wrote outputs/first_frontier.pdf
wrote outputs/first_frontier.png
wrote outputs/first_frontier_summary.csv
wrote outputs/first_frontier_vertices.csv
wrote outputs/first_frontier_halfplanes.csv
wrote outputs/first_frontier_quotes.csv
wrote outputs/first_frontier.json
```

## Generated figure

<figure class="cf-example-figure">
  <img src="../assets/images/examples/00_synthetic_frontier.png" alt="Synthetic risk-neutral crash frontier and conditional-depth panels" loading="eager">
  <figcaption>
    Output from the complete script above. It uses synthetic quotes generated
    in memory; no market data is bundled with the package.
  </figcaption>
</figure>

The left panel is the joint set of crash probability and expected loss. The
right panel applies the conditional-depth transformation \(s=L/p\). Solid
fill is the inner approximation; the dashed outline is the certified outer
set.

## What each step does

### 1. Build a `CrossSection`

`CrossSection.from_quotes` is the shortest entry point when one observation
date, one expiry, and the forward are already known. The table must contain at
least:

```text
strike, cp_flag, bid, ask
```

`date` and `expiry` are supplied as arguments. `volume` is included because
the default filter retains positive-volume quotes.

With real OptionMetrics `strike_price`, pass `strike_scale=1000`. With ordinary
index-point or normalized strikes, keep `strike_scale=1`.

### 2. Compute the frontier

```python
frontier = compute_frontier(section, K=0.90)
```

The defaults are:

- `gamma=0`: risk-neutral measure
- `quote_set="put_wing"`
- unbounded right support
- hard bid-ask constraints, with minimum-slack fallback only if necessary
- adaptive hull reconstruction

The returned `Frontier` contains the marginal bounds, inner and outer
polygons, conditional-depth interval, diagnostics, and the quotes that entered
the program.

### 3. Plot and export

`save_figure` writes PNG and PDF by default. `write_frontier` writes:

- `_summary.csv`
- `_vertices.csv`
- `_halfplanes.csv`
- `_quotes.csv`
- `.json`

Pass a path stem, not a `.png` filename.

## Move to market data

To load a quote directory you already hold:

```python
from crashfrontier import Dataset

dataset = Dataset.from_directory("/path/to/market-data")
dates = dataset.available_dates()
print(dataset.expiries(dates[-1]))
section = dataset.cross_section(dates[-1], target_days=30)
```

See [Market data](data.md) before loading a vendor extract. Column formats and
aliases are in the [data guide](../data-guide/index.md).

## Next

- [Read the figure](read-the-chart.md)
- [Risk-neutral model](../user-guide/risk-neutral.md)
- [Physical \(\gamma=1,2\)](../user-guide/physical.md)
- [More examples](../examples/index.md)
