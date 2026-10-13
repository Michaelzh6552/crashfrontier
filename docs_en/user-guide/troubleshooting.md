# Troubleshooting

Work in this order: **load data → check the cross-section → read solver status → interpret the result**. The script below reproduces a missing date, an inapplicable quote set, and an infeasible physical cross-section.

## Diagnostic script

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `troubleshooting.py` and run `python troubleshooting.py`.

```python
import os
from pathlib import Path

import numpy as np

from crashfrontier import Dataset, QuoteFilter, compute_frontier, nested_frontiers
from crashfrontier.errors import (
    InfeasibleCrossSectionError,
    NoEligibleExpiryError,
    NoSuchDateError,
    SchemaError,
)


def heading(text: str) -> None:
    print("\n====", text, "====")


def main() -> None:
    DATA_ROOT = Path(os.environ["CRASHFRONTIER_DATA"])

    heading("0. confirm data is loaded")
    ds = Dataset.from_directory(DATA_ROOT)
    info = ds.describe()
    print({k: info[k] for k in ("date_format", "strike_scale", "forwards", "rates")})
    print("n_dates", info["options"]["n_dates"], "files", info["options"]["files"])

    heading("1. Dataset(root) does not enter RAW_DATA")
    try:
        Dataset(DATA_ROOT)
    except SchemaError as exc:
        print(type(exc).__name__, exc)

    heading("2. date does not exist")
    print("nearest 2017-07-04 ->", ds.nearest_date("2017-07-04"))
    try:
        ds.option_rows("2017-07-04")
    except NoSuchDateError as exc:
        print(type(exc).__name__, exc)

    heading("3. no eligible expiry")
    try:
        ds.cross_section("2017-07-05", days=2)
    except NoEligibleExpiryError as exc:
        print(str(exc).splitlines()[0])
        print("n_put_strikes for days=2 in expiries:",
              int(ds.expiries("2017-07-05").query("days == 2")["n_put_strikes"].iloc[0]))

    heading("4. correct cross-section check")
    cs = ds.cross_section("2017-07-05", target_days=14, window=(7, 21))
    print(cs.summary())
    print("filter_report", cs.filter_report)

    heading("5. gamma=0 cannot use full_otm")
    try:
        compute_frontier(cs, 0.90, quote_set="full_otm")
    except ValueError as exc:
        print(exc)

    heading("6. 2020 default physical full_otm + B=5 is infeasible")
    cs30 = ds.cross_section("2020-03-25", days=30, am_settlement=0)
    try:
        compute_frontier(cs30, 0.90, gamma=2)
    except InfeasibleCrossSectionError as exc:
        print(exc)
        print("attrs", exc.quote_set, exc.gamma, exc.support_cap, exc.status)

    heading("7. same day put_wing + B=5 is feasible")
    ok = compute_frontier(cs30, 0.90, gamma=2, quote_set="put_wing", support_cap=5.0)
    print(ok.describe())
    print("depth finite?", np.isfinite(ok.depth_lo), np.isfinite(ok.depth_hi))

    heading("8. physical nested default includes full_otm and fails entirely")
    try:
        nested_frontiers(cs30, 0.90, gamma=2, support_cap=5.0)
    except InfeasibleCrossSectionError as exc:
        print(exc)

    heading("9. compare SOCP and reconstructed depth")
    pair = compute_frontier(cs30, 0.90, gamma=2, quote_set="put_pair", support_cap=5.0)
    print(pair.describe())
    print("inner vertices", len(pair.inner), "prob interval", pair.lo[0], pair.hi[0])
    print("solver depth", pair.diagnostics["solver_depth_lo"],
          pair.diagnostics["solver_depth_hi"])
    print("inner depth", pair.diagnostics["inner_depth_lo"],
          pair.diagnostics["inner_depth_hi"])
    print("reconciled?", pair.diagnostics["depth_reconciled"])

    heading("10. filter comparison")
    loose = Dataset.from_directory(DATA_ROOT, quote_filter=QuoteFilter.permissive())
    print("default kept", cs.filter_report["kept"],
          "permissive kept",
          loose.cross_section("2017-07-05", days=14).filter_report["kept"])


if __name__ == "__main__":
    main()
```

Step 1 raises `SchemaError` when option tables live in a `RAW_DATA/` subdirectory (the constructor does not enter it). A flat directory of option files can succeed but still omits forwards, rates, and spot.

## Diagnose by symptom

### Choose the check from the symptom

| Symptom | First step |
|---|---|
| `KeyError: 'CRASHFRONTIER_DATA'` | Export the path to your quote directory |
| `ModuleNotFoundError` | Use the interpreter where you ran `python -m pip install crashfrontier`; see [install.md](../getting-started/install.md) |
| `no option tables in directory ...` | Switch to `Dataset.from_directory` so an optional `RAW_DATA/` folder is scanned |
| `NoSuchDateError` | `available_dates()` / `nearest_date`; a failed string parse is a different problem |
| Date exists, no eligible expiry | Print `expiries` and inspect `days`, `forward`, `n_put_strikes`, AM/PM |
| Strikes look 1000× too large | Inspect `describe()["strike_scale"]`; leave \(K\) alone |
| `mode=relaxed` | Hard no-arbitrage is infeasible; record the slack. Treat it as a different result from ordinary `hard` |
| `unknown quote set 'full_otm'` | You are at \(\gamma=0\) |
| `InfeasibleCrossSectionError` and `quote_set=full_otm` | Model and quote-set outcome; this guide uses `put_wing` |
| `nested_frontiers failed at quote_set='full_otm'` | Pass `sets` explicitly |
| `depth` is NaN and the figure is still there | Joint frontier is usable; depth endpoints are uncertified |
| Time-series band breaks | Read `error` in the CSV first |
| `SolverFailure` | The solver returned no usable support; that is distinct from model infeasibility |

Exceptions imported from `crashfrontier.errors`: `SchemaError`, `NoSuchDateError`, `NoEligibleExpiryError`, `MissingForwardError`, `InfeasibleCrossSectionError`, `SolverFailure`. All subclass `CrashFrontierError`.

### Loading

The date index for a large CSV is `{cache_dir}/{stem}.index.json`. A stored source path that does not match the file, or a change in size or mtime, rebuilds it. Several files with the same stem opened together get a short path hash in the name. If you omit `cache_dir`, the default is a `CACHE/` folder two levels above the first option file. If `Dataset.describe()` runs, the index is usually usable.

`cross_section(..., forward=F)` cannot revive an expiry whose forward is NaN. A rate series with no observation on or before the quote date raises `ValueError` (the message mentions `rates=0.0`). `rates=None` is explicit \(r=0\).

### Model infeasibility versus solver failure

`InfeasibleCrossSectionError` carries `date`, `expiry`, `quote_set`, `gamma`, `support_cap`, and `status`. At \(\gamma>0\) the message mentions default `full_otm` and retrying `put_wing` or `support_cap=None`.

`SolverFailure` is often `support failed for direction ...`. Physical programs default to CLARABEL; inexact optima then try ECOS and SCS if installed: `python -m pip install 'crashfrontier[solvers]'`. Do not change the quote set to get past the solver. `status` equal to `infeasible` means there is no measure; an uncertified inexact optimum should be treated as a solver problem first.

### Depth NaN

At \(\gamma=0\), depth is \(L/p\) on inner vertices. At \(\gamma>0\) a
second cone program is solved and cross-checked against the reconstructed
inner polygon. A small discrepancy widens the public interval and sets
`diagnostics["depth_reconciled"]`; an uncertified program, a negligible
probability upper bound, or a value outside \([0,K]\) can still produce NaN.
That means “conditional-depth endpoints unavailable,” not “conditional depth
infeasible.”

### Where files are written

Relative paths are resolved against the current directory from which you ran `python`. `save_figure` / `write_frontier` take an extension-free stem and create parent directories. Print `path.resolve()` for the absolute path.

The script writes no figures. Steps 0, 4, 7, and 9 should succeed; steps 2,
3, 5, 6, and 8 should print the matching exception. Step 1 depends on whether
options sit in `RAW_DATA/`. Step 7 \(\gamma=2\) put wing prints
`status=converged` and 85 quotes.

## Options

When troubleshooting, prefer **leaving parameters unchanged**. For a controlled comparison:

| Comparison | Call |
|---|---|
| Relax volume (changes the sample) | `QuoteFilter.permissive()` |
| Same specification, extra solver fallbacks | Install `crashfrontier[solvers]`; leave `quote_set` alone |
| Physical, avoid `full_otm` | `quote_set="put_wing"` and pass `sets` when nesting |
| All expiries | `print(ds.expiries(date).to_string(index=False))` |

## Common errors

- **Catching `Exception` and continuing to plot.** You lose the date, quote set, and `support_cap`.
- **Changing \(K\), the filter, or \(B\) in order to produce a figure.** Confirm the cross-section first.
- **Reading only the exception string.** Read the type first, then `exc.quote_set`, `exc.gamma`, `exc.support_cap`, `exc.status`.
- **Calling nonexistent `analyze`, `result.plot`, `Dataset.from_files`, or `Dataset.summary`.** Real entry points are in the scripts on each page.
