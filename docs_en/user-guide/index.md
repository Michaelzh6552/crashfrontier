# Analysis guide

Choose the page by the result you want to produce.

## Core frontiers

- [Select a cross-section](select-cross-section.md): inspect dates, expiries,
  tenors, and AM/PM settlement before solving.
- [Risk-neutral](risk-neutral.md): compute the \(\gamma=0\) LP frontier.
- [Physical](physical.md): compute the \(\gamma=1\) or \(\gamma=2\) SOCP
  frontier.
- [Quote sets](quote-sets.md): compare `put_pair`, `local_eight`,
  `put_wing`, and `full_otm`.

## Diagnostics and reporting

- [Report](reports.md): assemble quotes, frontiers, depth, and certificates.
- [Scenario](scenario.md): test whether one probability-loss pair remains
  admissible.
- [Export](results-export.md): write CSV and JSON and interpret their units.
- [Troubleshooting](troubleshooting.md): distinguish data, feasibility, and
  solver failures.

## Profiles and series

- [Term structure](term-structure.md): compare expiry tenors on one date.
- [Threshold profile](threshold-profile.md): vary the crash threshold \(K\).
- [Time series](time-series.md): repeat the frontier over observation dates.

## Output convention

The first figure is fully synthetic and reproducible from its listing.
Historical SPX tables and figures in the analysis pages are sample outputs.
Their purpose is to show schemas, units, and chart
interpretation; run the same code on your own cross-sections for empirical
results.

Data preparation and vendor fields are covered in [Market data](../getting-started/data.md).
