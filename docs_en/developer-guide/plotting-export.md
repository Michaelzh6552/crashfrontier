# Plot units and export

Modules: `crashfrontier/plotting.py`, `export.py`. Plot sizes, units, labels,
and palettes are defined in the package source.

## Units (must be kept distinct)

| Quantity | Tables / `summary()` / `scenario` | `plot_frontier` | `plot_conditional_depth` | `plot_time_series` |
|---|---|---|---|---|
| Probability | $[0,1]$ decimal | ×100, axis label `(%)` | ×100, axis label `(%)` | ×100, `Crash probability (%)` |
| Loss | decimal relative to the forward | default ×$10^4$ basis points; `loss_unit="percent"` → ×100 | loss is not drawn | ×$10^4$ basis points |
| Depth | $[0,1]$ decimal | written as percent in annotations | ×100, axis label includes `(%)` | ×100 |

1 basis point = 0.01 percentage points = 0.0001; 100 bp = 1%.
The parameter name stays `loss_unit="bp"`.

`scenario_check` `prob`/`loss` must be decimals, e.g. `0.02` and `0.004`, not
2% and 4bp. CLI `scenario` is the same.

`loss_unit` accepts only `"bp"` | `"percent"`; anything else → `ValueError`.
Scales are unchanged: `"bp"` is ×$10^4$, `"percent"` is ×100.

## Axis labels

`_axis_labels`: when every frontier has the same γ,

- γ=0: `Crash probability $p_K$` / `Expected crash loss $L_K$ (basis points)` /
  `Conditional depth\n$s_K = L_K/p_K$ (%)`
- γ=1 or 2: `$q_{γ,K}$` / `$\ell_{γ,K}$ (basis points)` /
  `s_{γ,K}=ℓ/q` (percent)
- mixed γ: generic English names; the loss axis still carries the basis-point
  line, and depth still carries `(%)`
- `loss_unit="percent"`: the loss axis ends with `(%)` and does not add the
  basis-point note

`test_gamma0_axis_labels`, `test_gamma_positive_axis_labels`. Do not treat
the three γ values as three colours of one indicator.

## Style constants (source)

Styles are set on the figure and its artists. Importing `plotting` does not
change global `matplotlib.rcParams`.

| Constant | Value | Role |
|---|---|---|
| `INK` / `MUTED` | `#17202A` / `#5E6D7C` | text and secondary text |
| `GRID_COLOR` / `SPINE` | `#E8EDF2` / `#CBD3DC` | major grid and spines |
| `FONT_SIZE` | 8.5 | axis labels |
| `TITLE_SIZE` / `CAPTION_SIZE` / `PANEL_TITLE_SIZE` | 11 / 8 / 9 | figure title, interval caption, panel letter |
| `TICK_SIZE` / `ANNOTATION_SIZE` | 7.5 / 7 | ticks and in-axes notes |
| `INNER_LINEWIDTH` | 1.3 | solid inner outline (kept in 1.2–1.4) |
| `OUTER_LINEWIDTH` / `OUTER_LINESTYLE` | 1.05 / `(0, (1.6, 1.35))` | short dash, only when the outer set is distinct |
| `FILL_ALPHAS` | `(0.10, 0.14, 0.18)`, then clipped at 0.18 | light inner fills |
| `BAND_FILL_ALPHA` / `MIDLINE_WIDTH` | 0.11 / 1.35 | interval band and its center line |
| `GRID_LINEWIDTH` | 0.4 | major grid only |
| `PANELS_SIZE` | `(7.2, 3.1)` in | unchanged |
| `REPORT_SIZE` | `(9.2, 7.0)` in | unchanged |
| `INSET_BOUNDS` | `(0.08, 0.08, 0.34, 0.34)` | unchanged |
| `SAVE_DPI` | 300 | unchanged |
| log x-axis | `upper/lower ≥ 12` | unchanged |

Information-set colours live in `config.COLORS`:

| Set | Colour |
|---|---|
| `put_pair` | `#2C4E86` |
| `local_eight` | `#0F7F78` |
| `put_wing` | `#A66B00` |
| `full_otm` | `#2C7A45` |
| `full_panel` | `#8E3D5B` |

Unknown series names cycle `PALETTE`, a quieter set of the same family.
Quote roles are not those colours: OTM put `#5C4B9A`, OTM call `#C2512C`,
ITM `#7E8A96`. `LABELS` are full legend names; annotations use `SHORT_LABELS`
(Pair / Eight / Put wing / OTM / All).

Legends sit above the axes, frameless. On the two-panel figure the shared
legend moves into the header so it does not cover the panel letter.
`fronts` accepts `Frontier` | `dict` | `Sequence`. A single object is keyed
by `quote_set`. A sequence is keyed by the concatenation of those among
`quote_set`, `γ=`, `K=`, and date that **differ from the other entries**
(`_as_frontiers`). Empty sequence → `ValueError`.

## Figures

`plot_frontier`: x-axis probability %, y-axis loss in basis points or %.
Inner fill plus a solid outline. `outline_outer` draws the outer short dash
only when `area_outer` is visibly larger than `area`. `show_bounds` draws the
marginal box with a dotted outline. `annotate` writes short labels inside the
axes (Pair: `p …`, not `Complete put wing: p ∈`). γ=0 pads the y-axis from 0;
γ>0 does not force 0.

`plot_conditional_depth`: see [depth-transform.md](depth-transform.md).
`log_p=None` follows the factor-12 rule; it can be forced. `inset` appears
when the last layer is much smaller than the full range
(`DEPTH_INSET_RATIO=1.35`). `legend` can be turned off for two-panel reuse.

`plot_frontier_panels`: 1×2, size `PANELS_SIZE`, default title
`{date} → {expiry} ({days}d), K={K}`, plus an interval caption under the
title (`_caption_intervals`).

`plot_quotes`: x-axis $x$, y-axis normalised call price. Vertical segments
`[norm_l,norm_u]`. OTM puts, OTM calls, and ITM use the role colours above,
not the information-set colours. When `used` is given, unused quotes have
alpha 0.3. `highlight` marks constraint endpoints. $K$ is a vertical dashed
line. Intrinsic value $\max(1-x,0)$ is a thin line.

`plot_binding_quotes`: log axis on `|weight|`. A vertical line between the
quote group and the structural group. Weights $|w|\le 10^{-8}$ are omitted.
Each displayed weight is annotated once, on the middle mark of that value,
so identical weights are not labeled on every quote.

`plot_report`: 2×2 = quotes, joint set, depth, certificate or text summary.
The summary states once that loss is in basis points (`1 bp = 0.01%`).
Certificates are meaningful only at γ=0. CLI `--layout report --gamma 0`
computes `binding_quotes(direction=(1,0))` itself.

`plot_term_structure` / `plot_threshold_profile`: x-axis `days` or `K`,
probability band on the left and depth band on the right. The band is light;
the midline is the heavier line. Rows with nonempty `error` are skipped
(`_error_mask`).

`plot_time_series`: x-axis dates, `value` ∈ `{prob,loss,depth}`. Failed dates
are left blank. Loss uses the same basis-point wording as `plot_frontier`.
`overlay` draws a muted right-axis line.

`save_figure(fig, stem, formats=("pdf","png"), dpi=300)`: if `stem` already
has an image suffix, strip it then append. `bbox_inches="tight"`,
`pad_inches=0.12`.

Library functions do not set Agg; Jupyter embeddings choose the backend.

## Export

`frontier_frames(fronts)` → four tables, each with a `name` column:

| Key | Contents |
|---|---|
| `summary` | one `Frontier.summary()` row + `name` |
| `vertices` | `kind, order, prob, loss, depth` (decimals, depth=$L/p$) |
| `halfplanes` | `a_prob, a_loss, h` |
| `quotes` | full columns of quotes that entered the model |

`summary()` column order is fixed (`frontier._SUMMARY_KEYS`) and includes
`solver_gap` (γ=0 max duality gap, γ>0 max violation).

`write_frontier(fronts, path_stem, formats=("csv","json"), certificate=None)`:

- csv: four files `{stem}_summary.csv` etc.; with a certificate also
  `{stem}_binding.csv` (`DualCertificate.to_frame()`)
- json: `{stem}.json` =
  `{"frontiers": {name: to_dict()}, "certificate": records|null}`
- Unrequested formats are not written. No certificate → no binding file
  (`test_write_frontier_nested_and_write_json`)

`Frontier.to_dict()`: timestamps `YYYY-MM-DD`, arrays as lists, includes
`normalizer.origin/scale` and `quotes` records. `json.dumps` can serialise
(`test_summary_keys_to_dict_and_contains`).

`write_json`: write `{path}.tmp` then `replace`; a `.tmp` must not remain
after exit.

The export name of a single `Frontier` is `frontier.quote_set` (custom tables
are `"custom"`).

## CLI mapping

`crashfrontier frontier --export csv,json --out STEM` calls `write_frontier`.
`--formats` is **figure** formats, not table formats. Missing `--out` prints
only `describe()`. The `binding` subcommand with `--out` stores the figure
only, not a binding CSV; tables come from report export.
