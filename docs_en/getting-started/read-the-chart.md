# Read the figure

`plot_frontier_panels` places the joint probability–loss set on the left and
its conditional-depth representation on the right.

<figure class="cf-example-figure">
  <img src="../assets/images/examples/00_synthetic_frontier.png" alt="Synthetic crash frontier with probability-loss and probability-depth panels" loading="eager">
  <figcaption>
    Synthetic output from the self-contained first example. Later pages use
    the same panel layout.
  </figcaption>
</figure>

## Coordinates and units

Let \(X=S_T/F\) and let \(K\) be the forward-normalized crash threshold.

| Quantity | Meaning | Stored values | Default figure |
|---|---|---|---|
| `prob` | probability of \(X\le K\) | decimal | percent |
| `loss` | expected shortfall \(\mathbb E[(K-X)^+]\) | fraction of forward | basis points (bp) |
| `depth` | conditional shortfall \(s=L/p\) | decimal | percent |

For example, `prob=0.02` is 2%, while `loss=0.0002` is 2 basis
points of the forward. 1 basis point = 0.01 percentage points = 0.0001;
100 bp = 1%. Default loss axes spell the unit out as basis points; this
definition is kept in the surrounding documentation rather than repeated
inside every chart. The parameter name stays `loss_unit="bp"`.
Expected loss is unconditional; depth conditions on the crash event.

Every compatible point satisfies

\[
0\le L\le Kp,\qquad 0\le s=L/p\le K.
\]

At `gamma=0`, the labels are \(p_K,L_K,s_K\). At `gamma=1` or `gamma=2`,
they are \(q_{\gamma,K},\ell_{\gamma,K},s_{\gamma,K}\). The exported column
names remain `prob`, `loss`, and `depth`.

## Inner and outer boundaries

- The solid filled polygon is the **inner approximation**, the convex hull of
  support points already found. The fill is light; the outline is a solid line.
- The short dashed outline is the **outer approximation**, the intersection of
  supporting half-planes. It is drawn only when that outer set is visibly
  wider than the inner set.
- A narrow gap means reconstruction is tight.

Colors name the information set, not the probability measure: put pair,
local eight, put wing, all OTM quotes, and all quotes each have their own
color. Quote bid–ask marks use a separate palette.

`frontier.contains(prob, loss)` reports:

- `inside` when the point is in the inner set;
- `outside` when it lies beyond the outer set;
- `boundary` when it lies in the unresolved shell.

Use `scenario_check` to solve a boundary case directly.

## Marginal bounds are not a corner

`prob_hi` and `loss_hi` are optimized separately. They need not come from the
same admissible distribution, so the point `(prob_hi, loss_hi)` can be outside
the joint set.

The same warning applies to depth: `loss_hi / prob_lo` is not a valid depth
upper bound. Use `depth_lo` and `depth_hi`, or compute `loss / prob` on one
vertex row.

## Nested quote sets

`nested_frontiers` compares how the identified set changes as quotes are
added:

1. `put_pair`
2. `local_eight`
3. `put_wing`
4. `full_otm` for the physical design

Each later information set should weakly tighten the frontier. The colors
distinguish information sets, not different probability measures.

```python
from crashfrontier import nested_frontiers, plot_frontier_panels

frontiers = nested_frontiers(
    section,
    K=0.90,
    sets=("put_pair", "local_eight", "put_wing"),
)
figure = plot_frontier_panels(frontiers)
```

## Inspect the numbers

```python
summary = frontier.summary()
print(summary["prob_lo"], summary["prob_hi"])
print(summary["loss_lo"], summary["loss_hi"])
print(summary["depth_lo"], summary["depth_hi"])
print(frontier.vertices().head())
print(frontier.halfplane_table().head())
```

For physical frontiers, a joint set can be valid even when the separate
depth endpoint program is not numerically certified. In that case
`depth_lo` and `depth_hi` are `NaN`; the plotted depth curve remains the
geometric image of the inner polygon.

Continue with [Quote sets](../user-guide/quote-sets.md),
[Scenario checks](../user-guide/scenario.md), or
[Results and schemas](../reference/results.md).
