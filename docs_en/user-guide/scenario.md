# Scenario check: is this point still compatible?

Given a crash-probability and expected-loss pair, `contains` decides geometrically first; if the point sits in the shell between inner and outer, `scenario_check` solves the program again. The script also demonstrates `portfolio_bound` on the risk-neutral model.

## Run

Point `CRASHFRONTIER_DATA` at your quote directory. See [Market data](../getting-started/data.md).

The dates in the listing are sample dates. They must exist in that folder; otherwise pick a date from `ds.available_dates()`.

Save as `scenario.py` and run `python scenario.py`.

```python
import os
from pathlib import Path

from crashfrontier import Dataset, compute_frontier, portfolio_bound, scenario_check


def report(label: str, result: dict) -> None:
    print(
        f"{label}: feasible={result['feasible']}  "
        f"geometry={result['geometry']}  method={result['method']}"
    )


def main() -> None:
    ds = Dataset.from_directory(Path(os.environ["CRASHFRONTIER_DATA"]))
    cs = ds.cross_section("2017-07-05", target_days=14, window=(7, 21))
    frontier = compute_frontier(cs, K=0.90, quote_set="put_wing")
    print(frontier.describe())

    centroid = frontier.inner.mean(axis=0)
    midbox = 0.5 * (frontier.lo + frontier.hi)

    cases = {
        "inner centroid": (float(centroid[0]), float(centroid[1])),
        "marginal-box center": (float(midbox[0]), float(midbox[1])),
        "inner vertex (prob_lo, loss_lo)": (float(frontier.lo[0]), float(frontier.lo[1])),
        "marginal corner (prob_hi, loss_hi)": (float(frontier.hi[0]), float(frontier.hi[1])),
        "clearly outside 0.02, 0.004": (0.02, 0.004),
    }
    print("\ncontains:")
    for label, (prob, loss) in cases.items():
        print(f"  {label}: {frontier.contains(prob, loss)}")

    print("\nscenario_check (same quote set put_wing):")
    for label, (prob, loss) in cases.items():
        result = scenario_check(cs, 0.90, prob, loss, gamma=0, quote_set="put_wing")
        report(label, result)

    lo, hi = portfolio_bound(cs, 0.90, 1.0, 0.0, quote_set="put_wing")
    print("\nportfolio_bound direction (1,0) =", (lo, hi))
    print("should match prob_lo/prob_hi:", frontier.lo[0], frontier.hi[0])

    print("\nphysical scenarios: switch the cross-section and pass the same support_cap:")
    cs30 = ds.cross_section("2020-03-25", days=30, am_settlement=0)
    physical = scenario_check(
        cs30,
        0.90,
        0.13,
        0.013,
        gamma=2,
        quote_set="put_wing",
        support_cap=5.0,
    )
    report("2020 gamma=2 near-interior 0.13, 0.013", physical)


if __name__ == "__main__":
    main()
```

## Decision rules

### Three geometric states

`Frontier.contains(prob, loss)` returns:

- `inside`: the point is on the inner polygon (including its boundary)
- `outside`: the point is strictly beyond the outer polygon
- `boundary`: the shell, including the outer boundary when it disagrees with the inner set

On the 2017 put wing the inner centroid, the marginal-box center, and `(prob_lo, loss_lo)` are `inside`; `(prob_hi, loss_hi)` and `(0.02, 0.004)` are `outside`. A corner of the marginal box need **not** lie in the joint set.

### `scenario_check` solves again

```text
scenario_check(cs, K, prob, loss, *, gamma=0, quote_set=None, **kwargs)
```

Returns a dictionary: `feasible`, `geometry`, `method` (`lp` or `socp`), `frontier`. Rules:

- An interior point is immediately `feasible=True`
- An exterior point is immediately `feasible=False`
- The shell is resolved by another LP at \(\gamma=0\) and another SOCP at \(\gamma>0\)

`**kwargs` are forwarded to `compute_frontier`, so a physical scenario must carry the same `support_cap` and `quote_set`. Temporarily shrinking the quote set to make a scenario feasible answers a different information set.

`scenario_check` recomputes a frontier (cost comparable to `compute_frontier`). If you already have `frontier`, screen points with `contains` and call it only on `boundary`.

### `portfolio_bound` is defined only at \(\gamma=0\)

`portfolio_bound(cs, K, u_prob, u_loss)` returns the normalized price interval \((-h(-u), h(u))\) of the linear crash payoff \(u_p 1\{X\le K\}+u_L(K-X)^+\). `u=(1,0)` is the marginal bound on crash probability and should match `prob_lo` / `prob_hi`. Passing `gamma=1` raises an error.

The script writes no figures. Sample output:

```text
inner centroid: feasible=True  geometry=inside  method=lp
clearly outside 0.02, 0.004: feasible=False  geometry=outside  method=lp
portfolio_bound direction (1,0) = (0.0006565, 0.0069257)
2020 gamma=2 near-interior 0.13, 0.013: feasible=True  geometry=inside  method=socp
```

The 2020 point is close to a \(\gamma=2\) inner centroid `(0.1305, 0.0132)`. If you change `quote_set` or \(B\), recompute the centroid before checking.

<figure class="cf-example-figure">
  <img src="../assets/images/examples/06_scenario_check.png" alt="Risk-neutral frontier used for scenario classification" loading="lazy">
  <figcaption>
    2017-07-05 → 2017-07-19 (14 days), K=0.90, γ=0 put wing. Scenario checks return a dictionary; this figure is the joint set they are classified against.
  </figcaption>
</figure>

## Options

| Task | Call |
|---|---|
| Another quote layer | `scenario_check(..., quote_set="local_eight")` |
| Physical | `gamma=2, quote_set="put_wing", support_cap=5.0` |
| Price the loss direction | `portfolio_bound(cs, 0.90, 0.0, 1.0)` |
| Your own direction | `binding_quotes(cs, 0.90, direction=(1.0, 1.0))` |

Inputs must be decimals: “2% and 40 bp” on a figure is `prob=0.02, loss=0.004`.

## Common errors

- **Feeding figure 0.69 (percent) and 1.75 (bp) straight into `contains`.** The point lands far outside the set.
- **Using `(prob_hi, loss_hi)` as an “upper-bound scenario.”** On this slice it is `outside`.
- **Omitting `support_cap` on a physical check.** The default \(B=5\) is only meaningful if that is what you used for the figure; if you used `None`, the check must use `None` as well.
- **Dropping quotes to force `feasible=True`.** You are already asking a different set.
- **Expecting `scenario_check` to return a figure.** It returns a dictionary. Plot with `plot_*`.
