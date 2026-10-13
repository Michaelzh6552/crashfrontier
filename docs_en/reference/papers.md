# Papers

`crashfrontier` implements the identified-set programs of two papers.
`gamma=0` is the risk-neutral linear program of the first paper.
`gamma=1` and `gamma=2` are the physical-measure cone programs of the
second (`gamma=2` is the main specification; `gamma=1` is the
log-investor case).

| Parameter | Paper |
|---|---|
| `gamma=0` | [The Risk-Neutral Crash Frontier](https://arxiv.org/abs/2607.25353), arXiv:2607.25353 |
| `gamma=1`, `gamma=2` | [The Physical Crash Frontier](https://arxiv.org/abs/2608.23274), arXiv:2608.23274 |

## Risk-neutral crash frontier

Jirong Zhuang, “[The Risk-Neutral Crash Frontier: Sharp Joint Bounds on
Crash Probability and Conditional Depth from Option Bid-Ask
Quotes](https://arxiv.org/abs/2607.25353),” arXiv:2607.25353.

```text
Zhuang, Jirong. "The Risk-Neutral Crash Frontier: Sharp Joint Bounds on Crash Probability and Conditional Depth from Option Bid-Ask Quotes." arXiv:2607.25353, 2026. https://arxiv.org/abs/2607.25353
```

```bibtex
@misc{zhuang2026riskneutral,
  author        = {Zhuang, Jirong},
  title         = {The Risk-Neutral Crash Frontier: Sharp Joint Bounds on Crash Probability and Conditional Depth from Option Bid-Ask Quotes},
  year          = {2026},
  eprint        = {2607.25353},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.CP},
  url           = {https://arxiv.org/abs/2607.25353}
}
```

## Physical crash frontier

Jirong Zhuang, “[The Physical Crash Frontier: What Finite Option Quotes
Can and Cannot Reveal](https://arxiv.org/abs/2608.23274),”
arXiv:2608.23274.

```text
Zhuang, Jirong. "The Physical Crash Frontier: What Finite Option Quotes Can and Cannot Reveal." arXiv:2608.23274, 2026. https://arxiv.org/abs/2608.23274
```

```bibtex
@misc{zhuang2026physical,
  author        = {Zhuang, Jirong},
  title         = {The Physical Crash Frontier: What Finite Option Quotes Can and Cannot Reveal},
  year          = {2026},
  eprint        = {2608.23274},
  archivePrefix = {arXiv},
  primaryClass  = {q-fin.CP},
  url           = {https://arxiv.org/abs/2608.23274}
}
```

The empirical applications use OptionMetrics SPX data. Data preparation is
covered in [Market data](../getting-started/data.md).
