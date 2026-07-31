---
permalink: /
title: clvkit
---

*Four parameters. Six decimal places. No trace to read.*

```bash
uv add clvkit     # or pip install clvkit
```

Python 3.13+, four dependencies, no compiler. Buy-till-you-die models for a
business where nobody cancels a subscription, so you never observe a customer
leaving. You only ever observe them not coming back yet.

```python
import pandas as pd

from clvkit import CLV, CustomerBase

log = pd.read_csv("transactions.csv", parse_dates=["date"])  # customer_id, date, amount

cb = CustomerBase.from_transactions(log, time_unit="W", collapse="D")
result = CLV().fit(cb).predict(horizon=52, discount_rate=0.001)

# expected_purchases, discounted_expected_transactions, expected_spend, clv
result.to_pandas()  # indexed by customer_id
result.plot()
```

For the question you actually came for, fit `BGNBD` and call
`probability_alive`. That's the number ranking by last year's spend can't reach.

## It lands where the paper said it would

BG/NBD on the CDNOW 1/10 systematic sample, 2,357 customers, 39-week
calibration. That's the same data Fader, Hardie and Lee published on in
*Marketing Science* in 2005.

| Parameter | Published, 2005 | `clvkit` |
|---|---|---|
| `r` | 0.243 | 0.242595 |
| `α` | 4.414 | 4.413602 |
| `a` | 0.793 | 0.792922 |
| `b` | 2.426 | 2.425907 |

The right-hand column isn't transcribed. The test suite fits it on every pull
request and every push to `main`, and the build fails if a value drifts past
half a unit in the paper's last printed digit.

## What it costs you

`time_unit="W"` without `collapse="D"` moves `α` 55%, from 4.41 to 6.85, because
same-day purchases stay separate instead of collapsing into one. The survival
curve is a cross-section across cohorts, not a panel followed forward. Seven of
these calls are written down with the option not taken and its price, in
[`opinions.md`](https://github.com/nasserboan/clvkit/blob/main/opinions.md).

## Why it exists

[`lifetimes`](https://github.com/CamDavidsonPilon/lifetimes) was archived on
2024-06-28 and last released in July 2020. It was still downloaded 248,263
times in the thirty days to 25 July 2026. Downloads, not people, since PyPI
counts CI runs, mirrors and bots.

Its successor [`btyd`](https://github.com/ColtAllen/btyd) stopped at 0.1b3 in
November 2022 and declares `requires_python = ">=3.8,<3.10"`, so it won't
install on anything current.

Both of their READMEs now point at
[`pymc-marketing`](https://github.com/pymc-labs/pymc-marketing), which is
Bayesian and good. Use it if you want covariates, priors or uncertainty
intervals. `clvkit` is the narrower tool, for an analyst who wants parameters
and a CSV this afternoon.

[The shrug](the-shrug/) is the long version. A week of work, and a CMO who
shrugged at it.

## Links

- [Source on GitHub](https://github.com/nasserboan/clvkit)
- [`clvkit` on PyPI](https://pypi.org/project/clvkit/)
- [The papers, by DOI](https://github.com/nasserboan/clvkit/blob/main/docs/references.md)
