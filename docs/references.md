# References

The papers `clvkit` implements. The volume, issue and page numbers below are
the ones the publishers registered in CrossRef.

## Licence status: none of these is open access

Every paper here was checked against its CrossRef licence record on 25 July
2026. The result is uniform:

| Paper | Publisher | Licence registered |
|---|---|---|
| BG/NBD (2005) | INFORMS | none |
| Iso-value curves (2005) | SAGE | text-and-data-mining only |
| Probability models (2009) | SAGE | text-and-data-mining only |
| Empirical validation (2007) | Elsevier | Elsevier TDM user licence |
| Gamma-gamma note | self-published by the authors | none stated |

A text-and-data-mining licence lets a subscriber run analysis over the text. It
is not permission to republish the PDF, and the two are easy to confuse. The
notes and reprints on [brucehardie.com](https://brucehardie.com/) are free to
download because the authors self-archived them, which is a permission to read
rather than a licence to redistribute.

So `clvkit` ships no PDFs. Fetch them yourself from the links below.

## The models

**BG/NBD** — the transaction-flow model behind `clvkit.BGNBD`, and the source of
the `r = 0.243, alpha = 4.414, a = 0.793, b = 2.426` the test suite reproduces.

> Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting Your Customers"
> the Easy Way: An Alternative to the Pareto/NBD Model. *Marketing Science*,
> 24(2), 275–284. <https://doi.org/10.1287/mksc.1040.0098>
> · [author copy](http://brucehardie.com/papers/018/)

**Gamma-gamma** — the spend model behind `clvkit.GammaGamma`, and the reason
monetary value is estimated separately from timing.

> Fader, P. S., & Hardie, B. G. S. (2013). *The Gamma-Gamma Model of Monetary
> Value*. Note 025. <https://www.brucehardie.com/notes/025/gamma_gamma.pdf>

**Iso-value curves** — where RFM and lifetime value are joined, and the source of
the discounted expected transactions in `clvkit.CLV`.

> Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). RFM and CLV: Using
> Iso-Value Curves for Customer Base Analysis. *Journal of Marketing Research*,
> 42(4), 415–430. <https://doi.org/10.1509/jmkr.2005.42.4.415>

## Context

**Probability models** — the survey that explains why any of this is a
likelihood rather than a heuristic.

> Fader, P. S., & Hardie, B. G. S. (2009). Probability Models for Customer-Base
> Analysis. *Journal of Interactive Marketing*, 23(1), 61–69.
> <https://doi.org/10.1016/j.intmar.2008.11.003>

**Empirical validation** — the independent comparison, and the reason to hold
BG/NBD's published numbers rather than trust an implementation.

> Batislam, E. P., Denizel, M., & Filiztekin, A. (2007). Empirical validation and
> comparison of models for customer base analysis. *International Journal of
> Research in Marketing*, 24(3), 201–209.
> <https://doi.org/10.1016/j.ijresmar.2006.12.005>

## BibTeX

```bibtex
@article{fader2005counting,
  author  = {Fader, Peter S. and Hardie, Bruce G. S. and Lee, Ka Lok},
  title   = {``Counting Your Customers'' the Easy Way: An Alternative to the {Pareto/NBD} Model},
  journal = {Marketing Science},
  volume  = {24},
  number  = {2},
  pages   = {275--284},
  year    = {2005},
  doi     = {10.1287/mksc.1040.0098}
}

@techreport{fader2013gammagamma,
  author      = {Fader, Peter S. and Hardie, Bruce G. S.},
  title       = {The Gamma-Gamma Model of Monetary Value},
  institution = {Note 025},
  year        = {2013},
  url         = {https://www.brucehardie.com/notes/025/gamma_gamma.pdf}
}

@article{fader2005rfm,
  author  = {Fader, Peter S. and Hardie, Bruce G. S. and Lee, Ka Lok},
  title   = {{RFM} and {CLV}: Using Iso-Value Curves for Customer Base Analysis},
  journal = {Journal of Marketing Research},
  volume  = {42},
  number  = {4},
  pages   = {415--430},
  year    = {2005},
  doi     = {10.1509/jmkr.2005.42.4.415}
}

@article{fader2009probability,
  author  = {Fader, Peter S. and Hardie, Bruce G. S.},
  title   = {Probability Models for Customer-Base Analysis},
  journal = {Journal of Interactive Marketing},
  volume  = {23},
  number  = {1},
  pages   = {61--69},
  year    = {2009},
  doi     = {10.1016/j.intmar.2008.11.003}
}

@article{batislam2007empirical,
  author  = {Batislam, Emine Persentili and Denizel, Meltem and Filiztekin, Alpay},
  title   = {Empirical validation and comparison of models for customer base analysis},
  journal = {International Journal of Research in Marketing},
  volume  = {24},
  number  = {3},
  pages   = {201--209},
  year    = {2007},
  doi     = {10.1016/j.ijresmar.2006.12.005}
}
```

## Data

CDNOW's 1/10 systematic sample, 2,357 customers, is the dataset every published
estimate above was fit on. It has circulated freely for two decades and ships
with several packages, including `lifetimes` and R's `CLVTools`. See
[`docs/research/academic-datasets.md`](research/academic-datasets.md) for which
copy matches which published fit.

CDNOW is a single 1997 acquisition cohort, so it cannot demonstrate a cohort
matrix. **UCI Online Retail II** carries the cohort-retention examples instead:
1,067,371 line items, 5,878 identified customers, December 2009 to December 2011,
and 25 natural monthly acquisition cohorts. Unlike the papers above it is CC BY
4.0, so it is redistributable, but `examples/online_retail_ii_cohort.ipynb`
fetches and caches it rather than vendoring 43 MB into every clone.

| | |
|---|---|
| Dataset | [UCI ML Repository, Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) |
| Licence | CC BY 4.0 |
| Paper | Chen, D., Sain, S. L. & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management* 19(3), 197–208. [doi](https://doi.org/10.1057/dbm.2012.17) |
