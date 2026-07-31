# Academic / Open Customer-Transaction Datasets — Survey + CDNOW Master-vs-Sample Verdict

Investigated 2026-07-16 against primary sources (brucehardie.com PDFs, package source on GitHub/CRAN, UCI, the papers themselves). Where a claim comes from a dataset object, the figure is a live read of the actual bundled data (loaded and counted), not documentation folklore. For the `clvkit` `(customer_id, date, amount)` contract, two pillars need datasets: **BTYD/CLV benchmarking** (needs repeat transactions + published parameter estimates) and **multi-cohort retention demos** (needs many acquisition cohorts spread over calendar time — CDNOW has only one).

---

## TL;DR

- **Goal B (verdict):** The canonical published estimates come from **two different Fader/Hardie/Lee (2005) papers, both on the 2,357-customer 1/10 systematic sample**, 39-week calibration:
  - **BG/NBD** `r=0.243, α=4.414, a=0.793, b=2.426`, LL `−9582.4` — Fader, Hardie & Lee (2005) *Marketing Science*, "Counting Your Customers the Easy Way". **Sample (2,357).**
  - **Gamma-Gamma** `p=6.25, q=3.74, γ=15.44` (+ Pareto/NBD `r=0.55, α=10.58, s=0.61, β=11.67`) — Fader, Hardie & Lee (2005) *JMR*, "RFM and CLV: Using Iso-value Curves". Validated on the **sample (2,357)**; only the final *aggregate RFM-group CLV table* uses the "full" 23,560 set.
  - **`lifetimes.load_cdnow_summary()` = the 2,357 sample** (verified: 2,357 rows, row 1 = `freq 2, recency 30.43, T 38.86`, identical to the paper's Excel screenshot).
  - **`CLVTools::cdnow` = the 2,357 sample** (verified by loading the `.rda`: 2,357 unique `Id`, 6,696 transaction rows, 1997-01-01→1998-06-30). Its roxygen *description* text quotes the 23,570-cohort context, which is misleading — the shipped object is the sample.
  - **A golden test targeting the published BG/NBD or Gamma-Gamma numbers MUST use the 2,357 sample, not the 23,570 master.** The current repo choice of "CDNOW master (23,570)" as the primary golden dataset is **wrong for parameter-matching** and should be switched to the sample. The master is only correct for reproducing the iso-value paper's final aggregate CLV-by-RFM-group table.

- **Goal A (shortlist):** Best multi-cohort candidate is **UCI Online Retail II** (2 years, ~5,900 customers, ~1.07M line items, natural monthly acquisition cohorts, CC BY 4.0). For BTYD golden #2, **BTYDplus `groceryElog`** (has published Pareto/NBD & MBG/CNBD-k estimates). CDNOW stays the primary BTYD anchor.

---

## Goal B — CDNOW master vs. sample, resolved

### The two datasets

Primary source: Fader & Hardie (2013), *"Notes on the CDNOW Master Data Set"* — <http://www.brucehardie.com/notes/026/notes_on_CDNOW_master.pdf>.

- **Master** (`CDNOW_master.zip`): "the entire purchase history up to the end of June 1998 of the cohort of **23,570** individuals who made their first-ever purchase at CDNOW in the **first quarter of 1997** ... **69,659** [records] in total, comprises four fields: the customer's ID, the date of the transaction, the number of CDs purchased, and the dollar value." Source download: <http://brucehardie.com/datasets/CDNOW_master.zip>.
- **Sample** (`CDNOW_sample.zip`): "A **1/10th systematic sample** of the whole cohort (**2357** customers) has become a canonical dataset." Drawn by stratifying on trial-week and sorting on 39-week repeat spend, then taking every 10th customer; renumbered IDs 1..2357. Source download: <http://brucehardie.com/datasets/CDNOW_sample.zip>.
- **Time structure (both):** 78 weeks total (Jan 1997 – Jun 1998). Standard split = **39-week calibration (273 days) + 39-week holdout**. Single acquisition cohort (all first-purchase in Q1 1997) — this is why CDNOW is weak for the cohort-retention pillar.

### Which variant produced each published estimate

| Published estimate | Paper (primary source) | Dataset variant | Values |
|---|---|---|---|
| **BG/NBD** `r, α, a, b` | Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way," *Marketing Science* 24(2), 275-284 | **2,357 sample**, 39-wk calibration | `r=0.243, α=4.414, a=0.793, b=2.426`, LL `−9582.4` |
| **Pareto/NBD** `r, α, s, β` | Fader, Hardie & Lee (2005), "RFM and CLV: Using Iso-value Curves," *JMR* 42(4), 415-430 | **2,357 sample**, 39-wk calibration | `r=0.55, α=10.58, s=0.61, β=11.67` |
| **Gamma-Gamma** `p, q, γ` | same JMR iso-value paper | **2,357 sample** (946 repeat buyers in wks 1-39 for spend model) | `p=6.25, q=3.74, γ=15.44` |
| Aggregate CLV-by-RFM-group table | same JMR iso-value paper (final section only) | **"full" 23,560** (= 23,570 master minus 10 buyers who spent >$4,000 over 78 wks) | (table of $ per group, not a parameter set) |

Evidence quotes:

- **BG/NBD → sample.** BG/NBD paper: "For the purposes of this analysis, we take a **1/10th systematic sample** of the customers. We calibrate the model using the repeat transaction data for the **2357 sampled customers** over the first half of the 78-week period and forecast their future purchasing over the remaining 39 weeks." The parameter estimation worksheet screenshot shows `r 0.243 / alpha 4.414 / a 0.793 / b 2.426 / LL −9582.4` over rows ID 0001…2357. Source (working paper, brucehardie's authoritative copy; "identical estimates" to the published MKSC version): <https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf> §7 "Empirical Analysis" + Figure 1.
- **Gamma-Gamma / Pareto/NBD → sample.** Iso-value JMR paper: "Our initial empirical analysis will be based on a **1/10th systematic sample of the whole cohort (2357 customers)**, using the first 39 weeks of data for model calibration ... The maximum likelihood estimates of the model parameters are `r̂=0.55, α̂=10.58, ŝ=0.61, β̂=11.67`" and for spend "`p̂=6.25, q̂=3.74, γ̂=15.44`." Source: <https://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf> (§ empirical analysis / monetary-value validation).
- **Only the final CLV aggregation uses the master.** Confirmed twice: (a) iso-value paper: "This initial exploratory analysis is therefore based on the purchasing of **23,560 customers**"; and (b) the master-data note's closing comment: "The initial exploratory analysis presented in the paper uses the full dataset (23,570 customers) **excluding the purchasing data for ten buyers** who purchased more than $4,000 ... Having **validated the model on the 1/10 sample**, the final RFM-group analysis is based on the revised 'full' dataset of **23,560 customers**." (Note this note's "Fader et al. (2005)" = the *JMR iso-value* paper, not the MKSC BG/NBD paper.)

### What the packages actually ship (verified by loading the data)

| Package / loader | What it is | Verified content |
|---|---|---|
| `lifetimes.load_cdnow_summary()` → `cdnow_customers_summary.csv` | Pre-summarized CBS of the **2,357 sample**, 39-wk calibration | **2,357 rows**, cols `ID, frequency, recency, T`; row 1 = `2, 30.43, 38.86` — matches the BG/NBD paper's Excel screenshot exactly ⇒ this is the exact dataset the published `r=0.243…` came from |
| `lifetimes.load_cdnow_summary_data_with_monetary_value()` → `cdnow_customers_summary_with_transactions.csv` | Same 2,357 sample + monetary | **2,357 rows**, cols `ID, x, t_x, T, zbar` (zbar = avg spend/repeat-txn) — the Gamma-Gamma input |
| `CLVTools::cdnow` (`data/cdnow.rda`) | **2,357 sample**, raw event log (NOT summarized) | **2,357 unique `Id`**, **6,696 rows**, cols `Id, Date, CDs, Price`, dates **1997-01-01 → 1998-06-30**. Roxygen `@format` says "6696 rows"; its prose description quotes the 23,570 cohort — **prose is misleading, object is the sample**. Source: <https://github.com/bachmannpatrick/CLVTools/blob/master/R/data.R> and `data/cdnow.rda` |
| `BTYD::cdnowSummary` (R) | Derived summary of the **2,357 sample** (cbs matrix + weekly tracking) | Standard BTYD teaching object, same 2,357 sample lineage |

**Practical golden-test guidance for `clvkit`:**
- To match published **BG/NBD** `r=0.243, α=4.414, a=0.793, b=2.426` (and MBG/NBD as its near-sibling): fit on the **2,357 sample**, 39-week (273-day) calibration. `lifetimes`' summarized CSV or `CLVTools::cdnow` (summarize it yourself with a 273-day cutoff, T = (273 − first-purchase-day)/7 in weeks) both work; `lifetimes`' pre-summarized CSV is the lowest-friction exact match.
- To match published **Gamma-Gamma** `p=6.25, q=3.74, γ=15.44`: use the sample's **946 customers with ≥1 repeat purchase in weeks 1-39** and their average repeat-transaction value. (Excluding zero-repeat customers matters.)
- **Do NOT** use the 23,570 master to chase these numbers — it will not reproduce them. Reserve the master only if you ever want the iso-value paper's aggregate CLV-by-RFM-group table.

---

## Goal A — Multi-cohort & BTYD dataset survey

Ranked for two fitness axes: **[C]** multi-cohort retention (many acquisition cohorts over calendar time) and **[B]** BTYD/CLV benchmarking (repeat purchasing + ideally published estimates).

| # | Dataset | n cust / n txn / span | Acquisition cohorts | Schema fit `(id,date,amount)` | License | [C] | [B] | Papers using it |
|---|---|---|---|---|---|---|---|---|
| 1 | **UCI Online Retail II** | ~5,900 cust / ~1,067,371 line items / 2009-12-01→2011-12-09 (2 yr) | **Many** (customers acquired throughout 2 yr → natural monthly cohorts) | Direct after aggregating lines→invoice; `InvoiceDate, CustomerID, Quantity×UnitPrice` | **CC BY 4.0** | **A+** | B+ | Chen, Sain & Guo (2012), *J. Database Marketing* 19(3):197-208 |
| 2 | UCI Online Retail (I) | 4,372 cust / 541,909 rows / 2010-12-01→2011-12-09 (1 yr) | Several (1-yr subset of #1) | Same as #1 | CC BY 4.0 | A | B | same Chen et al. (2012) |
| 3 | Olist Brazilian E-Commerce | ~96k unique cust / ~100k orders / 2016-09→2018-10 (2 yr) | **Many** (2-yr acquisition) | `order_purchase_timestamp, customer_unique_id, payment_value` | **CC BY-NC 4.0** | A | **D** (repeat rate ~3%, too sparse for BTYD) | widely used in CLV/ML papers (e.g. arXiv 2308.08502) |
| 4 | Ta Feng grocery | 32,266 cust / 817,741 txn / 2000-11→2001-02 (4 mo) | Several, but **short window** limits cohort spread + holdout | `Transaction date, Customer ID, Sales price` | ACM RecSys release; redistribution unclear | B− | B (strong repeat signal, but no absolute-CLV benchmark) | ACM RecSys; many next-basket papers |
| 5 | dunnhumby "The Complete Journey" | 2,500 households / ~2 yr, full basket | **Weak** — recruited *panel* of frequent shoppers all present throughout, not natural acquisition cohorts | Household-level; multi-table, needs joins | dunnhumby academic terms (free w/ registration) | C | B | retail-science papers (e.g. arXiv 2007.01903) |
| 6 | **BTYDplus `groceryElog`** | 1,525 cust / 10,483 txn / 2006-2007 (2 yr) | **Single quasi-cohort** (all first-purchase Q1 2006) — like CDNOW | Native event log `cust, date, sales` | GPL (bundled in pkg) | D | **A** (has published Pareto/NBD, BG/NBD, MBG/CNBD-k estimates) | Platzer & Reutterer (2016), *Marketing Science* 35(5) |
| 7 | CDNOW sample (incumbent) | 2,357 / ~6,700 txn / 1997 Q1→1998-06 | **Single cohort** (Q1 1997) | Native | brucehardie public download | D | **A+** (the canonical golden) | Fader/Hardie/Lee (2005) ×2 |
| 8 | CLVTools `apparelTrans` | 600 cust / 3,187 txn / 2005-01→2010-12 | **Single cohort** (all 2005-01-03) + covariates | Native `Id, Date, Price` | GPL (bundled) | D | B (good for covariate/dyncov models) | Bachmann et al. CLVTools vignette |
| 9 | brucehardie `donationsSummary` | nonprofit donors, discrete-time | single cohort | summary only | brucehardie public | D | B (discrete-time BG/BB, not retail) | Fader, Hardie & Shang (2010) |

### Notes / caveats per candidate (primary sources)

- **UCI Online Retail II — best multi-cohort.** "all transactions occurring for a UK-based ... online retail between 01/12/2009 and 09/12/2011," gift-ware; 1,067,371 instances, ~5,900 customers, CC BY 4.0, created 2012. Line-item granularity (`InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, Customer ID, Country`) → aggregate lines to an invoice for `amount`. Real-world warts to handle: ~25% rows have missing `Customer ID`, cancellations carry `InvoiceNo` prefix `C` / negative `Quantity`, and it skews B2B/wholesale. Because customers keep being acquired across the 2 years, it yields genuine monthly acquisition cohorts — exactly what CDNOW lacks. Source: <https://archive.ics.uci.edu/dataset/502/online+retail+ii>; paper Chen, Sain & Guo (2012) "Data mining for the online retail industry," *J. Database Marketing & Customer Strategy Mgmt* 19(3):197-208.
- **Olist** — real, anonymized (company/partner names swapped for Game-of-Thrones houses), 100k orders 2016-2018, `customer_unique_id` links repeat buyers. Strong multi-cohort, but **repeat purchase rate ~3%** ⇒ almost no `x>0` signal ⇒ poor BTYD fit; fine as a cohort-retention *shape* demo. License **CC BY-NC 4.0** (non-commercial — matters for redistributing a bundled copy). Source: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>.
- **Ta Feng** — 4-month grocery log, 32,266 customers, 817,741 txn, `Transaction date, Customer ID, Sales price` present. Multi-cohort within months but the 4-month span gives thin calibration+holdout. Redistribution terms unclear (RecSys community release) — link, don't vendor. Source: <https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset>.
- **dunnhumby "The Complete Journey"** — 2,500 households, 2 years, complete baskets + campaigns/coupons. Rich but it is a **recruited panel**, not natural acquisitions, so acquisition-cohort structure is weak; multi-table complexity. Free for academic use via registration. Source: <https://www.dunnhumby.com/source-files/>.
- **BTYDplus `groceryElog` — best secondary BTYD golden.** 1,525 customers, 10,483 txn, 2006-2007; a *quasi-cohort* (restricted to customers whose first purchase fell in Q1 2006), so single-cohort like CDNOW — not a multi-cohort candidate, but it ships with **published Pareto/NBD, BG/NBD and MBG/CNBD-k parameter estimates** in the package vignette (Platzer & Reutterer 2016, *Marketing Science*), making it a strong second golden anchor for the CLV engine (incl. MBG/NBD). Native event log (`cust, date, sales`). Sources: <https://github.com/mplatzer/BTYDplus/blob/master/man/groceryElog.Rd>, vignette <https://rdrr.io/cran/BTYDplus/f/vignettes/BTYDplus-HowTo.Rmd>.
- **CLVTools apparel family** (`apparelTrans` + `apparelStaticCov`/`apparelDynCov`) — simulated single-cohort (250 first-buyers on 2005-01-03; 600 customers total in the covariate tables), useful only if `clvkit` ever adds covariate/time-varying models; not multi-cohort. Source: <https://github.com/bachmannpatrick/CLVTools/blob/master/R/data.R>.

### Explicitly rejected (save future time)

- **Instacart "Market Basket"** — huge, but has **no absolute dates** (only `order_dow`, `order_hour_of_day`, `days_since_prior_order`). Cannot place customers on a calendar ⇒ unusable for both calendar-time cohort retention and BTYD `T`/recency. Skip.
- **Retailrocket** — event stream (view/addtocart/transaction), 2015, ~1.4M events; transactions sparse and recsys-oriented. Low priority.

---

## Recommendations for `clvkit`

1. **Fix the primary golden dataset:** switch BG/NBD & Gamma-Gamma golden tests from the "CDNOW master (23,570)" to the **2,357 sample** — that is what every published FHL parameter set was fit on. Cheapest exact source: `lifetimes`' pre-summarized `cdnow_customers_summary.csv` (2,357 rows) for BG/NBD, and `..._with_transactions.csv` for Gamma-Gamma (filter to the 946 repeat buyers to match `p,q,γ`).
2. **Add `groceryElog` as golden #2** for the CLV engine (published Pareto/NBD, BG/NBD, MBG/CNBD-k) — covers MBG/NBD which CDNOW's canonical numbers don't.
3. **Ship/demo the cohort-retention pillar on Online Retail II** (CC BY 4.0, redistributable) — the only shortlisted set with real multi-cohort structure; use Olist only as a low-repeat contrast demo (and respect its NC license — link, don't vendor).

## Primary sources index

- Notes on the CDNOW Master Data Set — <http://www.brucehardie.com/notes/026/notes_on_CDNOW_master.pdf>
- BG/NBD "Counting Your Customers the Easy Way" (working paper w/ estimation worksheet) — <https://www.brucehardie.com/papers/bgnbd_2004-04-20.pdf> · published: *Marketing Science* 24(2), 2005, 275-284
- "RFM and CLV: Using Iso-value Curves" (Gamma-Gamma) — <https://www.brucehardie.com/papers/rfm_clv_2005-02-16.pdf> · published: *JMR* 42(4), 2005, 415-430
- CDNOW downloads — master <http://brucehardie.com/datasets/CDNOW_master.zip> · sample <http://brucehardie.com/datasets/CDNOW_sample.zip>
- `lifetimes` datasets — <https://github.com/CamDavidsonPilon/lifetimes/tree/master/lifetimes/datasets>
- `CLVTools` data + docs — <https://github.com/bachmannpatrick/CLVTools/blob/master/R/data.R> · <https://www.clvtools.com/reference/cdnow.html>
- `BTYDplus` groceryElog — <https://github.com/mplatzer/BTYDplus/blob/master/man/groceryElog.Rd>
- UCI Online Retail II — <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- Olist — <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>
- Ta Feng — <https://www.kaggle.com/datasets/chiranjivdas09/ta-feng-grocery-dataset>
- dunnhumby Complete Journey — <https://www.dunnhumby.com/source-files/>
