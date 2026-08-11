# Opinions

`clvkit` implements the BTYD / probability-model canon (Fader, Hardie, et al.)
verbatim wherever the literature settles a question. Where it doesn't — and a
default still has to be picked — we record the decision here: the question,
the options the literature actually offers, the choice `clvkit` makes, and
why. This is not canon; it's where we chose.

Some of these defaults are overridable and some are fixed policy. The
overridable ones are `on_negative` and `collapse`
(`CustomerBase.from_transactions`), `margin` (`CLV.predict`),
`check_independence` (`CLV`), the `max_correlation` / `max_eta_squared`
thresholds (`IndependenceCheck.holds`), and `period` (`CohortSurvival.predict`).
The rest are conventions baked into the summary or the result, and changing one
would change what the number means rather than how it is configured.

Canon that is *not* a choice (e.g. `frequency` meaning the BTYD repeat count,
not the raw transaction count) lives in the main docs, not here.

---

## Returns and refunds (negative amounts)

**Question.** A transaction log can contain negative amounts — refunds,
chargebacks, returns. The BTYD literature is silent on this; the classic
datasets (CDNOW) don't have them. What should `CustomerBase.from_transactions`
do with a negative amount?

**Options in the literature.** None — this is an implementation gap the
canonical papers don't address. Practitioner conventions vary:
1. Ignore the sign and treat every row as a purchase event (silently wrong —
   corrupts `monetary_value` and can fabricate purchase events out of pure
   refunds).
2. Discard negative rows entirely before any aggregation.
3. Net refunds against purchases within whatever granularity you're
   aggregating at, and only count the result as an event if it's still a net
   positive.
4. Refuse to guess — force the caller to clean the data first.

**Our choice.** `on_negative` shapes `monetary_value` and nothing else.
Under every mode, a bucket containing at least one transaction with a
non-negative amount is a purchase event, and `frequency`, `recency` and `T`
are computed from those events; switching the policy never moves them.
`on_negative="net"` (the default) nets each bucket's transactions and
carries the result into `monetary_value` only when it stays positive; a
bucket netting to zero or below keeps its place in the timing columns and
contributes no spend. `on_negative="drop"` discards negative rows before
summing spend. `on_negative="raise"` refuses the data and forces an explicit
choice. Whenever a policy nets a bucket away, discards rows, or erases a
customer whose every transaction is negative, the summary says so with a
counted warning. A bucket containing only refunds is not a purchase event
under any mode: netting corroborates spend, it never fabricates a purchase.

**Why.** Purchase timing is observed fact; spend is what the policy exists
to clean. A refund is evidence about money, not evidence that the purchase
beside it never happened, so it may empty an event's spend but not the
event. Netting per bucket still matches what the same-`time_unit` collapse
(below) implies — a customer's activity within one bucket is one economic
event — and it still gets a first-time user to useful output without a
data-cleanup step first. A customer with no non-negative transaction at all
has no purchase to anchor `T` to, so they still get no row; the difference
is that the warning now counts them out loud.

Until v0.1.x the net filter applied to the timing basis too, and this page
defended the result as deliberate. It was measured and reversed: on CDNOW
the default silently removed 8 of 2,357 customers whose only transaction is
$0.00, removed by a parameter named "negative", and moved the fitted BG/NBD
log-likelihood from −9582.429 to −9578.197. A refund landing in the same
bucket as a real purchase deleted that purchase from `frequency`, `recency`
and `T` entirely. Our own golden test carried the workaround
(`amount_col=None`) in a comment, which is a knowledge-base fix for a
runtime problem — the definition of the wrong default.

---

## Which grain collapses purchases (`collapse` vs `time_unit`)

**Question.** Two questions hide behind one word. *Which* transactions
collapse into a single purchase — the counting process the BG/NBD and
Gamma-Gamma likelihoods assume has no notion of "two events at the same
instant" — and *what unit* `recency` and `T` are reported in. These are not
the same question, and the canonical CDNOW fit answers them differently:
collapse at the data's daily resolution, report time in weeks.

**Options in the literature.** The papers are explicit that same-period
purchases collapse, and equally explicit that time is continuous — Fader,
Hardie & Lee write `T_i = 39 - time of first purchase` in weeks while working
from daily CDNOW records. `lifetimes` exposes a single `freq` doing both jobs.

On what happens to same-period transactions, the alternatives to collapsing
are to keep them as separate events at finer-than-modelled granularity, which
breaks the counting-process assumption the likelihoods rely on, or to drop all
but one, which silently discards spend. Neither appears in the canonical
papers; nothing there models sub-period multiplicity at all.

On how to expose the two questions, the options are: one knob (simple, but
cannot express the published fit), two knobs (says what the papers say, one
more argument), or a fixed daily collapse with a free ruler (no argument, but
takes away a real choice — some businesses genuinely want a weekly event
grain).

**Our choice.** Two arguments. `collapse` sets the event grain and defaults to
`time_unit`, so a caller who names only `time_unit` gets exactly the old
single-knob behaviour. Amounts are summed within a collapsed purchase. When
`collapse` is inherited *and* coarser than daily *and* actually merged
something, `from_transactions` warns with the count; naming `collapse`
explicitly is read as consent and stays silent.

**Why.** The collapse itself is canon — deviate and the fitted parameters stop
meaning what the papers say they mean. Tying it to the ruler was not. The
coupling made `time_unit="W"` look like a change of units when it was really a
deletion of data, and it deletes hardest from the most frequent buyers, which
biases the fit downward precisely where the model earns its keep. On the CDNOW
sample it moves `r` from .243 to .291 and `alpha` from 4.414 to 6.852. The
default stays coupled because that is the behaviour every existing caller
already has, and the warning — not a breaking change — is what stops it being
a silent trap. Summing (rather than averaging or taking the max) preserves
total spend, which is what `monetary_value` and Gamma-Gamma care about.

Only conversions with a fixed ratio are allowed: a week is always seven days,
but a month is 28-31, so `collapse="D", time_unit="M"` is refused rather than
served by an averaged month that would look precise and not be.

---

## `monetary_value` excludes the first transaction

**Question.** `monetary_value` feeds the Gamma-Gamma spend model, which
estimates a customer's expected spend *per future transaction*. Should it be
computed from all of a customer's transactions, or only the repeat ones?

**Options in the literature.** Fader, Hardie & Lee's Gamma-Gamma model
(2005) is explicit: the model conditions on frequency (repeat count) and
fits average spend across a customer's repeat transactions, excluding the
first. This is canon, not a live debate — but it's easy to implement wrong
(e.g. by averaging every transaction including the first), so we record it
here rather than leave it undocumented.

**Our choice.** `monetary_value` is the mean of a customer's repeat
transactions (post same-`time_unit` collapse), excluding the first. A
single-purchase customer gets `monetary_value = 0`.

**Why.** The first transaction is definitionally not a "repeat" and mixing
it into the average biases the statistic the Gamma-Gamma likelihood expects.
Zero (rather than `NaN`) for single-purchase customers keeps the RFM table
numeric and avoids propagating `NaN` into downstream model fits.

---

## When is monetary independence "violated"?

**Question.** `CLV` multiplies expected spend by expected transactions. That
factorisation — equation (1) of Fader, Hardie & Lee (2005), *RFM and CLV* — is
only legitimate if average transaction value is independent of the transaction
process (their assumption §2.1(iii)). §2.2 assesses that assumption on CDNOW
rather than asserting it. At what point should `clvkit` tell you the assessment
has failed on *your* data?

**Options in the literature.** The paper reports the evidence and then
exercises judgement: a simple correlation of 0.11 between average transaction
value and number of transactions across the 946 repeat buyers, 0.06
(`p = 0.08`) once one outlier is removed, and Figure 4 read as "the variation
within each number-of-transactions group dominates the between-group
variation" — concluding "we do not feel that it represents a substantial
violation". No threshold is stated, there or anywhere else in the BTYD
literature. The options are therefore: report the statistics and stay silent;
pick a threshold and warn; or pick a threshold and refuse to fit.

**Our choice.** `CLV.fit` runs the §2.2 assessment and emits a
`MonetaryIndependenceWarning` — never an error — when either

- `|Spearman ρ|` between `frequency` and `monetary_value` across repeat buyers
  reaches **0.30**, or
- `η²`, the share of `monetary_value` variance explained by `frequency`,
  reaches **0.25**.

Both are arguments to `IndependenceCheck.holds(...)`, and the whole check is
switched off with `CLV(check_independence=False)`.

**Why.** A warning, because the paper's own accepted dataset shows a real
positive correlation: only the analyst can judge whether the relationship in
their business is substantial. Spearman rather than the paper's Pearson,
because §2.2's own headline number was distorted by a single customer and had
to be recomputed by hand — a rank correlation does that outlier removal
automatically. `η²` alongside it, because it is exactly the paper's Figure 4
sentence stated as a number, and it catches a non-monotone relationship that
a correlation would miss. The levels: 0.30 is the conventional boundary
between a weak and a moderate correlation, and CDNOW — the case the authors
explicitly accept — sits at ρ = 0.21, η² = 0.10, so the defaults pass the
literature's own worked example with room to spare.

---

## `margin` defaults to 1.0 (CLV is revenue, not contribution)

**Question.** Equation (1) of Fader, Hardie & Lee (2005) is
`CLV = margin × revenue/transaction × DET`. The Gamma-Gamma model estimates
revenue per transaction, so the margin has to come from somewhere. What should
`CLV.predict` assume when the caller doesn't say?

**Options in the literature.** The papers take margin as a known business
input and don't default it. Practitioner tools split: some report revenue-based
lifetime value and leave margin to the caller; some bake in a placeholder
margin (often 0.05–0.10, from the CDNOW-era examples), which silently makes
every number a twentieth of what a reader expects.

**Our choice.** `margin=1.0`, so `CLV.predict(...)` returns **revenue**-based
lifetime value by default. Pass your own gross margin to get contribution-based
CLV.

**Why.** A wrong margin is invisible: it scales every customer identically, so
rankings, plots, and holdout comparisons all look fine while the absolute
numbers are wrong by an order of magnitude. `1.0` is the only default that
cannot be silently wrong — it says "this is revenue", which is what the
monetary model actually estimated, and any other value is a business fact
`clvkit` has no way to know.

---

## The cohort matrix's incomplete triangle

**Question.** `CohortMatrix` is descriptive, so there is no canon to appeal
to — it's a pivot, and the BTYD papers never define one. A cohort acquired
last month has not lived through period 3 yet, so its period-3 cell has no
observation behind it. What goes in that cell, and are the cells that do
have observations counts or rates?

**Options in the literature.** None; this is practitioner convention, and
the common hand-rolled pandas pivot gets it wrong in a specific way. A
`pivot_table` with `fill_value=0`, or a `.fillna(0)` after `unstack`, writes
zeros into cells that were never observed — so a young cohort renders as a
cohort that churned to nothing. Rates vs. counts splits the same way: some
tools show the raw count, most show the percentage of the cohort.

**Our choice.** An unobserved cell is `NaN`; only a period the cohort
actually lived through can hold `0`. `to_pandas()` returns absolute values
(active-customer counts, or summed revenue) and `to_pandas(relative=True)`
divides each cohort by its own period-0 value. `plot()` inverts that default
— it draws the relative view, and paints unobserved cells flat grey rather
than at the bottom of the colour ramp.

**Why.** "Nobody bought" and "we don't know yet" are different facts, and
collapsing them into `0` is the single most common bug in a hand-rolled
cohort pivot — the empty triangle it produces looks exactly like catastrophic
churn. Keeping the stored matrix absolute means no information is destroyed
(counts and rates are recoverable from counts; the reverse needs the cohort
sizes back), while the chart defaults to relative because cohorts differ in
size and the eye is there to compare decay curves, not cohort volumes. The
cost of the `NaN` policy is that a retention matrix is float-typed even
though it counts customers.

---

## What a model-based survival curve is

**Question.** A non-contractual business never observes a customer leaving,
so there is no event, no censoring indicator, and nothing for a survival
analysis to consume. `CohortSurvival` is specified as "aggregated fitted
P(alive)" — but P(alive) is a *per-customer* posterior evaluated at that
customer's own age `T`. What curve should aggregating it produce, at what
cohort grain, and what goes on the x-axis?

**Options in the literature.** None. The BTYD papers define P(alive) (Fader,
Hardie & Lee 2005, the denominator of eq. (10)) and stop there; they never
aggregate it, and the survival curves in the marketing literature —
Kaplan–Meier, the shifted-beta-geometric of Fader & Hardie (2007) — are
contractual, fitted to observed cancellations this data does not contain.
Deriving a curve therefore means choosing: (a) mean P(alive) per acquisition
cohort, each cohort plotted at the age it has reached; (b) sum P(alive) per
cohort, an expected head-count rather than a share; (c) pool the whole base
and plot mean P(alive) against `T`, which is (a) with the cohort grain
pinned to `time_unit`; (d) refuse, and expose only per-customer P(alive).

**Our choice.** (a). `predict()` returns one row per acquisition cohort:
`survival` is the cohort's **mean** P(alive), `customers` its size, and `age`
the whole periods it had lived by `observation_period_end`. Cohorts are
derived from `T` — an acquisition is `T` periods before the observation
period end, by definition — so no data beyond the `CustomerBase` is needed.
The grain defaults to the base's own `time_unit` and is coarsened with
`predict(period="M")`; a grain *finer* than `time_unit` is refused rather
than interpolated.

**Why.** The mean of a posterior probability over a group is the expected
share of that group still alive — that is arithmetic, not a new assumption,
which is what keeps this "survival" honest in a setting with no observed
churn. A share rather than a head-count because cohorts differ in size and
the curve exists to be compared across them (the same argument as the cohort
matrix's `relative` default); the head-count is one multiplication away, and
`customers` is in the frame for exactly that. Cohort grain defaults to
`time_unit` because that is the finest grain the RFM summary can resolve and
the only one that needs no choice from the caller — but a base summarised
daily has as many cohorts as it has days, which is unreadable rather than
wrong, so coarsening is a one-word argument. Refusing a finer grain is the
same principle in reverse: `T` is a whole number of `time_unit`s, so daily
cohorts from a weekly base would be a precise-looking answer to a question
the data cannot resolve. When a coarser grain cuts a period in half — a week
straddling a month end — the cohort is filed by where that period *starts*,
so a cohort label always means "the bucket this customer's first period
opened in". Pandas' own default when changing frequency is the opposite (by
where the period ends), which would post six-sevenths of a January week to
February.

The honest cost, stated in the class docstring too: every point on the curve
comes from a *different* set of customers, measured at one moment in calendar
time. A cohort acquired through a bad channel reads as a dip in survival that
has nothing to do with age. It is a cross-section across cohorts, not a panel
followed forward — the price of a survival curve in a world where nobody ever
announces that they have left.
