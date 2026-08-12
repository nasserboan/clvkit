Models
======

Four objects, four published sources. Each one below is the idea behind it, the
code that runs it, the single assumption it stands on, and the price of the call
``clvkit`` makes on your behalf.

Full citations and DOIs live in `docs/references.md
<https://github.com/nasserboan/clvkit/blob/main/docs/references.md>`__. Every call
priced below is argued in full in `opinions.md
<https://github.com/nasserboan/clvkit/blob/main/opinions.md>`__. Each snippet
assumes a fitted ``CustomerBase`` named ``cb``, from :doc:`Getting started
<getting_started>`.

----

``BGNBD``, the transaction-flow model
-------------------------------------

*Timing: how many purchases, and is the customer still alive?*

Each customer buys at their own steady rate while they're active. After any
purchase they may quietly stop for good, which in a non-contractual business you
never observe directly. BG/NBD learns the spread of buying rates and dropout risk
across the whole base, then reads each customer against it: how many purchases to
expect next, and how likely they're still around.

.. code:: python

   from clvkit import BGNBD

   bg = BGNBD().fit(cb)
   bg.predict(t=12).to_pandas()         # expected purchases in the next 12 time units
   bg.probability_alive().to_pandas()   # P(still active) per customer

``probability_alive`` is the number ranking by last year's spend can't reach. A
customer who spent heavily and then went silent scores high on history and low
here.

How sure are those four numbers?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``fit`` hands back four point estimates and nothing beside them, so a parameter
the data pins down looks exactly like one it barely constrains.
``parameter_uncertainty`` runs a parametric bootstrap: simulate ``n`` synthetic
customer bases from the fitted parameters, refit the model on each, and report
the spread as a standard error and a percentile interval per parameter.

.. code:: python

   unc = bg.parameter_uncertainty(n=100, seed=42)
   unc.to_pandas()   # estimate, se, ci_low, ci_high per parameter
   unc.plot()        # each estimate drawn with its interval

Expect the intervals to be honest rather than comforting. On the CDNOW
calibration base ``r`` comes back tight, 0.243 inside [0.219, 0.269], while the
dropout pair doesn't: ``a`` at 0.793 spans [0.518, 1.184] and ``b`` at 2.426
spans [1.532, 3.934]. Those wide intervals are the model saying 2,357 customers
only weakly identify where dropout risk sits, which is exactly what the method
exists to show.

It's opt-in and adds no dependency. Nothing runs until you call it, and the
price is ``n`` refits, about 30 seconds at ``n=100`` on CDNOW. ``MBGNBD`` has
the same method returning the same result type.

.. note::

   **Assumption.** Purchases are Poisson while alive, and dropout can only follow
   a purchase.

   **Price.** ``collapse="D"`` is canon. Fold same-day purchases into one event.
   Drop it under ``time_unit="W"`` and ``α`` moves 55%, from 4.41 to 6.85. The fit
   still converges, it's answering a different question.

Fader, Hardie & Lee (2005), *Marketing Science*, `doi:10.1287/mksc.1040.0098
<https://doi.org/10.1287/mksc.1040.0098>`__.

----

``GammaGamma``, the spend model
-------------------------------

*Money: what is a transaction worth?*

How much a customer spends per purchase, estimated separately from how often they
buy. A customer with only a handful of purchases isn't judged on their own average
alone. It's pulled toward the population's average, and the more purchases they
have, the more their own history wins out.

.. code:: python

   from clvkit import GammaGamma

   gg = GammaGamma().fit(cb)
   gg.predict().to_pandas()   # expected spend per transaction, per customer
   gg.population_mean()       # average spend across the whole base

.. note::

   **Assumption.** Spend per transaction is independent of purchase frequency.
   ``CLV.fit`` runs the paper's own test and warns when your base violates it.

   **Price.** ``monetary_value`` is the mean of a customer's *repeat* transactions.
   The first is excluded, because it isn't a repeat.

Fader & Hardie (2013), *The Gamma-Gamma Model of Monetary Value*, Note 025,
`brucehardie.com/notes/025 <https://www.brucehardie.com/notes/025/gamma_gamma.pdf>`__.

----

``CLV``, lifetime value
-----------------------

*Joins timing and money into one number per customer.*

Lifetime value is "how many more purchases" times "how much each is worth", with
future money discounted back to what it's worth today. ``CLV`` composes the
transaction model and the spend model and does exactly that, with no separate
joint model to fit.

.. code:: python

   from clvkit import CLV

   result = CLV().fit(cb).predict(horizon=52, discount_rate=0.001, margin=1.0)
   result.to_pandas()["clv"]   # discounted lifetime value, per customer

It only needs "expected purchases over a horizon", so any transaction model with a
``predict`` composes in. Swap BG/NBD for MBG/NBD without touching the rest:

.. code:: python

   from clvkit import CLV, MBGNBD

   CLV(transaction_model=MBGNBD()).fit(cb).predict(horizon=52, discount_rate=0.001)

.. note::

   **Assumption.** Average transaction value is independent of the transaction
   process, the same independence ``GammaGamma`` needs. Without it the product of
   two correct expectations isn't the expectation of the product.

   **Price.** ``margin`` defaults to ``1.0``, so ``predict()`` returns revenue, not
   contribution. A wrong margin scales every customer identically, so it's
   invisible. That's why the default is explicit rather than a guess.

Fader, Hardie & Lee (2005), *Journal of Marketing Research*,
`doi:10.1509/jmkr.2005.42.4.415 <https://doi.org/10.1509/jmkr.2005.42.4.415>`__.

----

``CohortMatrix``, descriptive retention
---------------------------------------

*What the log already says, before any model.*

No likelihood, no fitting, no parameters. Every cell is an observed count or sum.
Group customers by the period of their first purchase, follow each group forward on
its own clock, and you get one row per cohort: how many stayed active, or how much
they spent, period after period.

.. code:: python

   from clvkit import CohortMatrix

   retention = CohortMatrix.from_transactions(orders, period="M", metric="retention")
   retention.to_pandas(relative=True)   # retention rates, one row per cohort
   retention.plot()                     # the triangle

.. note::

   **Assumption.** None to appeal to. This is practitioner convention.

   **Price.** An unobserved cell is ``NaN``, never ``0``. "Nobody bought" and "we
   don't know yet" are different facts, and conflating them reads a young cohort as
   a churned one. The cost is that a retention matrix is float-typed even though it
   counts whole customers, because those ``NaN`` cells have to live somewhere.

Practitioner convention. The ``NaN`` policy is argued in ``opinions.md``.
