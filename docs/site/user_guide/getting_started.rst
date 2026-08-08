Getting started
===============

From an empty environment to a per-customer CLV table in one sitting. No
API keys, no compiler, no sampler.

Install
-------

.. code:: bash

   uv add clvkit     # or pip install clvkit

Python 3.11+, four runtime dependencies. If ``uv`` is new to you, it's
the fastest way in; ``pip install clvkit`` gets you the same package.

The one contract
----------------

Every entry point reads the same three columns, a transaction log with one
row per purchase:

=============== ==========================
Column          Meaning
=============== ==========================
``customer_id`` who bought
``date``        when (a real date, parsed)
``amount``      how much
=============== ==========================

Anything else in the frame is ignored. That's the whole contract. If
your data is line items rather than orders, aggregate to one row per
basket first. One purchase is one event, not forty.

Your first fit
--------------

.. code:: python

   import pandas as pd

   from clvkit import CLV, CustomerBase

   log = pd.read_csv("transactions.csv", parse_dates=["date"])

   cb = CustomerBase.from_transactions(log, time_unit="W", collapse="D")
   result = CLV().fit(cb).predict(horizon=52, discount_rate=0.001)

   result.to_pandas()   # expected_purchases, discounted_expected_transactions,
                        # expected_spend, clv, indexed by customer_id
   result.plot()

``CustomerBase.from_transactions`` turns the raw log into the
recency/frequency/T summary the models fit on. ``CLV`` composes a
transaction model (BG/NBD) and a spend model (Gamma-Gamma) and discounts
the result. ``predict`` returns a frame you can write straight to CSV.

The two arguments that move the numbers
---------------------------------------

``time_unit`` and ``collapse`` are separate on purpose, and getting them
wrong is the most common way to fit a defensible-looking model to the
wrong data.

- **``time_unit``** is the clock the parameters are expressed in,
  ``"W"`` for weeks or ``"D"`` for days.
- **``collapse``** is the grain at which same-period purchases become
  one event. ``collapse="D"`` folds a customer's same-day purchases into
  a single transaction, which is what the buy-till-you-die literature
  assumes.

Drop ``collapse="D"`` with ``time_unit="W"`` and ``α`` moves 55%, from
4.41 to 6.85, because same-day purchases stay separate instead of
collapsing. The fit still converges. It's just answering a different
question than the paper's.

The question you probably came for
----------------------------------

CLV ranks customers by future value. The sharper question in a
non-contractual business (where nobody cancels, so you never see a
customer leave) is whether a given customer is *gone or just quiet*:

.. code:: python

   from clvkit import BGNBD

   bg = BGNBD().fit(cb)
   bg.probability_alive().to_pandas()   # P(still active) per customer

That's the number ranking by last year's spend can't reach. A customer
who spent heavily and then went silent scores high on history and low on
``probability_alive``.

Where to go next
----------------

- :doc:`Examples </examples/index>`, three worked runs on real data, each
  starting from a business question.
- :doc:`Models <models>`, what each model estimates, its assumption,
  and the price it charges.
- `opinions.md <https://github.com/nasserboan/clvkit/blob/main/opinions.md>`__,
  the calls this library makes on your behalf, each with its price.
