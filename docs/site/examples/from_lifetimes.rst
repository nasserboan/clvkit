From lifetimes
==============

If you have working ``lifetimes`` code, the same fit is shorter here, and the two
knobs that quietly change the answer are named instead of hidden. ``lifetimes`` is
archived and takes no new releases; clvkit fits the same four models on Python
3.13.

The whole migration, side by side
----------------------------------

``lifetimes`` makes you build the RFM summary, then thread ``frequency``,
``recency`` and ``T`` through every call by hand:

.. code-block:: python

   from lifetimes import BetaGeoFitter, GammaGammaFitter
   from lifetimes.utils import summary_data_from_transaction_data

   summary = summary_data_from_transaction_data(
       log, "customer_id", "date", monetary_value_col="amount", freq="W",
   )

   bgf = BetaGeoFitter(penalizer_coef=0.0)
   bgf.fit(summary["frequency"], summary["recency"], summary["T"])

   returning = summary[summary["frequency"] > 0]
   ggf = GammaGammaFitter(penalizer_coef=0.0)
   ggf.fit(returning["frequency"], returning["monetary_value"])

   clv = ggf.customer_lifetime_value(
       bgf,
       summary["frequency"], summary["recency"], summary["T"],
       summary["monetary_value"],
       time=52, freq="W", discount_rate=0.001,
   )

clvkit builds the summary once, as a ``CustomerBase``, and every model reads it:

.. code-block:: python

   from clvkit import CustomerBase, CLV

   cb = CustomerBase.from_transactions(log, time_unit="W", collapse="D")
   clv = CLV().fit(cb).predict(horizon=52, discount_rate=0.001).to_pandas()

Call by call
------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - ``lifetimes``
     - ``clvkit``
   * - ``summary_data_from_transaction_data(log, "customer_id", "date", monetary_value_col="amount", freq="W")``
     - ``CustomerBase.from_transactions(log, time_unit="W", collapse="D")``
   * - ``BetaGeoFitter().fit(frequency, recency, T)``
     - ``BGNBD().fit(cb)``
   * - ``ModifiedBetaGeoFitter().fit(frequency, recency, T)``
     - ``MBGNBD().fit(cb)``
   * - ``GammaGammaFitter().fit(frequency, monetary_value)``
     - ``GammaGamma().fit(cb)``
   * - ``bgf.conditional_probability_alive(frequency, recency, T)``
     - ``bg.probability_alive()``
   * - ``bgf.conditional_expected_number_of_purchases_up_to_time(t, frequency, recency, T)``
     - ``bg.predict(t)``
   * - ``ggf.conditional_expected_average_profit(frequency, monetary_value)``
     - ``gg.predict()``
   * - ``ggf.customer_lifetime_value(bgf, frequency, recency, T, monetary_value, time=52, freq="W", discount_rate=0.001)``
     - ``CLV().fit(cb).predict(horizon=52, discount_rate=0.001)``

What changes
------------

**One object, not three arrays.** ``lifetimes`` hands you ``frequency``,
``recency`` and ``T`` and trusts you to pass the right one to every call. Line
them up wrong and the model still fits, on the wrong data, without a word.
``CustomerBase`` holds them together, so there's nothing to line up.

**The reporting unit and the event grain are separate.** ``lifetimes`` folds both
into one ``freq``. clvkit splits them: ``time_unit`` is the clock the parameters
are reported in, ``collapse`` is the grain at which same-period purchases count as
one event. ``freq="W"`` in lifetimes silently deletes every second purchase inside
a week; ``time_unit="W", collapse="D"`` keeps them and reports in weeks, the pair
that reproduces the published CDNOW fit. Get it wrong and ``α`` moves 55%, from
4.41 to 6.85.

**Swapping the transaction model is one line.** ``BGNBD`` to ``MBGNBD`` is a
one-word change on the same ``cb``, with nothing re-summarised. In ``lifetimes``
you re-thread the arrays into a different fitter.

See :doc:`../user_guide/customer_base` for the contract these all read, or
:doc:`cdnow_clv` for the CDNOW fit end to end.
