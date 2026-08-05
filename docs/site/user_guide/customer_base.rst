CustomerBase, the shared contract
=================================

Every model in clvkit reads the same object. ``BGNBD``, ``MBGNBD``,
``GammaGamma`` and ``CLV`` don't each parse your transaction log their own way.
They take a ``CustomerBase``, the one place a raw log becomes an RFM summary, and
fit on that.

.. code-block:: python

   from clvkit import CustomerBase, CLV

   cb = CustomerBase.from_transactions(log, time_unit="W", collapse="D")
   CLV().fit(cb).predict(horizon=52, discount_rate=0.001)

Three columns go in, defaulting to ``customer_id``, ``date`` and ``amount``, one
row per purchase. Everything else in the frame is ignored. Building the summary
once and handing the same object to every model is the point. The alternative,
each model re-deriving recency and frequency from the raw log, is where two
models in the same script quietly disagree about what a customer's recency is.

The philosophy: refuse rather than guess
----------------------------------------

The contract has opinions, and each states its price out loud.

**Naming the grain is consent to it.** ``time_unit`` is the clock the parameters
are reported in; ``collapse`` is the grain at which same-period purchases become
one event. Leave ``collapse`` off and it inherits ``time_unit``, which is the most
common way to fit a defensible-looking model to the wrong data. ``time_unit="W"``
alone doesn't re-scale the ruler, it deletes every second purchase inside a week,
most of them from your heaviest buyers. So the summary makes you name the grain
instead of inheriting it in silence.

**It won't invent a ruler it doesn't have.** Report weeks off daily events and it
converts cleanly, seven days to a week. Report months and it raises, because a
month is 28 to 31 days and there's no exact day-to-month conversion. It refuses
rather than pick an average and call it precision.

**Negative amounts are your call.** Returns and refunds arrive as negative rows.
``on_negative`` names the choice: ``"net"``, the default, lets them cancel spend,
``"drop"`` discards them, and ``"raise"`` stops so you can look before deciding.

The cost is a step you can't skip. You build the ``CustomerBase`` first, and you
answer its questions about grain and signs before any model reads your data.
That's the trade, a little ceremony up front against two models disagreeing in
silence later.

See :doc:`../api/core` for the full signature.
