:html_theme.sidebar_secondary.remove:

clvkit
======

.. rst-class:: clv-tagline

   Four parameters. Six decimal places. No trace to read.

Buy-till-you-die models for a business where nobody cancels a subscription, so
you never observe a customer leaving — you only ever observe them not coming back
*yet*. Six lines take a transaction log to a per-customer CLV table, in 2.0 s
of wall clock on the CDNOW sample. Python 3.11+, four dependencies, no compiler.

.. code-block:: bash

   uv add clvkit     # or pip install clvkit

.. image:: _static/clv-scatter.png
   :alt: CLV against discounted transactions and expected spend
   :align: center
   :width: 80%

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 4 0 0 0

   .. grid-item-card:: :octicon:`download` Install
      :link: install/index
      :link-type: doc

      One line, four dependencies, no compiler. Get it into your environment.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide/index
      :link-type: doc

      The key ideas, the four models, and the calls clvkit makes on your behalf.

   .. grid-item-card:: :octicon:`code-square` API Reference
      :link: api/index
      :link-type: doc

      Every class and method, documented from the source.

   .. grid-item-card:: :octicon:`beaker` Examples
      :link: examples/index
      :link-type: doc

      Three worked runs on real data, each answering a question a business asks.

It lands where the paper said it would
--------------------------------------

The BG/NBD model comes from a 2005 paper by Fader, Hardie and Lee in *Marketing
Science*. They fit it on the CDNOW 1/10 systematic sample, 2,357 customers with a
39-week calibration period, and printed the four fitted parameters. clvkit fits
the same model on the same data and recovers the same four numbers, to the last
printed digit.

CDNOW is the yardstick, not the boundary. clvkit fits any transaction log that
carries a customer id, a date and an amount, so point it at your own data and the
four models run the same way.

.. list-table::
   :header-rows: 1

   * - What it is
     - Parameter
     - Published, 2005
     - ``clvkit``
   * - Gamma shape for how the purchase rate varies across customers
     - ``r``
     - 0.243
     - 0.242595
   * - Gamma scale for that purchase rate
     - ``α``
     - 4.414
     - 4.413602
   * - Beta shape for the dropout probability after a purchase
     - ``a``
     - 0.793
     - 0.792922
   * - Beta shape for that same dropout probability
     - ``b``
     - 2.426
     - 2.425907

.. toctree::
   :hidden:
   :maxdepth: 1

   install/index
   user_guide/index
   examples/index
   api/index
   The shrug <user_guide/the_shrug>
