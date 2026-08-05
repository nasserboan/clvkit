Examples
========

Start from the question you brought, not the notebook. Each run below is a
worked example on real, published data — nothing is transcribed.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Your question
     - What answers it
     - Worked example
   * - What is a customer worth?
     - ``CLV().fit().predict()``
     - :doc:`CDNOW <cdnow_clv>`
   * - Is this customer gone, or just quiet?
     - ``BGNBD().probability_alive()``
     - :doc:`Start here <start_here>`
   * - Who should get the retention budget?
     - Ranking on ``clv``, not past spend
     - :doc:`CDNOW <cdnow_clv>`
   * - Why does my retention chart disagree with Marketing's?
     - ``CohortMatrix`` and its ``NaN``\ s
     - :doc:`Online Retail II <online_retail_ii_cohort>`
   * - I already have this working in ``lifetimes``
     - The same fit, fewer moving parts
     - :doc:`From lifetimes <from_lifetimes>`

.. grid:: 1 3 3 3
   :gutter: 3

   .. grid-item-card:: Start here
      :link: start_here
      :link-type: doc

      A router, not a lesson. Four business questions, each answered on real data
      and ending in one line you could say in a meeting.

   .. grid-item-card:: CDNOW
      :link: cdnow_clv
      :link-type: doc

      The money side, end to end: raw log to lifetime value, reproducing the
      published Fader, Hardie & Lee (2005) estimates.

   .. grid-item-card:: Online Retail II
      :link: online_retail_ii_cohort
      :link-type: doc

      The descriptive side: cohort retention and revenue triangles, and why an
      unobserved cell is ``NaN`` and never ``0``.

   .. grid-item-card:: From lifetimes
      :link: from_lifetimes
      :link-type: doc

      Moving off the archived incumbent. The same four models side by side, and
      the ``freq`` split that quietly changed your fit.

.. toctree::
   :hidden:

   start_here
   cdnow_clv
   online_retail_ii_cohort
   from_lifetimes
