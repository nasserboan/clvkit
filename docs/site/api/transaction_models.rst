Transaction models
==================

The purchase-timing half: how many purchases to expect, and whether a customer
is still active.

.. currentmodule:: clvkit

.. autosummary::
   :toctree: generated/
   :nosignatures:

   BGNBD
   MBGNBD

``parameter_uncertainty`` on either model returns a ``ParameterUncertainty``:
one row per parameter with the estimate, its bootstrap standard error and a
percentile interval, with ``to_pandas``, ``to_json`` and ``plot``.

.. currentmodule:: clvkit.clv

.. autosummary::
   :toctree: generated/
   :nosignatures:

   ParameterUncertainty
