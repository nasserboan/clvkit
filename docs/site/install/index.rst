Install
=======

.. code-block:: bash

   uv add clvkit     # or pip install clvkit

Python 3.13+, four runtime dependencies — ``numpy``, ``scipy``, ``pandas`` and
``matplotlib`` — and no compiler. If ``uv`` is new to you, it's the fastest way
in; ``pip install clvkit`` gets you the same package.

Verify
------

.. code-block:: python

   import clvkit

   clvkit.__version__

The :doc:`dask` extra is the one optional install, for summarising transaction
logs past about four million rows. From here, the :doc:`../user_guide/index` walks
through your first fit, or the :doc:`../examples/index` show it on real data.

.. toctree::
   :hidden:

   dask
