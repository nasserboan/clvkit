Dask, for logs over four million rows
=====================================

clvkit stays small by default: the four runtime dependencies and nothing else.
``CustomerBase.from_transactions`` is the one step whose cost scales with
transactions rather than customers, so it's the only place a second engine is
offered:

.. code-block:: bash

   uv add "clvkit[dask]"

Pass ``engine="dask"`` to ``CustomerBase.from_transactions`` with a
``dask.dataframe.DataFrame`` in place of the pandas one. The summary it returns
is an ordinary small pandas-backed ``CustomerBase`` all the same.

Where it starts to win
----------------------

Synthetic logs in the input contract, read from Parquet by both engines and
summarised at ``time_unit="W", collapse="D"`` on a 10-core M-series MacBook Pro.
Three transactions per customer, a CDNOW-shaped log:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - rows
     - pandas
     - Dask
     -
   * - 100,000
     - 0.06 s · 192 MB
     - 0.22 s · 216 MB
     - pandas 3.7× faster
   * - 1,000,000
     - 0.35 s · 474 MB
     - 0.50 s · 556 MB
     - pandas 1.4× faster
   * - 4,000,000
     - 1.99 s · 1,349 MB
     - 1.68 s · 1,698 MB
     - **Dask 1.2× faster**
   * - 16,000,000
     - 17.40 s · 4,973 MB
     - 9.59 s · 5,589 MB
     - **Dask 1.8× faster**

When to reach for it
--------------------

**The crossover is at about four million transactions.** Below it, Dask's graph
construction and shuffle cost more than the work they distribute, and pandas wins
by up to 3.7×. That's the default, and it's the right call for most logs. Above
4M, ``engine="dask"`` roughly halves the wall-clock, as long as the log is already
on disk in a format Dask reads in parallel, like Parquet or partitioned CSV.

**It won't fix running out of memory on one machine.** Dask's peak memory runs
12 to 25% *higher* than pandas' at every size measured. The default scheduler runs
its workers as threads in this same process, so the shuffle buffers and the
finished summary share one address space, and the raw log never being held whole
doesn't help when the buffers replace it. If pandas runs out of RAM, the fix is
``dask.distributed`` with workers in separate processes, which this library
doesn't configure for you.

Absolute times are machine-specific; the crossover is the part worth comparing.
Reproduce the table with ``uv run --extra dask python benchmarks/dask_crossover.py``.
