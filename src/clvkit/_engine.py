"""Which dataframe library summarises the log.

Summarising is the only step in this library whose cost scales with
*transactions* rather than with customers, so it is the only step that runs out
of memory first, and the only one Dask is offered for. Everything downstream
works on four numpy arrays of length n_customers — small, and sequential in the
optimiser — so there is nothing there for a cluster to do.

Dask is an optional extra. It is imported here, inside the call, so that
``import clvkit`` stays a four-dependency import for everyone who never asks
for it.
"""

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from types import ModuleType

Engine = Literal["pandas", "dask"]
_ENGINES: tuple[Engine, ...] = ("pandas", "dask")

_MISSING = (
    "engine='dask' needs the optional dask extra, which is not installed. "
    "Install it with `pip install 'clvkit[dask]'` (or `uv sync --extra dask`), "
    "or leave engine='pandas'."
)

_NOT_DASK = (
    "engine='dask' expects a dask DataFrame, got {kind}. A pandas frame is "
    "already in memory, so wrapping it in Dask buys nothing — read the log "
    "with dask.dataframe.read_csv / read_parquet instead, or leave "
    "engine='pandas'."
)


def resolve(engine: str, transactions: object) -> "ModuleType | None":
    """Validate `engine`; return the `dask.dataframe` module, or None for pandas.

    Raises before any work starts, and names the extra rather than letting an
    ImportError surface from three frames down. The return doubles as the flag
    every caller branches on, which is why nothing else in this package has to
    ask whether dask is installed.
    """
    if engine not in _ENGINES:
        raise ValueError(f"engine must be one of {_ENGINES}, got {engine!r}")
    if engine == "pandas":
        return None

    try:
        import dask.dataframe as dd
    except ImportError as exc:  # pragma: no cover — needs dask uninstalled
        raise ImportError(_MISSING) from exc

    if not isinstance(transactions, dd.DataFrame):
        raise TypeError(_NOT_DASK.format(kind=type(transactions).__name__))
    return dd


def to_datetime(dates, dd: "ModuleType | None", *, format: str | None):
    """Parse a date column with whichever engine is in play."""
    return (pd if dd is None else dd).to_datetime(dates, format=format)


def buckets(dates, freq: str, dd: "ModuleType | None"):
    """Stamp each timestamp with the `freq` period it falls in.

    Periods under pandas, bare period *ordinals* under Dask, whose coverage of
    the period dtype is thin. Nothing downstream notices: the summary reads the
    ordinal out of either form with `astype("int64")`. The ordinal is absolute,
    so converting one partition at a time cannot disagree with converting the
    whole frame.
    """
    if dd is None:
        return dates.dt.to_period(freq)
    return dates.map_partitions(
        lambda s: s.dt.to_period(freq).astype("int64"), meta=("_bucket", "int64")
    )


def counts(values: tuple, dd: "ModuleType | None") -> tuple[int, ...]:
    """Materialise count expressions, sharing one pass over a Dask log.

    The callers batching these are deciding whether a policy warning fires;
    computing each count separately would re-read the log once per number.
    Under pandas the values are already numbers and only get cast.
    """
    if dd is None:
        return tuple(int(v) for v in values)
    import dask

    return tuple(int(v) for v in dask.compute(*values))


def row_counts(frames: tuple, dd: "ModuleType | None") -> tuple[int, ...]:
    """How many rows in each frame, in one pass over the data.

    Counting two frames separately would read a Dask log twice, and the caller
    that needs this is comparing a log against its own deduplication — two
    frames off the same graph, so one `compute` shares the read.
    """
    if dd is None:
        return tuple(len(frame) for frame in frames)
    import dask

    return dask.compute(*(frame.shape[0] for frame in frames))
