"""How a `CustomerBase` describes itself.

Two forms, because Python offers two hooks and they get used differently.
``__repr__`` is what shows up inside a list, a dict, a traceback, a failed
assertion — so it is one line. ``__str__`` is what you get when you echo the
object or print it deliberately, so it can afford to explain itself.

The long form pairs each choice with its consequence rather than listing the
provenance and the readiness separately, because they are the same facts read
twice: ``has_monetary`` being false *is* Gamma-Gamma being unavailable.

Three rules hold everywhere in here.

*Inert* — nothing below computes anything a caller has not already paid for,
and nothing copies a frame. Displays fire on every notebook echo, unasked, and
on a large base a copy per line is real latency for a glance nobody asked for.

*Total* — nothing below raises, for any attribute or any frame. A
half-constructed object is exactly what you most want to look at, and an
exception thrown from a display reads as a broken library rather than a broken
object. `_get` is the only way this module reads the base.

*ASCII* — no box drawing, no arrows, no ticks. `print(cb)` on a Windows console
encodes to cp1252, where `┌` and `→` raise `UnicodeEncodeError` — turning a
glance into a crash. Every other repr in this package is plain ASCII too.
"""

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from clvkit.customer_base import CustomerBase

_MISSING = object()

# Which models the input contract admits. The question is structural — does
# this base carry the columns the likelihood needs — and it has a certain
# answer. Whether the base is *large enough* to fit well is a different claim,
# it belongs in the notes, and conflating the two is how a display starts
# lying: a tick that means "advisable" is a tick nobody can check.
_TIMING_MODELS = ("BGNBD", "MBGNBD", "CohortSurvival")
_MONETARY_MODELS = ("GammaGamma", "CLV")

# A base where most customers bought once tells the models almost nothing
# per-customer; half is the point where that stops being a detail. A hundred
# repeat buyers is roughly where the BG/NBD parameters settle down on the
# canonical datasets. Both are judgement calls, which is exactly why they are
# notes rather than part of the `fits` verdict.
_MOSTLY_ONE_TIME = 0.5
_THIN_REPEAT_BASE = 100


def signature(cb: "CustomerBase") -> str:
    """One line: the grains, the size, the horizon. Safe to paste anywhere."""
    time_unit = _get(cb, "time_unit")
    collapse = _get(cb, "collapse")
    grain = time_unit if collapse == time_unit else f"{time_unit}/{collapse}"
    money = "$" if _get(cb, "has_monetary") else "-"
    n, n_repeat = _counts(cb)
    return (
        f"<CustomerBase[{grain}, {money}] "
        f"{n:,} customers ({n_repeat:,} repeat) "
        f"to {_date(_get(cb, 'observation_period_end'))}>"
    )


def describe(cb: "CustomerBase") -> str:
    """The long form: every choice beside what it bought you."""
    n, n_repeat = _counts(cb)
    share = f" ({n_repeat / n:.0%})" if n else ""
    has_monetary = bool(_get(cb, "has_monetary"))

    rows = [
        ("ruler", f"time_unit={_get(cb, 'time_unit')!r}", _ruler_why(cb)),
        ("grain", f"collapse={_get(cb, 'collapse')!r}", _grain_why(cb)),
        ("amounts", "yes" if has_monetary else "no", _amounts_why(has_monetary)),
        ("negatives", repr(_get(cb, "on_negative")), _negatives_why(cb)),
        (
            "observed to",
            _date(_get(cb, "observation_period_end")),
            _horizon_why(cb),
        ),
    ]

    fits = list(_TIMING_MODELS) + (list(_MONETARY_MODELS) if has_monetary else [])
    rows.append(("", "", ""))
    rows.append(("fits", "  ".join(fits), ""))
    if not has_monetary:
        rows.append(
            ("refused", ", ".join(_MONETARY_MODELS), "no spend column to model")
        )

    notes = _notes(cb, n, n_repeat)
    if notes:
        rows.append(("", "", ""))
        for i, note in enumerate(notes):
            rows.append(("note" if i == 0 else "", note, ""))

    key_width = max(len(k) for k, _, _ in rows)
    value_width = max(len(v) for _, v, why in rows if why)

    out = []
    for key, value, why in rows:
        if not key and not value:
            out.append("")
        elif why:
            out.append(
                f"  {key.rjust(key_width)}  {value.ljust(value_width)}  -> {why}"
            )
        else:
            out.append(f"  {key.rjust(key_width)}  {value}")

    header = f"CustomerBase  {n:,} customers, {n_repeat:,} repeat{share}"
    rule = "-" * max(len(line) for line in [header, *out])
    return "\n".join([header, rule, *out])


# ---------------------------------------------------------------------------
# what each choice bought you
# ---------------------------------------------------------------------------


def _ruler_why(cb: "CustomerBase") -> str:
    return f"recency and T are counted in {_get(cb, 'time_unit')!r}"


def _grain_why(cb: "CustomerBase") -> str:
    collapse, time_unit = _get(cb, "collapse"), _get(cb, "time_unit")
    if collapse == time_unit:
        return "purchases inside one period merge into one"
    return f"events kept at {collapse!r}, reported in {time_unit!r}"


def _amounts_why(has_monetary: bool) -> str:
    if has_monetary:
        return "GammaGamma and CLV are available"
    return "built with amount_col=None"


def _negatives_why(cb: "CustomerBase") -> str:
    return {
        "net": "netted per period; periods not staying positive were dropped",
        "drop": "negative rows discarded before summarising",
        "raise": "a negative amount would have refused the log",
    }.get(_get(cb, "on_negative"), "")


def _horizon_why(cb: "CustomerBase") -> str:
    span = _span(cb)
    if span is None:
        return "the as-of date every T is measured against"
    return f"{span:.0f} {_get(cb, 'time_unit')} of history at the oldest customer"


def _notes(cb: "CustomerBase", n: int, n_repeat: int) -> list[str]:
    """Only what a reader could not have worked out from the rows above.

    Deliberately silent about the ruler/grain split: the `grain` row already
    states it, and an earlier draft fired a warning here on
    ``time_unit="W", collapse="D"`` — the exact configuration this library
    recommends for the CDNOW benchmark. Flagging the intended setup as a
    problem is how a notes section teaches people to stop reading it.
    """
    notes = []
    if _get(cb, "engine") == "dask":
        notes.append(
            "summarised with engine='dask' - the event frame was not kept, "
            "so .split() is unavailable on this base"
        )
    if n and n_repeat / n < _MOSTLY_ONE_TIME:
        notes.append(
            f"{1 - n_repeat / n:.0%} bought once - the models see them only "
            "through the population, not their own history"
        )
    if n_repeat and n_repeat < _THIN_REPEAT_BASE:
        notes.append(
            f"only {n_repeat:,} repeat buyers carry the likelihood - expect "
            "wide parameter uncertainty"
        )
    return notes


# ---------------------------------------------------------------------------
# reading the base — the only place this module touches it
# ---------------------------------------------------------------------------


def _get(cb: "CustomerBase", name: str, default: Any = "?") -> Any:
    """Read one attribute, tolerating an object that does not have it yet.

    `CustomerBase.__init__` can raise part-way through — `_ruler_ratio` refuses
    an impossible grain pair — leaving a live object with some attributes set.
    That frame is exactly where somebody inspects `self`.
    """
    value = getattr(cb, name, _MISSING)
    return default if value is _MISSING else value


def _frame(cb: "CustomerBase") -> pd.DataFrame | None:
    """The summary itself, uncopied. Read-only by convention — see *Inert*."""
    frame = _get(cb, "_data", None)
    return frame if isinstance(frame, pd.DataFrame) else None


def _counts(cb: "CustomerBase") -> tuple[int, int]:
    frame = _frame(cb)
    if frame is None or "frequency" not in frame:
        return 0, 0
    try:
        return len(frame), int((frame["frequency"] > 0).sum())
    except Exception:  # noqa: BLE001 — totality: a display must not raise
        return len(frame), 0


def _span(cb: "CustomerBase") -> float | None:
    frame = _frame(cb)
    if frame is None or "T" not in frame or not len(frame):
        return None
    try:
        return float(frame["T"].max())
    except Exception:  # noqa: BLE001 — totality
        return None


def _date(value: Any) -> str:
    try:
        return str(pd.Timestamp(value).date())
    except Exception:  # noqa: BLE001 — totality: any junk still has a str()
        return str(value)
