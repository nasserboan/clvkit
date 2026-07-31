"""What a `CustomerBase` says about itself.

The RFM table is the generic part — pandas already prints that. What only
this library can say is which knobs were set and what they bought you, so
the long form pairs each choice with its consequence and the short form is
a signature you can paste into a log line.
"""

import pandas as pd
import pytest

from clvkit import CustomerBase

LOG = pd.DataFrame(
    [
        ("alice", "2024-01-01", 10.0),
        ("alice", "2024-01-03", 20.0),
        ("alice", "2024-02-05", 30.0),
        ("bob", "2024-01-02", 15.0),
        ("bob", "2024-03-01", 25.0),
        ("carol", "2024-01-08", 40.0),
    ],
    columns=["customer_id", "date", "amount"],
)


@pytest.fixture
def cb() -> CustomerBase:
    return CustomerBase.from_transactions(LOG, time_unit="W", collapse="D")


class TestTheShortFormIsASignature:
    def test_repr_fits_on_one_line(self, cb):
        assert "\n" not in repr(cb)

    def test_it_names_both_grains_when_they_differ(self, cb):
        assert "W/D" in repr(cb)

    def test_it_names_one_grain_when_they_agree(self):
        base = CustomerBase.from_transactions(LOG, time_unit="D")
        assert "[D" in repr(base)
        assert "D/D" not in repr(base)

    def test_it_carries_the_counts_and_the_observation_end(self, cb):
        text = repr(cb)
        assert "3 customers" in text
        assert "2 repeat" in text
        assert "2024-03-01" in text


class TestTheLongFormPairsChoiceWithConsequence:
    def test_every_provenance_field_appears(self, cb):
        text = str(cb)
        for token in ("time_unit='W'", "collapse='D'", "'net'", "2024-03-01"):
            assert token in text, token

    def test_having_amounts_is_stated_as_what_it_unlocks(self, cb):
        assert "GammaGamma" in str(cb)

    def test_lacking_amounts_is_stated_as_what_it_blocks(self):
        base = CustomerBase.from_transactions(
            LOG, amount_col=None, time_unit="W", collapse="D"
        )
        text = str(base)
        assert "amount_col=None" in text
        assert "CLV" in text


class TestReadinessAnswersOnlyTheStructuralQuestion:
    """`fits` is a fact about the contract, never a judgement about sample size."""

    def test_a_timing_only_base_cannot_fit_the_monetary_models(self):
        base = CustomerBase.from_transactions(
            LOG, amount_col=None, time_unit="W", collapse="D"
        )
        fits = _fits_line(str(base))
        assert "BGNBD" in fits
        # Named, but on the refusal row rather than the tick list.
        assert "GammaGamma" not in fits.splitlines()[0]
        assert "refused" in fits

    def test_a_monetary_base_fits_everything(self, cb):
        fits = _fits_line(str(cb))
        for model in ("BGNBD", "MBGNBD", "GammaGamma", "CLV", "CohortSurvival"):
            assert model in fits, model

    def test_a_tiny_base_still_fits_structurally(self, cb):
        # Three customers is far too few to fit well, but "too few to be wise"
        # is a different claim from "cannot be fitted" — the first is a note,
        # the second is the verdict, and conflating them is how a display
        # starts lying.
        assert "BGNBD" in _fits_line(str(cb))

    def test_a_thin_base_says_so_as_a_note_not_as_a_verdict(self, cb):
        # "2 repeat" also appears in the header, so target the note block.
        assert "carry the likelihood" in _note_block(str(cb))


def _big(*, repeat_share: float, n: int = 400) -> pd.DataFrame:
    """A log where a chosen share of customers buy twice and the rest once."""
    rows = []
    for i in range(n):
        rows.append((f"c{i}", "2024-01-01", 10.0))
        if i < round(n * repeat_share):
            rows.append((f"c{i}", "2024-03-01", 10.0))
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


class TestNotesFireOnlyWhenThereIsSomethingToSay:
    def test_a_mostly_one_time_base_is_noted(self):
        base = CustomerBase.from_transactions(_big(repeat_share=0.2), time_unit="D")
        assert "80% bought once" in _note_block(str(base))

    def test_a_mostly_repeat_base_is_not(self):
        base = CustomerBase.from_transactions(_big(repeat_share=0.9), time_unit="D")
        assert "bought once" not in _note_block(str(base))

    def test_a_thin_repeat_base_is_noted(self):
        base = CustomerBase.from_transactions(
            _big(repeat_share=0.9, n=60), time_unit="D"
        )
        assert "54 repeat buyers carry the likelihood" in _note_block(str(base))

    def test_a_deep_repeat_base_is_not(self):
        base = CustomerBase.from_transactions(_big(repeat_share=0.9), time_unit="D")
        assert "carry the likelihood" not in _note_block(str(base))

    def test_the_grain_split_is_not_noted_because_the_grain_row_says_it(self, cb):
        # An earlier draft fired here on time_unit="W", collapse="D" — the
        # configuration this library recommends for its own benchmark.
        assert "7:1" not in str(cb)
        assert "events kept at 'D'" in str(cb)

    def test_a_clean_base_carries_no_note_section(self):
        clean = pd.DataFrame(
            [
                (f"c{i}", d, 10.0)
                for i in range(150)
                for d in ("2024-01-01", "2024-02-01")
            ],
            columns=["customer_id", "date", "amount"],
        )
        base = CustomerBase.from_transactions(clean, time_unit="D")
        assert "note" not in str(base)


class TestTheDisplayIsInertAndTotal:
    """Guardrails: a repr that computes or raises is worse than no repr."""

    def test_it_never_touches_the_underlying_frame(self, cb):
        before = cb.to_pandas()
        str(cb)
        repr(cb)
        pd.testing.assert_frame_equal(before, cb.to_pandas())

    def test_it_never_copies_the_frame(self, cb, monkeypatch):
        # `to_pandas()` copies. On a large base that is real latency charged
        # to every notebook echo, for a glance nobody asked for.
        calls = []
        monkeypatch.setattr(
            type(cb), "to_pandas", lambda self: calls.append(1) or pd.DataFrame()
        )
        str(cb)
        repr(cb)
        assert calls == []

    def test_a_half_constructed_base_still_renders(self):
        # `__init__` raises part-way through on an impossible grain pair,
        # leaving a live object with some attributes set. That frame is
        # exactly where somebody inspects `self`.
        half = CustomerBase.__new__(CustomerBase)
        half._data = pd.DataFrame({"frequency": [1, 0]})
        half.time_unit = "W"
        assert "CustomerBase" in repr(half)
        assert "CustomerBase" in str(half)

    def test_an_object_with_nothing_set_still_renders(self):
        assert "CustomerBase" in repr(CustomerBase.__new__(CustomerBase))
        assert "CustomerBase" in str(CustomerBase.__new__(CustomerBase))

    @pytest.mark.parametrize("form", [repr, str])
    def test_output_survives_a_windows_console(self, cb, form):
        # cp1252 is the Windows console default, and it has no box drawing,
        # no arrows and no ticks — `print(cb)` there would raise.
        form(cb).encode("cp1252")

    def test_a_base_built_without_a_log_still_renders(self):
        # Constructed directly, so `_events` is None and `split()` would raise.
        # Display must not.
        direct = CustomerBase(
            pd.DataFrame({"frequency": [1], "recency": [2.0], "T": [3.0]}, index=["z"]),
            time_unit="W",
            observation_period_end=pd.Timestamp("2024-03-01"),
            has_monetary=False,
            on_negative="net",
        )
        assert "CustomerBase" in repr(direct)
        assert "CustomerBase" in str(direct)

    def test_an_empty_base_renders_rather_than_dividing_by_zero(self):
        empty = CustomerBase(
            pd.DataFrame({"frequency": [], "recency": [], "T": []}),
            time_unit="D",
            observation_period_end=pd.Timestamp("2024-01-01"),
            has_monetary=False,
            on_negative="net",
        )
        assert "0 customers" in repr(empty)
        assert "CustomerBase" in str(empty)


def _note_block(text: str) -> str:
    """Everything from the `note` label onward."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("note")), None
    )
    return "" if start is None else "\n".join(lines[start:])


def _fits_line(text: str) -> str:
    """The `fits` row plus the continuation rows hanging under it."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if "fits" in ln)
    out = [lines[start]]
    for ln in lines[start + 1 :]:
        if "note" in ln or ln.strip() in ("│", "└", ""):
            break
        out.append(ln)
    return "\n".join(out)
