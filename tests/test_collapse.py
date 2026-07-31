"""`collapse` — separating the event grain from the ruler.

`time_unit` used to do two jobs at once: it set the bucket that same-period
transactions collapse into *and* the unit `recency`/`T` are measured in. Those
are different questions, and the canonical CDNOW fit answers them differently —
collapse by day (the data's own resolution), report time in weeks. With one
knob you cannot ask for that, and asking for `time_unit="W"` silently deletes
purchases instead.
"""

import warnings

import pandas as pd
import pytest

from clvkit import CustomerBase


def _log(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["customer_id", "date", "amount"])


# A customer buying twice inside one week, then again five weeks later.
WITHIN_ONE_WEEK = _log(
    [
        ("alice", "2024-01-01", 10.0),  # Monday
        ("alice", "2024-01-03", 20.0),  # Wednesday, same week
        ("alice", "2024-02-05", 30.0),  # five weeks later
    ]
)


class TestTheDefaultIsUnchanged:
    """`collapse` defaults to `time_unit`, so nothing moves for existing callers."""

    def test_omitting_collapse_matches_passing_time_unit_explicitly(self):
        implicit = CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="D")
        explicit = CustomerBase.from_transactions(
            WITHIN_ONE_WEEK, time_unit="D", collapse="D"
        )
        pd.testing.assert_frame_equal(implicit.to_pandas(), explicit.to_pandas())

    def test_the_collapse_grain_is_carried_as_provenance(self):
        cb = CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="D")
        assert cb.collapse == "D"


class TestCollapseSetsWhichPurchasesSurvive:
    def test_a_daily_collapse_keeps_two_purchases_in_one_week(self):
        cb = CustomerBase.from_transactions(
            WITHIN_ONE_WEEK, time_unit="W", collapse="D"
        )
        # Three transactions on three distinct days -> two repeat purchases.
        assert cb.to_pandas().loc["alice", "frequency"] == 2

    def test_a_weekly_collapse_merges_them(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cb = CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="W")
        # Monday and Wednesday become one event -> one repeat purchase.
        assert cb.to_pandas().loc["alice", "frequency"] == 1


class TestTheRulerIsIndependentOfTheGrain:
    def test_recency_and_T_are_reported_in_time_unit_not_collapse_grain(self):
        cb = CustomerBase.from_transactions(
            WITHIN_ONE_WEEK, time_unit="W", collapse="D"
        )
        row = cb.to_pandas().loc["alice"]
        # 2024-01-01 -> 2024-02-05 is 35 days; the ruler is weeks, so 5.
        assert row["recency"] == pytest.approx(5.0)
        assert row["T"] == pytest.approx(5.0)

    def test_a_daily_ruler_reports_the_same_span_in_days(self):
        cb = CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="D")
        assert cb.to_pandas().loc["alice", "recency"] == 35


class TestTheTrapWarns:
    def test_a_coarser_than_daily_collapse_warns_when_it_merges_purchases(self):
        with pytest.warns(UserWarning, match="collapsed"):
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="W")

    def test_the_warning_says_how_many_transactions_were_absorbed(self):
        with pytest.warns(UserWarning, match="1 of 3"):
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="W")

    def test_it_stays_quiet_when_a_coarse_grain_merges_nothing(self):
        spread = _log(
            [
                ("bob", "2024-01-01", 10.0),
                ("bob", "2024-02-05", 20.0),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CustomerBase.from_transactions(spread, time_unit="W")

    def test_a_finer_than_daily_grain_never_warns(self):
        # An hourly grain keeps *more* purchases than the canon, so there is
        # nothing to warn about — and the daily remedy would be backwards.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="h")

    def test_the_daily_default_never_warns(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="D")

    def test_naming_the_collapse_explicitly_is_taken_as_consent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="W", collapse="W")


class TestUnsupportedCombinationsAreRefused:
    def test_a_ruler_finer_than_the_grain_is_refused(self):
        with pytest.raises(ValueError, match="finer"):
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="D", collapse="W")

    def test_a_conversion_without_a_fixed_ratio_is_refused(self):
        # Months are not a whole number of days, so there is no exact ruler.
        with pytest.raises(ValueError, match="exact"):
            CustomerBase.from_transactions(WITHIN_ONE_WEEK, time_unit="M", collapse="D")


class TestTheProvenanceGuardSeesTheGrain:
    """A model fit at one grain must refuse a base summarised at another."""

    @staticmethod
    def _base(collapse: str) -> CustomerBase:
        rows = []
        for who in "abcdef":
            for day in (1, 3, 20, 45):
                rows.append(
                    (who, f"2024-01-{day:02d}" if day < 32 else "2024-02-14", 10.0)
                )
        return CustomerBase.from_transactions(
            _log(rows), amount_col=None, time_unit="W", collapse=collapse
        )

    def test_a_model_refuses_a_base_with_a_different_collapse(self):
        from clvkit import BGNBD

        model = BGNBD().fit(self._base("D"))
        with pytest.raises(ValueError, match="collapse"):
            model.predict(t=4, cb=self._base("W"))


class TestCohortSurvivalInvertsAFractionalT:
    """`CohortSurvival` recovers acquisition from `T`; a split grain makes it fractional."""

    # Friday and Saturday acquisitions, observed to a Wednesday. Chosen because
    # they are exactly where truncating `T` misfiles a customer: 2024-01-05 is
    # 5.714 weeks before 2024-02-14, and `int(5.714) == 5` puts them a week late.
    # Recovering the day first and converting afterwards is exact.
    FRIDAY = "2024-01-05"
    SATURDAY = "2024-01-13"
    OBSERVED_TO = "2024-02-14"

    @classmethod
    def _log(cls) -> pd.DataFrame:
        rows = []
        for acquired, ids in ((cls.FRIDAY, "abc"), (cls.SATURDAY, "de")):
            start = pd.Timestamp(acquired)
            for who in ids:
                # A second purchase two days later, inside the same week: only
                # a daily collapse keeps it as its own event.
                for offset in (0, 2):
                    rows.append((who, (start + pd.Timedelta(days=offset)).date(), 10.0))
        return _log(rows)

    def test_customers_land_in_the_week_they_were_acquired_in(self):
        from clvkit import CohortSurvival

        cb = CustomerBase.from_transactions(
            self._log(),
            amount_col=None,
            time_unit="W",
            collapse="D",
            observation_period_end=self.OBSERVED_TO,
        )
        curve = CohortSurvival().fit(cb).predict().to_pandas()

        assert [str(c) for c in curve.index] == [
            "2024-01-01/2024-01-07",  # the week containing Friday the 5th
            "2024-01-08/2024-01-14",  # the week containing Saturday the 13th
        ]
        assert list(curve["customers"]) == [3, 2]

    def test_a_daily_cohort_grain_is_allowed_when_the_collapse_is_daily(self):
        from clvkit import CohortSurvival

        cb = CustomerBase.from_transactions(
            self._log(), amount_col=None, time_unit="W", collapse="D"
        )
        # The ruler is weekly, but acquisitions are dated to the day.
        curve = CohortSurvival().fit(cb).predict(period="D").to_pandas()
        assert len(curve) == 2


class TestSplitRespectsTheGrain:
    def test_holdout_duration_is_reported_in_time_unit(self):
        log = _log(
            [
                ("alice", "2024-01-01", 10.0),
                ("alice", "2024-01-03", 20.0),
                ("alice", "2024-02-05", 30.0),
                ("bob", "2024-01-02", 15.0),
            ]
        )
        cb = CustomerBase.from_transactions(log, time_unit="W", collapse="D")
        _, holdout = cb.split(calibration_period_end="2024-01-15")
        # 2024-01-15 -> 2024-02-05 is three whole weeks.
        assert holdout["duration_holdout"].iloc[0] == pytest.approx(3.0)
