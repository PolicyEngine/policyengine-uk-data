"""Council tax is the bill after discounts and before council tax reduction.

The FRS CTANNUAL is net of the reported reduction (CTREB, CTREBAMT), and in
Scotland it also carries water and sewerage charges. ``derive_council_tax``
adds the reported reduction back and nets the Scottish charges off. The
fixtures below are synthetic households, not survey records.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policyengine_uk_data.datasets.frs import (
    SCOTLAND_GVTREGNO,
    SCOTTISH_WATER_CHARGES_MAXIMUM_REDUCTION,
    WEEKS_IN_YEAR,
    derive_council_tax,
)

LONDON = 7
WALES = 11
NORTHERN_IRELAND = 13
BAND_C = 3
RECIPIENT, NON_RECIPIENT = 1, 2


def _households(rows: list[dict]) -> pd.DataFrame:
    """Raw-FRS-shaped household table; unspecified fields are blank."""
    columns = [
        "gvtregno",
        "ctband",
        "adulth",
        "ctannual",
        "ctreb",
        "ctrebamt",
        "cwatamt1",
        "csewamt1",
        "cwatamtd",
        "csewamt",
    ]
    table = pd.DataFrame(rows).reindex(columns=columns).astype(float)
    table.index = pd.RangeIndex(100, 100 + len(table), name="household_id")
    return table


def _cell(region, ctannual, ctreb=NON_RECIPIENT, ctrebamt=np.nan, adults=2, **kw):
    return dict(
        gvtregno=region,
        ctband=BAND_C,
        adulth=adults,
        ctannual=ctannual,
        ctreb=ctreb,
        ctrebamt=ctrebamt,
        **kw,
    )


def test_full_reduction_recipient_with_zero_bill_gets_gross_bill():
    households = _households(
        [_cell(LONDON, 0.0, RECIPIENT, 30.0), _cell(LONDON, 1_600.0)]
    )
    council_tax = derive_council_tax(households, 2024)
    assert council_tax[0] == pytest.approx(30.0 * WEEKS_IN_YEAR)


def test_partial_reduction_recipient_gets_bill_plus_reduction():
    households = _households(
        [_cell(WALES, 900.0, RECIPIENT, 12.0), _cell(WALES, 1_700.0)]
    )
    council_tax = derive_council_tax(households, 2024)
    assert council_tax[0] == pytest.approx(900.0 + 12.0 * WEEKS_IN_YEAR)


def test_non_recipients_keep_their_bill_including_zero():
    households = _households(
        [
            _cell(LONDON, 1_500.0),
            _cell(LONDON, 0.0),  # e.g. an exempt dwelling
            _cell(WALES, 1_200.0, adults=1),
        ]
    )
    council_tax = derive_council_tax(households, 2024)
    np.testing.assert_allclose(council_tax, [1_500.0, 0.0, 1_200.0])


def test_recipient_with_unknown_reduction_gets_non_recipient_cell_mean():
    households = _households(
        [
            _cell(LONDON, 0.0, RECIPIENT, np.nan),
            _cell(LONDON, 0.0, RECIPIENT, 0.0),
            _cell(LONDON, 1_900.0, RECIPIENT, np.nan),
            _cell(LONDON, 1_400.0),
            _cell(LONDON, 1_600.0),
        ]
    )
    council_tax = derive_council_tax(households, 2024)
    # The cell mean (1,500), or the reported bill if that is larger: the
    # bill before the reduction cannot be below the bill after it.
    np.testing.assert_allclose(council_tax[:3], [1_500.0, 1_500.0, 1_900.0])


def test_missing_bills_are_imputed_from_non_recipients_only():
    households = _households(
        [
            _cell(LONDON, np.nan),
            _cell(LONDON, -1.0),
            _cell(LONDON, 1_400.0),
            _cell(LONDON, 1_600.0),
            # A recipient's net bill must not pull the imputed bill down.
            _cell(LONDON, 200.0, RECIPIENT, 25.0),
            # A different cell (single adult) must not enter the mean.
            _cell(LONDON, 1_125.0, adults=1),
        ]
    )
    council_tax = derive_council_tax(households, 2024)
    np.testing.assert_allclose(council_tax[:2], [1_500.0, 1_500.0])


def test_northern_ireland_has_no_council_tax():
    households = _households([_cell(NORTHERN_IRELAND, np.nan, RECIPIENT, 10.0)])
    assert derive_council_tax(households, 2024)[0] == 0.0


def test_scottish_gross_water_and_sewerage_are_netted_from_2024():
    water, sewerage = 4.0, 5.0
    gross_charges = (water + sewerage) * WEEKS_IN_YEAR
    households = _households(
        [
            _cell(SCOTLAND_GVTREGNO, 2_000.0, cwatamt1=water, csewamt1=sewerage),
            _cell(
                SCOTLAND_GVTREGNO,
                900.0,
                RECIPIENT,
                10.0,
                cwatamt1=water,
                csewamt1=sewerage,
            ),
        ]
    )
    council_tax = derive_council_tax(households, 2024)
    assert council_tax[0] == pytest.approx(2_000.0 - gross_charges)
    # Recipients' CTANNUAL carries the charges after the Water Charges
    # Reduction Scheme's maximum 35% reduction.
    recipient_charges = gross_charges * (1 - SCOTTISH_WATER_CHARGES_MAXIMUM_REDUCTION)
    assert council_tax[1] == pytest.approx(
        900.0 - recipient_charges + 10.0 * WEEKS_IN_YEAR
    )


def test_scotland_before_2024_keeps_discounted_charge_netting():
    households = _households(
        [
            _cell(
                SCOTLAND_GVTREGNO,
                2_000.0,
                cwatamtd=3.0,
                csewamt=4.0,
                cwatamt1=9.0,
                csewamt1=9.0,
            )
        ]
    )
    council_tax = derive_council_tax(households, 2023)
    assert council_tax[0] == pytest.approx(2_000.0 - 7.0 * WEEKS_IN_YEAR)


def test_charges_are_not_netted_outside_scotland():
    households = _households([_cell(LONDON, 2_000.0, cwatamt1=9.0, csewamt1=9.0)])
    assert derive_council_tax(households, 2024)[0] == pytest.approx(2_000.0)


def test_without_reduction_columns_the_bill_is_ctannual():
    households = _households(
        [_cell(LONDON, 700.0, RECIPIENT, 20.0), _cell(LONDON, np.nan)]
    ).drop(columns=["ctreb", "ctrebamt"])
    council_tax = derive_council_tax(households, 2020)
    np.testing.assert_allclose(council_tax, [700.0, 700.0])


# Property-based checks over arbitrary small household tables.

_amount = st.one_of(
    st.none(),
    st.just(-1.0),
    st.just(0.0),
    st.floats(min_value=0.0, max_value=5_000.0),
)
_weekly = st.one_of(st.none(), st.just(0.0), st.floats(min_value=0.0, max_value=100.0))
_household = st.fixed_dictionaries(
    dict(
        gvtregno=st.sampled_from([1, LONDON, WALES, SCOTLAND_GVTREGNO, 13]),
        ctband=st.one_of(st.none(), st.integers(1, 9)),
        adulth=st.integers(1, 4),
        ctannual=_amount,
        ctreb=st.one_of(st.none(), st.sampled_from([RECIPIENT, NON_RECIPIENT])),
        ctrebamt=_weekly,
        cwatamt1=_weekly,
        csewamt1=_weekly,
        cwatamtd=_weekly,
        csewamt=_weekly,
    )
)


@settings(max_examples=300, deadline=None)
@given(
    rows=st.lists(_household, min_size=1, max_size=12),
    year=st.sampled_from([2022, 2023, 2024, 2025]),
)
def test_council_tax_invariants(rows, year):
    households = _households(rows)
    council_tax = derive_council_tax(households, year)
    ctannual = households.ctannual.to_numpy()
    reduction = households.ctrebamt.to_numpy()
    recipient = (households.ctreb == 1).to_numpy()
    known_reduction = recipient & (households.ctrebamt > 0).to_numpy()
    has_bill = ctannual >= 0
    scotland = (households.gvtregno == SCOTLAND_GVTREGNO).to_numpy()

    # One finite, non-negative bill per household.
    assert council_tax.shape == (len(households),)
    assert np.isfinite(council_tax).all() and (council_tax >= 0).all()

    # Non-recipients with a bill outside Scotland keep it exactly.
    keep = ~recipient & has_bill & ~scotland
    np.testing.assert_allclose(council_tax[keep], ctannual[keep])

    # A known reduction is added back in full, so recipients outside
    # Scotland pay CTANNUAL plus it.
    add_back = known_reduction & has_bill & ~scotland
    np.testing.assert_allclose(
        council_tax[add_back],
        ctannual[add_back] + reduction[add_back] * WEEKS_IN_YEAR,
    )

    # The bill before the reduction is never below the bill after it.
    no_charges = has_bill & recipient & ~scotland
    assert (council_tax[no_charges] >= ctannual[no_charges] - 1e-9).all()

    # Recomputing on a subset of rows cannot change the kept bills (the
    # cell means only feed missing bills and unknown reductions).
    council_tax_keep_only = derive_council_tax(households[keep], year)
    np.testing.assert_allclose(council_tax_keep_only, council_tax[keep])


@settings(max_examples=200, deadline=None)
@given(rows=st.lists(_household, min_size=1, max_size=12))
def test_recipients_do_not_move_the_imputed_bill(rows):
    """Changing recipients' bills never changes a non-recipient's result."""
    households = _households(rows)
    recipient = (households.ctreb == 1).to_numpy()
    changed = households.copy()
    changed.loc[recipient, "ctannual"] = 4_321.0
    np.testing.assert_allclose(
        derive_council_tax(households, 2024)[~recipient],
        derive_council_tax(changed, 2024)[~recipient],
    )
