"""FRS household categorical codes against the UKDS data dictionaries.

Value labels are from the UKDS FRS variable listings for 2022-23 (SN 9252),
2023-24 (SN 9367) and 2024-25 (SN 9563), household table.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from policyengine_uk.variables.input.council_tax_band import CouncilTaxBand

from policyengine_uk_data.datasets.frs import (
    COUNCIL_TAX_PLACEHOLDER_BAND,
    derive_council_tax_band_from_frs,
    map_frs_codes,
)

# CTBAND labels, identical in all three releases except code 9 ("Band I" to
# 2023-24, "Band I - Wales only" in 2024-25) and code 10.
CTBAND_BANDS = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
}
# Code 10: "Household not valued separately" (2022-23, 2023-24),
# "Information not found" (2024-25).
CTBAND_NO_BAND = 10


@pytest.mark.parametrize("code, band", CTBAND_BANDS.items())
def test_ctband_codes_1_to_9_are_bands_a_to_i(code, band):
    assert derive_council_tax_band_from_frs([code]).tolist() == [band]


def test_ctband_10_takes_the_policyengine_uk_default_band():
    assert derive_council_tax_band_from_frs([CTBAND_NO_BAND]).tolist() == ["D"]
    assert COUNCIL_TAX_PLACEHOLDER_BAND == CouncilTaxBand.D.name


def test_blank_and_missing_value_ctband_take_the_same_band_as_code_10():
    # A blank band used to become A while code 10 became D.
    assert derive_council_tax_band_from_frs([np.nan, -1, 0, 10]).tolist() == [
        "D",
        "D",
        "D",
        "D",
    ]


@pytest.mark.parametrize("code", [11, 12, 99, 4.5])
def test_a_ctband_code_outside_the_dictionary_fails_naming_ctband(code):
    with pytest.raises(ValueError, match="CTBAND has codes missing .*" + f"{code:g}"):
        derive_council_tax_band_from_frs([1, code])


@given(
    st.lists(
        st.one_of(
            st.sampled_from(list(CTBAND_BANDS) + [CTBAND_NO_BAND, 0, -1]),
            st.just(np.nan),
        ),
        max_size=50,
    )
)
def test_every_ctband_value_is_a_policyengine_uk_band(codes):
    bands = derive_council_tax_band_from_frs(codes)
    # policyengine-uk raises on any string that is not a band name.
    CouncilTaxBand.encode(np.asarray(bands, dtype=str))
    assert len(bands) == len(codes)


TENURES = dict(
    zip(
        range(1, 7),
        [
            "RENT_FROM_COUNCIL",
            "RENT_FROM_HA",
            "RENT_PRIVATELY",
            "RENT_PRIVATELY",
            "OWNED_OUTRIGHT",
            "OWNED_WITH_MORTGAGE",
        ],
    )
)


def test_map_frs_codes_maps_codes_and_blanks():
    assert map_frs_codes([1, np.nan, 6], TENURES, "PTENTYP2", blank=3).tolist() == [
        "RENT_FROM_COUNCIL",
        "RENT_PRIVATELY",
        "OWNED_WITH_MORTGAGE",
    ]


@pytest.mark.parametrize("code", [0, -1, 7])
def test_map_frs_codes_names_the_variable_and_unknown_code(code):
    with pytest.raises(ValueError, match=f"FRS PTENTYP2 .*: {code:g}\\."):
        map_frs_codes([1, code], TENURES, "PTENTYP2", blank=3)
