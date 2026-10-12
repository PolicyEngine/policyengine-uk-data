"""Modelled CTR on synthetic households with no reported SPI benefits.

The build stores the imputed claim decision on the household reference
person's benefit unit. These small situations use that same route without
creating a dataset or running a production microsimulation.
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk import Simulation

YEAR = 2025
SCHEMES = [
    ("LONDON", "ENGLAND", "MERTON"),
    ("SCOTLAND", "SCOTLAND", "CITY_OF_EDINBURGH"),
    ("WALES", "WALES", "CARDIFF"),
]


def _situation(households, would_claim=True):
    people, benunits, homes = {}, {}, {}
    for index, household in enumerate(households):
        region, country, local_authority = household["scheme"]
        head = f"head_{index}"
        members = [head]
        people[head] = {
            "age": {YEAR: household["age"]},
            "employment_income": {YEAR: household["earnings"]},
            "is_household_head": {YEAR: True},
            "council_tax_benefit_reported": {YEAR: 0},
        }
        head_members = [head]
        if household.get("partner_age") is not None:
            partner = f"partner_{index}"
            people[partner] = {
                "age": {YEAR: household["partner_age"]},
                "employment_income": {YEAR: household["partner_earnings"]},
                "is_household_head": {YEAR: False},
                "council_tax_benefit_reported": {YEAR: 0},
            }
            members.append(partner)
            head_members.append(partner)
        benunits[f"head_family_{index}"] = {
            "members": head_members,
            "would_claim_council_tax_reduction": {YEAR: would_claim},
            "claims_all_entitled_benefits": {YEAR: False},
            "would_claim_uc": {YEAR: False},
        }
        if household.get("non_dependant_age") is not None:
            non_dependant = f"non_dependant_{index}"
            people[non_dependant] = {
                "age": {YEAR: household["non_dependant_age"]},
                "employment_income": {YEAR: household["non_dependant_earnings"]},
                "is_household_head": {YEAR: False},
                "council_tax_benefit_reported": {YEAR: 0},
            }
            members.append(non_dependant)
            benunits[f"non_dependant_family_{index}"] = {
                "members": [non_dependant],
                "would_claim_council_tax_reduction": {YEAR: False},
                "claims_all_entitled_benefits": {YEAR: False},
                "would_claim_uc": {YEAR: False},
            }
        homes[f"home_{index}"] = {
            "members": members,
            "region": {YEAR: region},
            "country": {YEAR: country},
            "local_authority": {YEAR: local_authority},
            "council_tax": {YEAR: household["bill"]},
            "savings": {YEAR: household["savings"]},
            "tenure_type": {YEAR: "OWNED_OUTRIGHT"},
        }
    return {"people": people, "benunits": benunits, "households": homes}


def _calculate(simulation, variable):
    return np.asarray(simulation.calculate(variable, YEAR))


def test_zero_reported_ctr_can_receive_imputed_household_claim():
    """An eligible English pensioner receives CTR through the explicit flag."""
    household = {
        "scheme": ("SOUTH_EAST", "ENGLAND", "MAIDSTONE"),
        "age": 70,
        "earnings": 0,
        "bill": 2_000,
        "savings": 0,
    }
    claimant = Simulation(situation=_situation([household]))
    non_claimant = Simulation(situation=_situation([household], would_claim=False))

    assert (_calculate(claimant, "council_tax_benefit_reported") == 0).all()
    assert _calculate(claimant, "would_claim_council_tax_reduction").tolist() == [True]
    assert _calculate(claimant, "council_tax_reduction")[0] == pytest.approx(
        household["bill"], abs=0.01
    )
    assert _calculate(non_claimant, "council_tax_reduction")[0] == 0


_household = st.fixed_dictionaries(
    {
        "scheme": st.sampled_from(SCHEMES),
        "age": st.integers(25, 90),
        "earnings": st.integers(0, 50_000),
        "bill": st.integers(1, 5_000),
        "savings": st.integers(0, 20_000),
        "partner_age": st.one_of(st.none(), st.integers(25, 90)),
        "partner_earnings": st.integers(0, 50_000),
        "non_dependant_age": st.one_of(st.none(), st.integers(18, 90)),
        "non_dependant_earnings": st.integers(0, 50_000),
    }
)


@settings(max_examples=15, deadline=None, derandomize=True)
@given(households=st.lists(_household, min_size=1, max_size=3))
def test_gross_council_tax_covers_ctr_received_by_zero_report_spi_households(
    households,
):
    simulation = Simulation(situation=_situation(households))
    gross = _calculate(simulation, "council_tax")
    received = _calculate(simulation, "council_tax_reduction")

    assert (_calculate(simulation, "council_tax_benefit_reported") == 0).all()
    np.testing.assert_allclose(gross, [household["bill"] for household in households])
    assert np.isfinite(received).all()
    assert (received >= 0).all()
    assert (gross >= received - 0.01).all()
