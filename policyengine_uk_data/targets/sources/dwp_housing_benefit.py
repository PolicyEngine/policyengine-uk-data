"""DWP Housing Benefit targets by age group.

Housing Benefit spending and caseload over and under Pension Credit
qualifying age, from DWP's benefit expenditure and caseload tables for the
Spring Forecast 2026 (Housing benefits sheet, nominal £ million and
thousands of claims, 2022-23 to 2024-25 outturn, forecast after). Financial
year 2025-26 is stored as 2025.

Coverage is Great Britain: Housing Benefit for Northern Ireland residents is
paid under Northern Ireland legislation and sits outside DWP's figures, so
the model columns count GB households only.

These targets replace OBR EFO table 4.9 "Housing benefit (not on JSA)".
That line is DWP-funded spending only, while DWP's tables count all
Housing Benefit paid, including the part local authorities fund (£0.79bn
in 2025-26). The age split sums to that full amount, so targeting both
would ask for two different GB totals.

DWP splits claims by benefit rules rather than by age alone (Notes, note
5): from 2024-25 the two lines equal its Pension Credit plus State Pension
benefit groups and its ESA plus other working-age groups. Under regulation 5 of
both Housing Benefit Regulations 2006 (SI 2006/213 and 2006/214), the
pension-age rules apply when the claimant or partner has reached the
qualifying age for Pension Credit, unless either is on Universal Credit,
Income Support, income-based JSA or income-related ESA. A benefit unit is
therefore over Pension Credit qualifying age here when its claimant or
partner has reached State Pension age and it gets none of those benefits. Mixed-age
couples who kept pension-age Housing Benefit after May 2019 fall in the
older group; DWP does not publish its rule for them, and this assumes they
sit in its Pension Credit and State Pension groups.

Working-age Housing Benefit is calibrated net of supported and temporary
accommodation. policyengine-uk has no rules or input for specified
(supported) or temporary accommodation (``housing_benefit_eligible``: such
claims "are not modelled"; policyengine-uk#1911). It pays working-age
Housing Benefit only as a continuing award to a family that reports it and
does not claim Universal Credit, computed under the general-needs rules. The
FRS does not identify those accommodation types, so a respondent in them who
reports Housing Benefit is modelled the same way; treating the model's
working-age Housing Benefit as general needs is an approximation. DWP's
Spring 2026 forecast tables split accommodation type only across all ages
(Housing Benefit by Accomodation Type). The calibrated working-age figure is
therefore DWP's working-age line less all of its supported and temporary
accommodation: £584.8m and 107k claims in 2025-26, against the full line's
£5,778.1m and 460k. Part of that accommodation is pension-age, so the figure
is a lower bound on working-age general-needs Housing Benefit. It exists
only for 2024-25 and 2025-26, and the targets do not carry it forward
(``carry_forward=False``): from 2026-27, DWP's all-age supported and
temporary accommodation exceeds its whole working-age line, so these lines
give no positive lower bound. That leaves DWP's supported and temporary
accommodation Housing Benefit, £5.2bn across all ages in 2025-26, unmodelled
as such: a known limitation.

Seeded test calibrations on 2026-10-04 tried the full working-age line too.
With only the pension-age figures calibrated, the model pays about 17,000
working-age claims in 2025-26. Calibrated to the full line, it came about
12% below DWP's spending by loading it onto about three effective records.
With household weights capped at 20-40 times their prior, it reached only
54-60% of that spending, and income-related ESA claimants rose to 1.6-2.0
times DWP's count. Against the net figure, no other national target crosses
the 10% line compared with calibrating the pension-age figures only. The
full working-age lines stay here (group "under") for tests and diagnostics.

Source: https://www.gov.uk/government/publications/benefit-expenditure-and-caseload-tables-2026
"""

import numpy as np

from policyengine_uk_data.targets.schema import GREAT_BRITAIN, Target, Unit
from policyengine_uk_data.utils.benefit_units import claimant_or_partner_variable

_REFERENCE_URL = (
    "https://www.gov.uk/government/publications/"
    "benefit-expenditure-and-caseload-tables-2026"
)
_VINTAGE = "spring_2026"

# Benefits that keep a claim under the working-age Housing Benefit rules
# when the claimant or partner has reached Pension Credit qualifying age.
_WORKING_AGE_BENEFITS = (
    "universal_credit",
    "income_support",
    "jsa_income",
    "esa_income",
)

# Housing benefits sheet rows "Housing Benefit over/under Pension Credit
# qualifying age": expenditure in £ million (nominal) and caseload in
# thousands (annual average, rounded to the nearest thousand by DWP).
_SPENDING_GBP_M = {
    "over": {
        2022: 5_890.1,
        2023: 6_248.1,
        2024: 6_851.0,
        2025: 7_114.7,
        2026: 7_268.9,
        2027: 7_340.4,
        2028: 7_464.4,
        2029: 7_747.9,
        2030: 7_941.3,
    },
    "under": {
        2022: 9_689.1,
        2023: 9_524.5,
        2024: 8_603.5,
        2025: 5_778.1,
        2026: 5_205.3,
        2027: 5_493.8,
        2028: 5_788.5,
        2029: 6_151.5,
        2030: 6_405.4,
    },
}
_CASELOAD_THOUSANDS = {
    "over": {
        2022: 1_121,
        2023: 1_108,
        2024: 1_104,
        2025: 1_109,
        2026: 1_082,
        2027: 1_057,
        2028: 1_040,
        2029: 1_036,
        2030: 1_042,
    },
    "under": {
        2022: 1_388,
        2023: 1_243,
        2024: 968,
        2025: 460,
        2026: 328,
        2027: 339,
        2028: 349,
        2029: 361,
        2030: 372,
    },
}


# Housing benefits sheet, "Housing Benefit by Accomodation Type", rows "of
# which Supported Accomodation" and "of which Temporary Accomodation": all
# ages, same units, first published for 2024-25.
_SUPPORTED_GBP_M = {
    2024: 3_262.8,
    2025: 3_631.4,
    2026: 3_936.0,
    2027: 4_164.1,
    2028: 4_399.2,
    2029: 4_687.1,
    2030: 4_892.2,
}
_TEMPORARY_GBP_M = {
    2024: 1_431.1,
    2025: 1_561.9,
    2026: 1_643.7,
    2027: 1_732.1,
    2028: 1_819.1,
    2029: 1_929.8,
    2030: 2_006.6,
}
_SUPPORTED_THOUSANDS = {
    2024: 235,
    2025: 242,
    2026: 246,
    2027: 251,
    2028: 256,
    2029: 262,
    2030: 267,
}
_TEMPORARY_THOUSANDS = {
    2024: 104,
    2025: 111,
    2026: 117,
    2027: 123,
    2028: 129,
    2029: 136,
    2030: 143,
}


def net_of_supported_and_temporary(
    working_age: dict, supported: dict, temporary: dict
) -> dict:
    """Working-age figures less all-age supported and temporary
    accommodation, for the years with all three where the result is
    positive."""
    net = {}
    for year in sorted(working_age.keys() & supported.keys() & temporary.keys()):
        value = round(working_age[year] - supported[year] - temporary[year], 1)
        if value > 0:
            net[year] = value
    return net


_SPENDING_GBP_M["under_general_needs"] = net_of_supported_and_temporary(
    _SPENDING_GBP_M["under"], _SUPPORTED_GBP_M, _TEMPORARY_GBP_M
)
_CASELOAD_THOUSANDS["under_general_needs"] = net_of_supported_and_temporary(
    _CASELOAD_THOUSANDS["under"], _SUPPORTED_THOUSANDS, _TEMPORARY_THOUSANDS
)

_NAMES = {
    "over": "dwp/housing_benefit/over_pension_credit_age",
    "under": "dwp/housing_benefit/under_pension_credit_age",
    "under_general_needs": "dwp/housing_benefit/under_pension_credit_age_general_needs",
}


def _over_pension_credit_age(ctx) -> np.ndarray:
    """Benefit units assessed under the pension-age Housing Benefit rules."""
    claimant = np.asarray(
        ctx.pe_person(
            claimant_or_partner_variable(ctx.sim.tax_benefit_system.variables)
        ),
        dtype=bool,
    )
    over = claimant & np.asarray(ctx.pe_person("is_SP_age"), dtype=bool)
    any_over = (
        np.asarray(ctx.sim.map_result(over.astype(float), "person", "benunit")) > 0
    )
    on_working_age_benefit = np.zeros_like(any_over)
    for benefit in _WORKING_AGE_BENEFITS:
        on_working_age_benefit |= np.asarray(ctx.sim.calculate(benefit).values) > 0
    return any_over & ~on_working_age_benefit


# Age groups the calibration targets; see the module docstring.
_CALIBRATED_AGE_GROUPS = ("over", "under_general_needs")


def _make_compute(age_group: str, count: bool):
    def compute(ctx, target: Target, year: int) -> np.ndarray:
        housing_benefit = np.asarray(
            ctx.sim.calculate("housing_benefit").values, dtype=float
        )
        in_group = _over_pension_credit_age(ctx)
        if age_group != "over":
            # The model has no supported or temporary accommodation rules,
            # so both working-age groups share one column (see the module
            # docstring).
            in_group = ~in_group
        value = (housing_benefit > 0) if count else housing_benefit
        return np.asarray(ctx.household_from_family(value * in_group), dtype=float)

    return compute


def get_targets() -> list[Target]:
    return build_targets(_CALIBRATED_AGE_GROUPS)


def build_targets(
    age_groups=("over", "under", "under_general_needs"),
) -> list[Target]:
    """Housing Benefit spending and claims targets for these age groups."""
    targets = []
    for age_group in age_groups:
        name = _NAMES[age_group]
        targets.append(
            Target(
                name=name,
                variable="housing_benefit",
                source="dwp",
                unit=Unit.GBP,
                values={
                    year: value * 1e6
                    for year, value in _SPENDING_GBP_M[age_group].items()
                },
                reference_url=_REFERENCE_URL,
                forecast_vintage=_VINTAGE,
                countries=GREAT_BRITAIN,
                carry_forward=age_group != "under_general_needs",
                custom_compute=_make_compute(age_group, count=False),
            )
        )
        targets.append(
            Target(
                name=f"{name}_claims",
                variable="housing_benefit",
                source="dwp",
                unit=Unit.COUNT,
                values={
                    year: value * 1e3
                    for year, value in _CASELOAD_THOUSANDS[age_group].items()
                },
                is_count=True,
                reference_url=_REFERENCE_URL,
                forecast_vintage=_VINTAGE,
                countries=GREAT_BRITAIN,
                carry_forward=age_group != "under_general_needs",
                custom_compute=_make_compute(age_group, count=True),
            )
        )
    return targets
