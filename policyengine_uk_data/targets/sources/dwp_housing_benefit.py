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
therefore over Pension Credit qualifying age here when one of its adults
has reached State Pension age and it gets none of those benefits. Mixed-age
couples who kept pension-age Housing Benefit after May 2019 fall in the
older group; DWP does not publish its rule for them, and this assumes they
sit in its Pension Credit and State Pension groups.

Only the older group is calibrated. policyengine-uk pays working-age
Housing Benefit only as a continuing award to families that report it and
do not claim Universal Credit: 267 of the 770 working-age records that
report it, about 17,000 weighted claims in 2025-26 against DWP's 460,000.
DWP's working-age figure also includes temporary and supported
accommodation (together £5.2bn of Housing Benefit in 2025-26, not split by
age), which the FRS barely samples. A test build on 2026-09-30 that also
targeted the younger group reached DWP's £5.8bn by loading it onto about
three effective records, and fitted the other targets no better. The
younger group's figures stay here for diagnostics and tests.

Source: https://www.gov.uk/government/publications/benefit-expenditure-and-caseload-tables-2026
"""

import numpy as np

from policyengine_uk_data.targets.schema import GREAT_BRITAIN, Target, Unit

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


def _over_pension_credit_age(ctx) -> np.ndarray:
    """Benefit units assessed under the pension-age Housing Benefit rules."""
    adult = np.asarray(ctx.pe_person("is_adult"), dtype=bool)
    over = adult & np.asarray(ctx.pe_person("is_SP_age"), dtype=bool)
    any_over = (
        np.asarray(ctx.sim.map_result(over.astype(float), "person", "benunit")) > 0
    )
    on_working_age_benefit = np.zeros_like(any_over)
    for benefit in _WORKING_AGE_BENEFITS:
        on_working_age_benefit |= np.asarray(ctx.sim.calculate(benefit).values) > 0
    return any_over & ~on_working_age_benefit


# Age groups the calibration targets; see the module docstring.
_CALIBRATED_AGE_GROUPS = ("over",)


def _make_compute(age_group: str, count: bool):
    def compute(ctx, target: Target, year: int) -> np.ndarray:
        housing_benefit = np.asarray(
            ctx.sim.calculate("housing_benefit").values, dtype=float
        )
        in_group = _over_pension_credit_age(ctx)
        if age_group == "under":
            in_group = ~in_group
        value = (housing_benefit > 0) if count else housing_benefit
        return np.asarray(ctx.household_from_family(value * in_group), dtype=float)

    return compute


def get_targets() -> list[Target]:
    return build_targets(_CALIBRATED_AGE_GROUPS)


def build_targets(age_groups=("over", "under")) -> list[Target]:
    """Housing Benefit spending and claims targets for these age groups."""
    targets = []
    for age_group in age_groups:
        name = f"dwp/housing_benefit/{age_group}_pension_credit_age"
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
                custom_compute=_make_compute(age_group, count=True),
            )
        )
    return targets
