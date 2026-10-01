"""DWP Pension Credit targets: spending and claims in Great Britain.

From DWP's benefit expenditure and caseload tables for the Spring Forecast
2026 (Pension Credit sheet: nominal £ million and thousands of claims,
2022-23 to 2024-25 outturn, forecast after). Financial year 2025-26 is
stored as 2025.

DWP's figures cover Great Britain; Pension Credit in Northern Ireland is
paid by the Department for Communities, so the model columns count GB
households only. These targets replace OBR EFO table 4.9 "Pension credit",
the same GB spending series (2025-26: £6,146.4m against DWP's £6,144.5m),
which the model compared with UK-wide Pension Credit and which carries no
caseload.

Source: https://www.gov.uk/government/publications/benefit-expenditure-and-caseload-tables-2026
"""

import numpy as np

from policyengine_uk_data.targets.schema import GREAT_BRITAIN, Target, Unit

_REFERENCE_URL = (
    "https://www.gov.uk/government/publications/"
    "benefit-expenditure-and-caseload-tables-2026"
)
_VINTAGE = "spring_2026"

# "Total Pension Credit", £ million nominal.
_SPENDING_GBP_M = {
    2022: 4_935.31,
    2023: 5_467.14,
    2024: 6_007.70,
    2025: 6_144.47,
    2026: 6_054.90,
    2027: 5_911.31,
    2028: 5_783.32,
    2029: 5_832.22,
    2030: 5_821.77,
}
# "Pension Credit" caseload, thousands (annual average, rounded by DWP).
_CASELOAD_THOUSANDS = {
    2022: 1_374,
    2023: 1_370,
    2024: 1_376,
    2025: 1_382,
    2026: 1_312,
    2027: 1_253,
    2028: 1_188,
    2029: 1_142,
    2030: 1_133,
}


def _make_compute(count: bool):
    def compute(ctx, target: Target, year: int) -> np.ndarray:
        pension_credit = np.asarray(
            ctx.sim.calculate("pension_credit").values, dtype=float
        )
        value = (pension_credit > 0) if count else pension_credit
        return np.asarray(ctx.household_from_family(value), dtype=float)

    return compute


def get_targets() -> list[Target]:
    return [
        Target(
            name="dwp/pension_credit",
            variable="pension_credit",
            source="dwp",
            unit=Unit.GBP,
            values={year: v * 1e6 for year, v in _SPENDING_GBP_M.items()},
            reference_url=_REFERENCE_URL,
            forecast_vintage=_VINTAGE,
            countries=GREAT_BRITAIN,
            custom_compute=_make_compute(count=False),
        ),
        Target(
            name="dwp/pension_credit_claims",
            variable="pension_credit",
            source="dwp",
            unit=Unit.COUNT,
            values={year: v * 1e3 for year, v in _CASELOAD_THOUSANDS.items()},
            is_count=True,
            reference_url=_REFERENCE_URL,
            forecast_vintage=_VINTAGE,
            countries=GREAT_BRITAIN,
            custom_compute=_make_compute(count=True),
        ),
    ]
