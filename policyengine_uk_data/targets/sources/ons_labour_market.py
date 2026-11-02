"""ONS Labour Force Survey employment levels: employees and the self-employed.

The FRS records each adult's ILO employment status for their main job
(EMPSTATI, `employment_status`), the same concept the LFS uses. Without these
targets nothing in the calibration ties the number of employees or
self-employed people to an official count. The HMRC counts of income-tax
payers with employment income (SPI tables 3.6 and 3.15) are annual: they
include people with pay for part of the year whose status at interview is
out of work. So calibration can meet them by moving weight from people out of
work to employees.

The FRS child table (dependent children, including 16-19-year-olds in
non-advanced education) carries no employment status, so their jobs are not
counted here, while the LFS counts them. On the FRS 2024-25 grossing weights,
the FRS has 28.1m employees against the LFS's 29.1m for 2024.

Source: ONS Labour market overview, series MGRN (LFS: Employees: UK: All,
aged 16 and over, seasonally adjusted) and MGRQ (LFS: Self-employed: UK:
All), annual four-quarter averages, release of 15 September 2026. The loss
matrix holds the latest year's value for up to three later years and drops the
targets after that (``_resolve_value``), so add each new annual average.
"""

import numpy as np

from policyengine_uk_data.targets.schema import (
    GeographicLevel,
    Target,
    Unit,
)
from policyengine_uk_data.utils.employment_status import (
    EMPLOYEE_STATUSES,
    SELF_EMPLOYED_STATUSES,
)

_REF = (
    "https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/"
    "employmentandemployeetypes/timeseries/{series}/lms"
)

# Annual four-quarter averages, people.
EMPLOYEES = {
    2022: 28_564_000.0,
    2023: 28_821_000.0,
    2024: 29_126_000.0,
    2025: 29_590_000.0,
}
SELF_EMPLOYED = {
    2022: 4_244_000.0,
    2023: 4_380_000.0,
    2024: 4_340_000.0,
    2025: 4_395_000.0,
}


def _status_count(statuses: tuple[str, ...]):
    def compute(ctx, target, year) -> np.ndarray:
        status = np.asarray(ctx.pe_person("employment_status")).astype(str)
        return ctx.household_from_person(np.isin(status, statuses).astype(float))

    return compute


def get_targets() -> list[Target]:
    return [
        Target(
            name=f"ons/lfs_{name}",
            variable="employment_status",
            source="ons",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.NATIONAL,
            geo_code="K02000001",
            geo_name="United Kingdom",
            values=dict(values),
            is_count=True,
            reference_url=_REF.format(series=series),
            custom_compute=_status_count(statuses),
        )
        for name, values, series, statuses in (
            ("employees", EMPLOYEES, "mgrn", EMPLOYEE_STATUSES),
            ("self_employed", SELF_EMPLOYED, "mgrq", SELF_EMPLOYED_STATUSES),
        )
    ]
