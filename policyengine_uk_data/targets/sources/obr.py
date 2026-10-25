"""OBR Economic and Fiscal Outlook targets.

Parses the OBR's detailed forecast tables (receipts and expenditure
workbooks) to extract tax receipt forecasts, benefit expenditure, and
benefit caseloads. The workbooks are the committed, hash-verified files the
active manifest names (``targets/obr_sources.py``); nothing is downloaded.
Each table's year columns are read from its header row, and a missing
target or year fails the build.
"""

import logging

import openpyxl

from policyengine_uk_data.targets.obr_sources import (
    ObrManifest,
    SourceError,
    active_manifest,
    check_coverage,
    load_workbook,
    read_row,
)
from policyengine_uk_data.targets.schema import GREAT_BRITAIN, Target, Unit

logger = logging.getLogger(__name__)


def _find_receipts_sheet(wb):
    """Return the current-receipts worksheet, located by title.

    EFO releases renumber their tables between vintages. Referring to a sheet by
    number means a renumbering turns every lookup on it into a silent no-op, so
    the sheet is matched on its title instead.

    Several sheets mention "receipts" (e.g. "Other receipts", "Property
    transactions taxes: Receipts by sector"), so the match is on the full
    "current receipts" phrase. If a future vintage makes that ambiguous, prefer
    the cash-basis sheet and otherwise raise rather than silently taking
    whichever happens to come first.
    """
    matches = [
        name
        for name in wb.sheetnames
        if (title := wb[name].cell(row=2, column=2).value)
        and "current receipts" in str(title).lower()
    ]
    if not matches:
        raise ValueError(
            f"OBR receipts: no worksheet titled 'Current receipts' found in {wb.sheetnames}"
        )
    if len(matches) == 1:
        return wb[matches[0]]
    cash = [
        name
        for name in matches
        if "cash" in str(wb[name].cell(row=2, column=2).value).lower()
    ]
    if len(cash) == 1:
        return wb[cash[0]]
    raise ValueError(
        "OBR receipts: could not identify a single current-receipts sheet; "
        f"candidates were {matches}. Refusing to guess."
    )


def _find_row(ws, label: str, col: str = "B", max_row: int = 80) -> int:
    """Find the row number where a cell starts with label."""
    for row in range(1, max_row + 1):
        cell_val = ws[f"{col}{row}"].value
        if cell_val and str(cell_val).strip().startswith(label):
            return row
    raise ValueError(f"Row '{label}' not found in sheet")


def corporate_sdlt_receipts(years=None) -> dict[int, float]:
    """policyengine-uk's SDLT receipts from non-natural persons, by year.

    These are the statistics `corporate_sdlt` allocates to households: each
    household's `shareholding` divides its corporate-sector wealth by the
    weighted total, so while that total is not zero the weighted sum of
    `corporate_sdlt` is this statistic whatever the weights. Read from the
    installed policyengine-uk, which projects it from
    PolicyEngine/policyengine-uk#2322 on (see `unprojected_corporate_sdlt`).
    Years are PolicyEngine years (2025 is 2025-26); the default is the active
    OBR manifest's years.
    """
    from policyengine_uk.system import system

    statistics = system.parameters.gov.hmrc.stamp_duty.statistics
    return {
        year: float(
            statistics.residential.corporate.revenue(f"{year}-01-01")
            + statistics.non_residential.corporate.revenue(f"{year}-01-01")
        )
        for year in (active_manifest().years if years is None else years)
    }


def unprojected_corporate_sdlt() -> list[str]:
    """The corporate SDLT revenue parameters policyengine-uk does not project.

    PolicyEngine/policyengine-uk#2322 (first released in 2.152.0) gives each
    one an uprating index, the OBR's SDLT receipts for its property type, so
    that load-time uprating projects it past HMRC's last outturn. Earlier
    versions have no index and carry the 2020-21 figures forward unchanged,
    £3.13bn in all, so the check is on the index, not on the values.
    """
    from policyengine_uk.system import system

    statistics = system.parameters.gov.hmrc.stamp_duty.statistics
    return [
        f"gov.hmrc.stamp_duty.statistics.{kind}.corporate.revenue"
        for kind in ("residential", "non_residential")
        if not getattr(statistics, kind).corporate.revenue.metadata.get("uprating")
    ]


def household_sdlt(obr_sdlt: dict[int, float]) -> dict[int, float]:
    """The OBR's SDLT receipts less policyengine-uk's corporate SDLT, by year.

    The OBR's SDLT line covers every buyer. policyengine-uk counts receipts
    from non-natural persons (companies, partnerships, collective investment
    schemes) separately, as `corporate_sdlt`, so targeting the household
    variable `stamp_duty_land_tax` at the whole line counted them twice (#564).
    Raises SourceError, which the receipts parser does not catch, if the
    statistic exceeds the OBR's line in any year.
    """
    unprojected = unprojected_corporate_sdlt()
    if unprojected:
        logger.warning(
            "OBR receipts: policyengine-uk does not project %s, so it predates "
            "PolicyEngine/policyengine-uk#2322 and the household SDLT target "
            "overstates households' share. Full builds refuse it.",
            " or ".join(unprojected),
        )
    corporate = corporate_sdlt_receipts(obr_sdlt)
    household = {year: total - corporate[year] for year, total in obr_sdlt.items()}
    negative = sorted(year for year, value in household.items() if value < 0)
    if negative:
        raise SourceError(
            f"policyengine-uk's corporate SDLT exceeds the OBR's SDLT in {negative}"
        )
    return household


def _parse_receipts(
    wb: openpyxl.Workbook, manifest: ObrManifest | None = None
) -> list[Target]:
    """Parse tax receipts from the OBR EFO.

    Income tax uses Table 3.4 (accrued basis) for consistency with
    the standard fiscal forecasting convention. Other receipts use the
    current-receipts table (cash basis) since they only appear there; that
    table is located by title because EFO vintages renumber the sheets.
    NICs are not read here: ``_parse_nics`` targets them by class.
    """
    manifest = manifest or active_manifest()
    vintage, ref = manifest.vintage, manifest.files["receipts"].url
    targets = []

    # Income tax from Table 3.4 (accrued basis)
    try:
        ws34 = wb["3.4"]
        row_num = _find_row(
            ws34, "Income tax (gross of tax credits)", col="B", max_row=30
        )
        values = read_row(ws34, row_num, manifest)
        if values:
            targets.append(
                Target(
                    name="obr/income_tax",
                    variable="income_tax",
                    source="obr",
                    unit=Unit.GBP,
                    values=values,
                    reference_url=ref,
                    forecast_vintage=vintage,
                )
            )
    except ValueError:
        logger.warning("OBR receipts: income tax row not found in 3.4")

    # Other receipts from the current-receipts table (cash basis). The sheet is
    # located by title rather than number: EFO vintages renumber the tables, and
    # in March 2026 current receipts moved to 3.8 while 3.9 became the APD
    # forecast, which silently yielded no targets at all.
    #
    # The "National insurance contributions" row is deliberately not targeted.
    # It is total NICs, and it used to be targeted on ni_employee, which
    # _parse_nics also targets at the Class 1 employee figure (£200bn against
    # £50bn on one variable in 2025-26). Retargeting it on
    # total_national_insurance would still conflict: the total also counts
    # statutory payment recoveries and "Other NIC" (Class 1A, 1B and 3,
    # settlements, unallocated). PE-UK has no variable for statutory payment
    # recoveries, Class 1A, Class 1B, settlements or unallocated NICs, and
    # ni_class_3 exists but no dataset fills it (#378). The class targets
    # already cover every NIC class the data populates.
    ws39 = _find_receipts_sheet(wb)
    cash_rows = {
        "vat": ("Value added tax", "vat"),
        "fuel_duties": ("Fuel duties", "fuel_duty"),
        "capital_gains_tax": ("Capital gains tax", "capital_gains_tax"),
        "sdlt": ("Stamp duty land tax", "stamp_duty_land_tax"),
    }

    recovered: list[str] = []
    for name, (label, variable) in cash_rows.items():
        try:
            row_num = _find_row(ws39, label, col="B", max_row=80)
            values = read_row(ws39, row_num, manifest)
            if values and name == "sdlt":
                values = household_sdlt(values)
            if values:
                recovered.append(label)
                targets.append(
                    Target(
                        name=f"obr/{name}",
                        variable=variable,
                        source="obr",
                        unit=Unit.GBP,
                        values=values,
                        reference_url=ref,
                        forecast_vintage=vintage,
                    )
                )
        except ValueError:
            logger.warning("OBR receipts: row '%s' not found", label)

    # Guard on targets actually produced, not on exceptions caught: a layout
    # change that shifts the value columns finds every row and still yields
    # nothing, which is the same silent failure by a different route.
    if not recovered:
        raise ValueError(
            "OBR receipts: none of the cash-basis rows "
            f"({', '.join(cash_rows[k][0] for k in cash_rows)}) produced values "
            f"on sheet '{ws39.title}'. The EFO layout has probably changed "
            "again; silently dropping every receipts target is never correct."
        )

    return targets


def _parse_council_tax(
    wb: openpyxl.Workbook, manifest: ObrManifest | None = None
) -> list[Target]:
    """Parse Table 4.1 (council tax receipts) from expenditure xlsx."""
    manifest = manifest or active_manifest()
    vintage, ref = manifest.vintage, manifest.files["expenditure"].url
    ws = wb["4.1"]

    ct_rows = {
        "council_tax": ("Total net council tax receipts", "council_tax"),
        "council_tax_england": (
            "England council tax receipts",
            "council_tax",
        ),
        "council_tax_scotland": (
            "Scotland council tax receipts",
            "council_tax",
        ),
        "council_tax_wales": ("Wales council tax receipts", "council_tax"),
        "domestic_rates": ("NI domestic rates", "domestic_rates"),
    }

    targets = []
    for name, (label, variable) in ct_rows.items():
        try:
            row_num = _find_row(ws, label, col="B", max_row=30)
            values = read_row(ws, row_num, manifest)
            if values:
                targets.append(
                    Target(
                        name=f"obr/{name}",
                        variable=variable,
                        source="obr",
                        unit=Unit.GBP,
                        values=values,
                        reference_url=ref,
                        forecast_vintage=vintage,
                    )
                )
        except ValueError:
            logger.warning("OBR council tax: row '%s' not found", label)

    return targets


def _parse_nics(
    wb: openpyxl.Workbook, manifest: ObrManifest | None = None
) -> list[Target]:
    """Parse Table 3.4 (income tax and NICs detail) for all NIC classes.

    Covers Class 1 employee and employer (PAYE) and the self-employed
    Classes 2 and 4, as one combined line or as separate rows. Class 3
    (voluntary) is left out: no dataset populates ni_class_3 (#378).

    Where a row label is not found the parser logs a warning and skips that
    variable; the manifest's required targets then fail the build unless one
    of the self-employed alternatives was found.
    """
    manifest = manifest or active_manifest()
    vintage, ref = manifest.vintage, manifest.files["receipts"].url
    ws = wb["3.4"]

    # Map of target name → (list of candidate labels to search, variable
    # name). Candidates let us accept minor wording variation in OBR EFOs
    # (e.g. "Class 2 NICs" vs "Class 2 Self-Employed NICs") without
    # failing the whole parse.
    nic_rows: dict[str, tuple[list[str], str]] = {
        "ni_employee": (
            ["Class 1 Employee NICs"],
            "ni_employee",
        ),
        "ni_employer": (
            ["Class 1 Employer NICs"],
            "ni_employer",
        ),
        # Self-employed NICs.
        #
        # Recent OBR EFOs (e.g. March 2026) publish a single combined
        # "Class 4 and Class 2 Self employed NICs" line rather than two
        # separate rows. Earlier EFOs split them. We list the combined
        # label first; if a future EFO reverts to separate rows, the
        # legacy ``ni_class_2`` / ``ni_class_4`` entries below will pick
        # them up instead.
        "ni_self_employed": (
            [
                "Class 4 and Class 2 Self employed NICs",
                "Class 2 and Class 4 Self employed NICs",
            ],
            "ni_self_employed",
        ),
        "ni_class_2": (
            [
                "Class 2 NICs",
                "Class 2 Self-Employed NICs",
                "Class 2 self-employed NICs",
            ],
            "ni_class_2",
        ),
        # Class 3 NICs are voluntary contributions to fill state-pension
        # record gaps. The PE-UK variable ni_class_3 is an input with no
        # formula and no dataset path populates it, so a calibration target
        # would fall through to a flat-zero matrix column with no signal —
        # see issue #378. Class 3 is also small (~£50m vs ~£150bn total
        # NICs, ~0.03%), so the calibrator cannot meaningfully match it
        # without a record-level imputation. Skipped here pending an
        # imputation that addresses #88; restore the row when one lands.
        "ni_class_4": (
            [
                "Class 4 NICs",
                "Class 4 Self-Employed NICs",
                "Class 4 self-employed NICs",
            ],
            "ni_class_4",
        ),
    }

    targets = []
    for name, (labels, variable) in nic_rows.items():
        row_num = None
        last_error = None
        for label in labels:
            try:
                row_num = _find_row(ws, label, col="B", max_row=40)
                break
            except ValueError as e:
                last_error = e
                continue
        if row_num is None:
            logger.warning(
                "OBR NICs: no row matched labels %s for %s: %s",
                labels,
                variable,
                last_error,
            )
            continue

        values = read_row(ws, row_num, manifest)
        if not values:
            continue

        kwargs: dict = {
            "name": f"obr/{name}",
            "variable": variable,
            "source": "obr",
            "unit": Unit.GBP,
            "values": values,
            "reference_url": ref,
            "forecast_vintage": vintage,
        }
        targets.append(Target(**kwargs))

    return targets


def _universal_credit_rows(ws) -> tuple[int, int]:
    """Table 4.9's universal credit rows inside and outside the welfare cap.

    Each section has exactly one row starting "Universal credit"; the
    outside-the-cap section starts at the row headed "Welfare spending
    outside the welfare cap".
    """
    max_row = ws.max_row
    boundary = _find_row(
        ws, "Welfare spending outside the welfare cap", max_row=max_row
    )
    rows = [
        row
        for row in range(1, max_row + 1)
        if str(ws[f"B{row}"].value or "").strip().startswith("Universal credit")
    ]
    inside = [row for row in rows if row < boundary]
    outside = [row for row in rows if row > boundary]
    if len(inside) != 1 or len(outside) != 1:
        raise ValueError(
            f"expected one universal credit row each side of row {boundary}, "
            f"found rows {rows}"
        )
    return inside[0], outside[0]


def _parse_welfare(
    wb: openpyxl.Workbook, manifest: ObrManifest | None = None
) -> list[Target]:
    """Parse Table 4.9 (welfare spending) from expenditure xlsx."""
    manifest = manifest or active_manifest()
    vintage, ref = manifest.vintage, manifest.files["expenditure"].url
    ws = wb["4.9"]

    def read_49(row_num: int) -> dict[int, float]:
        return read_row(ws, row_num, manifest)

    # "Housing benefit (not on JSA)" is not targeted: it is DWP-funded GB
    # spending only, and dwp_housing_benefit.py targets total GB Housing
    # Benefit split by age instead. "Pension credit" is the same DWP GB line
    # that dwp_pension_credit.py targets, with its caseload.
    benefit_rows = {
        "pip": (
            "Disability living allowance and personal independence p",
            "pip",
        ),
        "esa": ("Incapacity benefits", "esa_income"),
        "attendance_allowance": (
            "Attendance allowance",
            "attendance_allowance",
        ),
        "carers_allowance": ("Carer's allowance", "carers_allowance"),
        "statutory_maternity_pay": (
            "Statutory maternity pay",
            "statutory_maternity_pay",
        ),
        "winter_fuel_allowance": (
            "Winter fuel payment",
            "winter_fuel_allowance",
        ),
        "child_benefit": ("Child benefit", "child_benefit"),
        "state_pension": ("State pension", "state_pension"),
        "jobseekers_allowance": (
            "Jobseeker's allowance",
            "jsa",
        ),
    }

    targets = []
    # Welfare cap section (rows 6-36)
    for name, (label, variable) in benefit_rows.items():
        try:
            row_num = _find_row(ws, label, col="B", max_row=55)
            values = read_49(row_num)
            if values:
                targets.append(
                    Target(
                        name=f"obr/{name}",
                        variable=variable,
                        source="obr",
                        unit=Unit.GBP,
                        values=values,
                        reference_url=ref,
                        forecast_vintage=vintage,
                    )
                )
        except ValueError:
            logger.warning("OBR welfare: row '%s' not found", label)

    # Universal credit has two rows: spending inside the welfare cap (row 18
    # of the March 2026 table) and outside it (row 43). The welfare cap is the
    # Charter for Budget Responsibility's limit on welfare spending, not the
    # household benefit cap. It excludes the State Pension and the payments
    # most sensitive to the economic cycle: JSA, associated Housing Benefit
    # and their UC equivalent. DWP's benefit expenditure and caseload tables
    # (Spring Forecast 2026) give the same £12.876bn for 2025-26 and define
    # it (Notes) as "the total expenditure ... directed to those in the
    # Intensive Work Search group"; EFO March 2026 para 4.24 n.20 uses the
    # same regime. policyengine-uk has no UC conditionality regime. A proxy
    # built from the legal tests and FRS employment status went from 5% under
    # to 27% over DWP's 2025-26 figure when one employment-status code was
    # mapped correctly, too fragile to calibrate to, so the two rows are
    # targeted as one total. Both sit under "DWP social security",
    # which covers Great Britain: Northern Ireland's UC is in the "NI social
    # security" rows.
    try:
        inside_row, outside_row = _universal_credit_rows(ws)
        inside, outside = read_49(inside_row), read_49(outside_row)
        if not inside or inside.keys() != outside.keys():
            raise ValueError("the two universal credit rows cover different years")
        targets.append(
            Target(
                name="obr/universal_credit",
                variable="universal_credit",
                source="obr",
                unit=Unit.GBP,
                values={year: inside[year] + outside[year] for year in inside},
                countries=GREAT_BRITAIN,
                reference_url=ref,
                forecast_vintage=vintage,
            )
        )
    except ValueError as e:
        logger.warning("OBR welfare: universal credit not parsed: %s", e)

    return targets


def _parse_tv_licence(
    wb: openpyxl.Workbook, manifest: ObrManifest | None = None
) -> list[Target]:
    """Parse Table 4.19 (BBC) from expenditure xlsx."""
    manifest = manifest or active_manifest()
    vintage, ref = manifest.vintage, manifest.files["expenditure"].url

    try:
        ws = wb["4.19"]
        # The receipts row only: neighbouring rows such as "Standard colour
        # licence fee (£)" share the words but not the unit.
        row_num = _find_row(ws, "BBC licence fee receipts", max_row=30)
        values = read_row(ws, row_num, manifest)
        if values:
            return [
                Target(
                    name="obr/tv_licence_fee",
                    variable="tv_licence",
                    source="obr",
                    unit=Unit.GBP,
                    values=values,
                    reference_url=ref,
                    forecast_vintage=vintage,
                )
            ]
    except (KeyError, ValueError):
        logger.warning("OBR: TV licence table not found")

    return []


# ISC census: private school students (roughly constant at ~557k)
_PRIVATE_SCHOOL = {y: 557_000 for y in range(2018, 2032)}

# Salary sacrifice NICs relief comes from HMRC Table 6.2 (hmrc_salary_sacrifice.py).

# Salary sacrifice headcount: 7.7m total (3.3m above £2k, 4.3m below)
# OBR para 1.7: SS population grows 0.9% faster than employees (~2.4%/yr)
_SS_HEADCOUNT_GROWTH = 1.024
_SS_TOTAL_USERS = {
    y: 7_700_000 * _SS_HEADCOUNT_GROWTH ** max(0, y - 2024) for y in range(2024, 2032)
}
_SS_BELOW_CAP_USERS = {
    y: 4_300_000 * _SS_HEADCOUNT_GROWTH ** max(0, y - 2024) for y in range(2024, 2032)
}
_SS_ABOVE_CAP_USERS = {
    y: 3_300_000 * _SS_HEADCOUNT_GROWTH ** max(0, y - 2024) for y in range(2024, 2032)
}


def workbook_targets(manifest: ObrManifest | None = None, storage=None) -> list[Target]:
    """The targets a vintage's two verified workbooks yield.

    Fails if either workbook is not the file its manifest names, or if any
    target or year the manifest requires is missing.
    """
    manifest = manifest or active_manifest()
    receipts = load_workbook("receipts", manifest, storage)
    expenditure = load_workbook("expenditure", manifest, storage)
    targets = [
        *_parse_receipts(receipts, manifest),
        *_parse_nics(receipts, manifest),
        *_parse_council_tax(expenditure, manifest),
        *_parse_welfare(expenditure, manifest),
        *_parse_tv_licence(expenditure, manifest),
    ]
    check_coverage(targets, manifest)
    return targets


def get_targets() -> list[Target]:
    # Not caught: a calibration without the OBR targets is never correct.
    targets = workbook_targets()

    # Static targets that don't come from the xlsx
    targets.append(
        Target(
            name="obr/private_school_students",
            variable="attends_private_school",
            source="obr",
            unit=Unit.COUNT,
            values=_PRIVATE_SCHOOL,
            is_count=True,
            reference_url="https://www.isc.co.uk/research/annual-census/",
        )
    )

    # Salary sacrifice headcount targets
    _SS_REF = (
        "https://www.gov.uk/government/publications/"
        "salary-sacrifice-reform-for-pension-contributions"
        "-effective-from-6-april-2029"
    )
    targets.append(
        Target(
            name="obr/salary_sacrifice_users_total",
            variable="pension_contributions_via_salary_sacrifice",
            source="obr",
            unit=Unit.COUNT,
            values=_SS_TOTAL_USERS,
            is_count=True,
            reference_url=_SS_REF,
        )
    )
    targets.append(
        Target(
            name="obr/salary_sacrifice_users_below_cap",
            variable="pension_contributions_via_salary_sacrifice",
            source="obr",
            unit=Unit.COUNT,
            values=_SS_BELOW_CAP_USERS,
            is_count=True,
            reference_url=_SS_REF,
        )
    )
    targets.append(
        Target(
            name="obr/salary_sacrifice_users_above_cap",
            variable="pension_contributions_via_salary_sacrifice",
            source="obr",
            unit=Unit.COUNT,
            values=_SS_ABOVE_CAP_USERS,
            is_count=True,
            reference_url=_SS_REF,
        )
    )

    return targets
