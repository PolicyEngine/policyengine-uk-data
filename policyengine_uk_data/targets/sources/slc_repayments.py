"""Student Loans Company repayment amount calibration targets.

These are 2024-25 higher-education repayment amounts from the official
Student Loans Company publications, mapped to calendar year 2025 for the
2025 calibration build. We deliberately do not project these forward here;
the target resolver will carry the latest observed value forward for nearby
years when needed.
"""

from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit

_ENGLAND_TABLES_URL = (
    "https://assets.publishing.service.gov.uk/media/"
    "6943ee619273c48f554cf5c5/slcsp012025_Corrected.xlsx"
)
_SCOTLAND_URL = (
    "https://www.gov.uk/government/statistics/"
    "student-loans-in-scotland-2024-to-2025/"
    "student-loans-for-higher-education-in-scotland-financial-year-2024-25"
)
_WALES_URL = (
    "https://www.gov.uk/government/statistics/"
    "student-loans-in-wales-2024-to-2025/"
    "student-loans-for-higher-education-in-wales-financial-year-2024-25"
)
_NORTHERN_IRELAND_URL = (
    "https://www.gov.uk/government/statistics/"
    "student-loans-in-northern-ireland-2024-to-2025/"
    "student-loans-for-higher-education-in-northern-ireland-financial-year-2024-25"
)

# England values come from Table 1A of the official corrected workbook for
# financial year 2024-25, using the exact £m entries converted to pounds.
_TARGETS_2025 = {
    "slc/student_loan_repayment/england": (
        5_018_231_834.95,
        _ENGLAND_TABLES_URL,
        GeographicLevel.COUNTRY,
        "ENGLAND",
    ),
    "slc/student_loan_repayment/england/plan_1": (
        1_852_699_178.55,
        _ENGLAND_TABLES_URL,
        GeographicLevel.COUNTRY,
        "ENGLAND",
    ),
    "slc/student_loan_repayment/england/plan_2": (
        2_778_253_361.64,
        _ENGLAND_TABLES_URL,
        GeographicLevel.COUNTRY,
        "ENGLAND",
    ),
    "slc/student_loan_repayment/england/postgraduate": (
        346_409_713.95,
        _ENGLAND_TABLES_URL,
        GeographicLevel.COUNTRY,
        "ENGLAND",
    ),
    "slc/student_loan_repayment/england/plan_5": (
        40_869_580.81,
        _ENGLAND_TABLES_URL,
        GeographicLevel.COUNTRY,
        "ENGLAND",
    ),
    "slc/student_loan_repayment/scotland": (
        203_300_000,
        _SCOTLAND_URL,
        GeographicLevel.COUNTRY,
        "SCOTLAND",
    ),
    "slc/student_loan_repayment/wales": (
        229_100_000,
        _WALES_URL,
        GeographicLevel.COUNTRY,
        "WALES",
    ),
    "slc/student_loan_repayment/northern_ireland": (
        181_700_000,
        _NORTHERN_IRELAND_URL,
        GeographicLevel.COUNTRY,
        "NORTHERN_IRELAND",
    ),
}


def get_targets() -> list[Target]:
    """Return SLC repayment amount targets."""
    return [
        Target(
            name=name,
            variable="student_loan_repayment",
            source="slc",
            unit=Unit.GBP,
            geographic_level=level,
            geo_code=geo_code,
            values={2025: value},
            reference_url=reference_url,
        )
        for name, (value, reference_url, level, geo_code) in _TARGETS_2025.items()
    ]
