"""DWP benefit targets.

PIP daily living standard/enhanced claimant counts, benefit cap,
UC payment distribution, UC claimant counts by children/family type,
two-child limit breakdowns, and Scotland UC households with child under 1.

Sources:
- DWP Stat-Xplore: https://stat-xplore.dwp.gov.uk
- DWP benefit cap: https://www.gov.uk/government/statistics/benefit-cap-number-of-households-capped-to-february-2025
- DWP two-child limit: https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024
"""

from pathlib import Path

from policyengine_uk_data.targets.schema import Target, Unit

_STORAGE = Path(__file__).parents[2] / "storage"


def get_targets() -> list[Target]:
    targets = []

    # PIP daily living standard and enhanced claimant counts
    # From Disability Rights UK analysis of DWP data
    targets.append(
        Target(
            name="dwp/pip_dl_standard_claimants",
            variable="pip_dl_category",
            source="dwp",
            unit=Unit.COUNT,
            values={2025: 1_283_000},
            is_count=True,
            reference_url="https://www.disabilityrightsuk.org/news/90-pip-standard-daily-living-component-recipients-would-fail-new-green-paper-test",
        )
    )
    targets.append(
        Target(
            name="dwp/pip_dl_enhanced_claimants",
            variable="pip_dl_category",
            source="dwp",
            unit=Unit.COUNT,
            values={2025: 1_608_000},
            is_count=True,
            reference_url="https://www.disabilityrightsuk.org/news/90-pip-standard-daily-living-component-recipients-would-fail-new-green-paper-test",
        )
    )

    # Benefit cap
    targets.append(
        Target(
            name="dwp/benefit_capped_households",
            variable="benefit_cap_reduction",
            source="dwp",
            unit=Unit.COUNT,
            values={2025: 115_000},
            is_count=True,
            reference_url="https://www.gov.uk/government/statistics/benefit-cap-number-of-households-capped-to-february-2025/benefit-cap-number-of-households-capped-to-february-2025",
        )
    )
    targets.append(
        Target(
            name="dwp/benefit_cap_total_reduction",
            variable="benefit_cap_reduction",
            source="dwp",
            unit=Unit.GBP,
            values={2025: 60 * 52 * 115_000},
            reference_url="https://www.gov.uk/government/statistics/benefit-cap-number-of-households-capped-to-february-2025/benefit-cap-number-of-households-capped-to-february-2025",
        )
    )

    # Scotland UC households with child under 1
    targets.append(
        Target(
            name="dwp/scotland_uc_households_child_under_1",
            variable="universal_credit",
            source="dwp",
            unit=Unit.COUNT,
            values={2025: 14_000},
            is_count=True,
            reference_url="https://stat-xplore.dwp.gov.uk/",
        )
    )

    # UC claimant counts by number of children
    _UC_BY_CHILDREN = {
        "1": 1_222_944,
        "2": 1_058_967,
        "3": 473_500,
        "4": 166_790,
        "5+": 74_050 + 1_860,
    }
    for num_children, count in _UC_BY_CHILDREN.items():
        targets.append(
            Target(
                name=f"dwp/uc/claimants_with_{num_children}_children",
                variable="universal_credit",
                source="dwp",
                unit=Unit.COUNT,
                values={2025: count},
                is_count=True,
                reference_url="https://stat-xplore.dwp.gov.uk/",
            )
        )

    # UC claimant counts by family type
    _UC_BY_FAMILY_TYPE = {
        "single_no_children": 2868.011,
        "single_with_children": 2156.879,
        "couple_no_children": 231.368,
        "couple_with_children": 839.379,
    }
    undercount_relative = 1.27921 / sum(_UC_BY_FAMILY_TYPE.values())
    for family_type, count_k in _UC_BY_FAMILY_TYPE.items():
        targets.append(
            Target(
                name=f"dwp/uc/claimants_{family_type}",
                variable="universal_credit",
                source="dwp",
                unit=Unit.COUNT,
                values={2025: count_k * (1 + undercount_relative) * 1e3},
                is_count=True,
                reference_url="https://stat-xplore.dwp.gov.uk/",
            )
        )

    # Two-child limit statistics (2026 data)
    targets.append(
        Target(
            name="dwp/uc/two_child_limit/households_affected",
            variable="uc_is_child_limit_affected",
            source="dwp",
            unit=Unit.COUNT,
            values={2026: 453_600},
            is_count=True,
            reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
        )
    )
    targets.append(
        Target(
            name="dwp/uc/two_child_limit/children_in_affected_households",
            variable="is_child",
            source="dwp",
            unit=Unit.COUNT,
            values={2026: 1_613_980},
            is_count=True,
            reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
        )
    )
    targets.append(
        Target(
            name="dwp/uc/two_child_limit/children_affected",
            variable="uc_is_child_limit_affected",
            source="dwp",
            unit=Unit.COUNT,
            values={2026: 580_400},
            is_count=True,
            reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
        )
    )

    # Two-child limit by number of children
    _TCL_BY_CHILDREN = [
        (3, 283_290, 849_860),
        (4, 115_630, 462_520),
        (5, 36_590, 182_940),
        (6, 18_090, 118_670),
    ]
    for num_children, households, children in _TCL_BY_CHILDREN:
        targets.append(
            Target(
                name=f"dwp/uc/two_child_limit/{num_children}_children_households",
                variable="uc_is_child_limit_affected",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: households},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            )
        )
        targets.append(
            Target(
                name=f"dwp/uc/two_child_limit/{num_children}_children_households_total_children",
                variable="is_child",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: children},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            )
        )

    # Two-child limit by disability
    targets.extend(
        [
            Target(
                name="dwp/uc/two_child_limit/adult_pip_households",
                variable="pip",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: 62_260},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            ),
            Target(
                name="dwp/uc/two_child_limit/adult_pip_children",
                variable="is_child",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: 225_320},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            ),
            Target(
                name="dwp/uc/two_child_limit/disabled_child_element_households",
                variable="uc_individual_disabled_child_element",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: 124_560},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            ),
            Target(
                name="dwp/uc/two_child_limit/disabled_child_element_children",
                variable="is_child",
                source="dwp",
                unit=Unit.COUNT,
                values={2026: 462_660},
                is_count=True,
                reference_url="https://www.gov.uk/government/statistics/universal-credit-and-child-tax-credit-claimants-statistics-related-to-the-policy-to-provide-support-for-a-maximum-of-2-children-april-2024",
            ),
        ]
    )

    # UC national payment distribution from xlsx
    targets.extend(_uc_payment_distribution_targets())

    return targets


def _uc_payment_distribution_targets() -> list[Target]:
    """Parse UC payment distribution from xlsx into Target objects."""
    from policyengine_uk_data.utils.uc_data import uc_national_payment_dist

    targets = []
    for _, row in uc_national_payment_dist.iterrows():
        lower = row.uc_annual_payment_min
        upper = row.uc_annual_payment_max
        family_type = row.family_type
        name = f"dwp/uc_payment_dist/{family_type}_annual_payment_{lower:_.0f}_to_{upper:_.0f}"
        targets.append(
            Target(
                name=name,
                variable="universal_credit",
                source="dwp",
                unit=Unit.COUNT,
                values={2025: float(row.household_count)},
                is_count=True,
                breakdown_variable="universal_credit",
                lower_bound=float(lower),
                upper_bound=float(upper),
                reference_url="https://stat-xplore.dwp.gov.uk/",
            )
        )
    return targets
