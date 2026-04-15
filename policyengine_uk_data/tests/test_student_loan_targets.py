"""Tests for SLC student loan calibration targets."""

import json
from types import SimpleNamespace

import numpy as np


def test_slc_targets_registered():
    """SLC targets appear in the target registry."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    assert "slc/plan_2_borrowers_above_threshold" in targets
    assert "slc/plan_5_borrowers_above_threshold" in targets
    assert "slc/plan_2_borrowers_liable" in targets
    assert "slc/plan_5_borrowers_liable" in targets
    assert "slc/student_loan_repayment/england" in targets
    assert "slc/student_loan_repayment/scotland" in targets
    assert "slc/student_loan_repayment/england/plan_2" in targets
    assert "slc/maintenance_loan_recipients" in targets
    assert "slc/maintenance_loan_spend" in targets
    assert "slc/parents_learning_allowance_recipients" in targets
    assert "slc/parents_learning_allowance_spend" in targets
    assert "slc/adult_dependants_grant_recipients" in targets
    assert "slc/adult_dependants_grant_spend" in targets


def test_policyengine_uk_release_exposes_maintenance_loan_variable():
    """The lockfile should point at a release with the student-support variables."""
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    assert "maintenance_loan" in system.variables
    assert "parents_learning_allowance" in system.variables
    assert "adult_dependants_grant" in system.variables


def test_slc_snapshot_values_match_higher_education_total_rows():
    """Snapshot values should match the HE-total borrower rows."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}

    assert targets["slc/plan_2_borrowers_above_threshold"].values[2025] == 3_985_000
    assert targets["slc/plan_2_borrowers_above_threshold"].values[2030] == 5_205_000
    assert targets["slc/plan_2_borrowers_liable"].values[2025] == 8_940_000
    assert targets["slc/plan_2_borrowers_liable"].values[2030] == 10_525_000

    assert targets["slc/plan_5_borrowers_above_threshold"].values[2025] == 0
    assert targets["slc/plan_5_borrowers_above_threshold"].values[2026] == 35_000
    assert targets["slc/plan_5_borrowers_above_threshold"].values[2030] == 1_235_000
    assert targets["slc/plan_5_borrowers_liable"].values[2025] == 10_000
    assert targets["slc/plan_5_borrowers_liable"].values[2030] == 3_400_000


def test_liable_targets_exceed_above_threshold_targets():
    """Liable counts should exceed above-threshold counts in the same year."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    for year, count in targets["slc/plan_2_borrowers_above_threshold"].values.items():
        assert targets["slc/plan_2_borrowers_liable"].values[year] > count
    for year, count in targets["slc/plan_5_borrowers_above_threshold"].values.items():
        assert targets["slc/plan_5_borrowers_liable"].values[year] > count


def test_slc_repayment_targets_match_official_2025_values():
    """Repayment amount targets should match the official 2024-25 releases."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}

    assert (
        targets["slc/student_loan_repayment/england"].values[2025] == 5_018_231_834.95
    )
    assert (
        targets["slc/student_loan_repayment/england/plan_1"].values[2025]
        == 1_852_699_178.55
    )
    assert (
        targets["slc/student_loan_repayment/england/plan_2"].values[2025]
        == 2_778_253_361.64
    )
    assert (
        targets["slc/student_loan_repayment/england/postgraduate"].values[2025]
        == 346_409_713.95
    )
    assert (
        targets["slc/student_loan_repayment/england/plan_5"].values[2025]
        == 40_869_580.81
    )
    assert targets["slc/student_loan_repayment/scotland"].values[2025] == 203_300_000
    assert targets["slc/student_loan_repayment/wales"].values[2025] == 229_100_000
    assert (
        targets["slc/student_loan_repayment/northern_ireland"].values[2025]
        == 181_700_000
    )


def test_slc_england_plan_repayments_sum_to_england_total():
    """England plan-level repayment targets should reconcile to the total."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}
    england_total = targets["slc/student_loan_repayment/england"].values[2025]
    england_plans = (
        targets["slc/student_loan_repayment/england/plan_1"].values[2025]
        + targets["slc/student_loan_repayment/england/plan_2"].values[2025]
        + targets["slc/student_loan_repayment/england/postgraduate"].values[2025]
        + targets["slc/student_loan_repayment/england/plan_5"].values[2025]
    )
    assert england_plans == england_total


def test_slc_maintenance_loan_targets_match_official_2025_values():
    """Maintenance-loan targets should match Table 3A for 2024/25."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}

    assert targets["slc/maintenance_loan_recipients"].values[2025] == 1_159_761
    assert targets["slc/maintenance_loan_spend"].values[2025] == 8_591_659_718


def test_slc_targeted_support_targets_match_official_2025_values():
    """PLA and ADG targets should match Table 4C(i) for 2024/25."""
    from policyengine_uk_data.targets.registry import get_all_targets

    targets = {t.name: t for t in get_all_targets()}

    assert targets["slc/parents_learning_allowance_recipients"].values[2025] == 99_645
    assert targets["slc/parents_learning_allowance_spend"].values[2025] == 181_421_659
    assert targets["slc/adult_dependants_grant_recipients"].values[2025] == 18_611
    assert targets["slc/adult_dependants_grant_spend"].values[2025] == 55_364_917


def test_slc_maintenance_loan_snapshot_matches_known_series_points():
    """Snapshot should preserve the published maintenance-loan time series."""
    from policyengine_uk_data.targets.sources import slc

    data = slc.get_maintenance_loan_snapshot_data()

    assert data["recipients"][2014] == 972_830
    assert data["recipients"][2024] == 1_154_427
    assert data["amount_paid"][2017] == 4_870_158_274
    assert data["amount_paid"][2025] == 8_591_659_718


def test_slc_targeted_support_snapshot_matches_known_series_points():
    """Snapshot should preserve the published PLA and ADG series."""
    from policyengine_uk_data.targets.sources import slc

    data = slc.get_targeted_support_snapshot_data()

    assert data["adult_dependants_grant"]["recipients"][2014] == 13_836
    assert data["adult_dependants_grant"]["amount_paid"][2025] == 55_364_917
    assert data["parents_learning_allowance"]["recipients"][2021] == 76_740
    assert data["parents_learning_allowance"]["amount_paid"][2025] == 181_421_659


def test_slc_testing_mode_uses_snapshot_without_network(monkeypatch):
    """Dataset-build CI should not depend on a live SLC endpoint."""
    from policyengine_uk_data.targets.sources import slc

    slc._fetch_slc_data.cache_clear()
    monkeypatch.setenv("TESTING", "1")

    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be used in TESTING mode")

    monkeypatch.setattr(slc.requests, "get", fail_network)

    assert slc._fetch_slc_data() == slc.get_snapshot_data()
    slc._fetch_slc_data.cache_clear()


def test_slc_maintenance_loan_testing_mode_uses_snapshot_without_network(monkeypatch):
    """Maintenance-loan targets should also avoid network in TESTING mode."""
    from policyengine_uk_data.targets.sources import slc

    slc._fetch_maintenance_loan_data.cache_clear()
    monkeypatch.setenv("TESTING", "1")

    def fail_excel(*args, **kwargs):
        raise AssertionError("network should not be used in TESTING mode")

    monkeypatch.setattr(slc.pd, "read_excel", fail_excel)

    assert (
        slc._fetch_maintenance_loan_data() == slc.get_maintenance_loan_snapshot_data()
    )
    slc._fetch_maintenance_loan_data.cache_clear()


def test_slc_targeted_support_testing_mode_uses_snapshot_without_network(monkeypatch):
    """Targeted-support SLC data should also avoid network in TESTING mode."""
    from policyengine_uk_data.targets.sources import slc

    slc._fetch_targeted_support_data.cache_clear()
    monkeypatch.setenv("TESTING", "1")

    def fail_excel(*args, **kwargs):
        raise AssertionError("network should not be used in TESTING mode")

    monkeypatch.setattr(slc.pd, "read_excel", fail_excel)

    assert slc._fetch_targeted_support_data() == slc.get_targeted_support_snapshot_data()
    slc._fetch_targeted_support_data.cache_clear()


def test_slc_parser_uses_higher_education_total_rows(monkeypatch):
    """Parser should read HE-total rows, not the first matching above-threshold row."""
    from policyengine_uk_data.targets.sources import slc

    table_json = {
        "thead": [
            [],
            [{"text": "2024-25"}] * 6 + [{"text": "2024-25"}] * 6,
        ],
        "tbody": [
            [{"text": "Higher education full-time"}, {"text": "liable"}]
            + [{"text": "1,000"}] * 12,
            [
                {
                    "text": "Number of borrowers liable to repay and earning above repayment threshold"
                }
            ]
            + [{"text": "100"}] * 12,
            [{"text": "Higher education total"}, {"text": "liable"}]
            + [{"text": "8,940,000"}] * 6
            + [{"text": "10,000"}] * 6,
            [
                {
                    "text": "Number of borrowers liable to repay and earning above repayment threshold"
                }
            ]
            + [{"text": "3,985,000"}] * 6
            + [{"text": "35,000"}] * 6,
        ],
    }
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(
            {"props": {"pageProps": {"data": {"table": {"json": table_json}}}}}
        )
        + "</script>"
    )

    class DummyResponse:
        text = html

        @staticmethod
        def raise_for_status():
            return None

    slc._fetch_slc_data.cache_clear()
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(slc.requests, "get", lambda *args, **kwargs: DummyResponse())

    data = slc._fetch_slc_data()
    assert data["plan_2"]["liable"][2025] == 8_940_000
    assert data["plan_2"]["above_threshold"][2025] == 3_985_000
    assert data["plan_5"]["liable"][2025] == 10_000
    assert data["plan_5"]["above_threshold"][2025] == 35_000

    slc._fetch_slc_data.cache_clear()


def test_slc_parser_preserves_zero_value_years(monkeypatch):
    """A literal zero should remain a real target year, not be dropped."""
    from policyengine_uk_data.targets.sources import slc

    table_json = {
        "thead": [
            [],
            [{"text": "2024-25"}] * 6 + [{"text": "2024-25"}] * 6,
        ],
        "tbody": [
            [{"text": "Higher education total"}, {"text": "liable"}]
            + [{"text": "8,940,000"}] * 6
            + [{"text": "10,000"}] * 6,
            [
                {
                    "text": "Number of borrowers liable to repay and earning above repayment threshold"
                }
            ]
            + [{"text": "3,985,000"}] * 6
            + [{"text": "0"}] * 6,
        ],
    }
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(
            {"props": {"pageProps": {"data": {"table": {"json": table_json}}}}}
        )
        + "</script>"
    )

    class DummyResponse:
        text = html

        @staticmethod
        def raise_for_status():
            return None

    slc._fetch_slc_data.cache_clear()
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(slc.requests, "get", lambda *args, **kwargs: DummyResponse())

    data = slc._fetch_slc_data()
    assert data["plan_5"]["above_threshold"][2025] == 0

    slc._fetch_slc_data.cache_clear()


def test_slc_maintenance_loan_parser_uses_grand_total_rows(monkeypatch):
    """Maintenance-loan parser should read the grand-total rows from Table 3A."""
    from policyengine_uk_data.targets.sources import slc

    table = np.full((24, 16), np.nan, dtype=object)
    table[6, 4] = "Number of students paid (000s) [27]"
    table[7, 4] = "2013/14"
    table[7, 5] = "2024/25"
    table[12, 1] = "Grand total"
    table[12, 4] = 972.830
    table[12, 5] = 1159.761
    table[15, 4] = "Amount paid (£m)"
    table[16, 4] = "2013/14"
    table[16, 5] = "2024/25"
    table[21, 1] = "Grand total"
    table[21, 4] = 3783.626551
    table[21, 5] = 8591.659718

    df = np.array(table, dtype=object)

    slc._fetch_maintenance_loan_data.cache_clear()
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(
        slc.pd,
        "read_excel",
        lambda *args, **kwargs: __import__("pandas").DataFrame(df),
    )

    data = slc._fetch_maintenance_loan_data()
    assert data["recipients"][2014] == 972_830
    assert data["recipients"][2025] == 1_159_761
    assert data["amount_paid"][2014] == 3_783_626_551
    assert data["amount_paid"][2025] == 8_591_659_718

    slc._fetch_maintenance_loan_data.cache_clear()


def test_slc_targeted_support_parser_uses_table_4c_rows(monkeypatch):
    """Targeted-support parser should read the published Table 4C rows."""
    from policyengine_uk_data.targets.sources import slc

    table = np.full((24, 15), np.nan, dtype=object)
    table[7, 3] = "2013/14"
    table[7, 14] = "2024/25"
    table[9, 1] = "Adult Dependents Grant"
    table[9, 3] = 13.836
    table[9, 14] = 18.611
    table[10, 1] = "Parents Learning Allowance"
    table[10, 3] = 49.219
    table[10, 14] = 99.645
    table[17, 3] = "2013/14"
    table[17, 14] = "2024/25"
    table[19, 1] = "Adult Dependents Grant"
    table[19, 3] = 33.05009068
    table[19, 14] = 55.36491681
    table[20, 1] = "Parents Learning Allowance"
    table[20, 3] = 70.52488619
    table[20, 14] = 181.42165932

    slc._fetch_targeted_support_data.cache_clear()
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.setattr(
        slc.pd,
        "read_excel",
        lambda *args, **kwargs: __import__("pandas").DataFrame(table),
    )

    data = slc._fetch_targeted_support_data()
    assert data["adult_dependants_grant"]["recipients"][2014] == 13_836
    assert data["adult_dependants_grant"]["amount_paid"][2025] == 55_364_917
    assert data["parents_learning_allowance"]["recipients"][2025] == 99_645
    assert data["parents_learning_allowance"]["amount_paid"][2014] == 70_524_886

    slc._fetch_targeted_support_data.cache_clear()


def test_student_loan_target_compute_distinguishes_liable_from_repaying():
    """Above-threshold counts should require repayments, while liable counts should not."""
    from policyengine_uk_data.targets.compute.other import (
        compute_student_loan_plan,
        compute_student_loan_plan_liable,
    )

    class DummyCtx:
        country = np.array(["ENGLAND", "WALES"])

        class sim:
            @staticmethod
            def calculate(variable, map_to=None):
                if variable == "country" and map_to == "person":
                    return SimpleNamespace(
                        values=np.array(["ENGLAND", "ENGLAND", "WALES", "ENGLAND"])
                    )
                raise AssertionError(f"Unexpected calculate call: {variable}, {map_to}")

        @staticmethod
        def pe_person(variable):
            values = {
                "student_loan_plan": np.array(["PLAN_2", "PLAN_2", "PLAN_2", "PLAN_5"]),
                "student_loan_repayments": np.array([10.0, 0.0, 15.0, 0.0]),
            }
            return values[variable]

        @staticmethod
        def household_from_person(values):
            return values

    above_threshold = compute_student_loan_plan(
        SimpleNamespace(name="slc/plan_2_borrowers_above_threshold"),
        DummyCtx(),
    )
    liable = compute_student_loan_plan_liable(
        SimpleNamespace(name="slc/plan_2_borrowers_liable"),
        DummyCtx(),
    )

    assert above_threshold.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert liable.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_person_support_target_compute_handles_count_and_spend():
    """Person-level support targets should bind through the shared compute path."""
    from policyengine_uk_data.targets.build_loss_matrix import _compute_column
    from policyengine_uk_data.targets.schema import GeographicLevel, Target, Unit

    class DummyCtx:
        @staticmethod
        def pe_person(variable):
            values = {
                "maintenance_loan": np.array([0.0, 5_000.0, 7_500.0]),
                "parents_learning_allowance": np.array([0.0, 2_024.0, 2_024.0]),
            }
            return values[variable]

        @staticmethod
        def household_from_person(values):
            return values

    recipients = _compute_column(
        Target(
            name="slc/maintenance_loan_recipients",
            variable="maintenance_loan",
            source="slc",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.NATIONAL,
            values={2025: 1_159_761},
        ),
        DummyCtx(),
        2025,
    )
    spend = _compute_column(
        Target(
            name="slc/maintenance_loan_spend",
            variable="maintenance_loan",
            source="slc",
            unit=Unit.GBP,
            geographic_level=GeographicLevel.NATIONAL,
            values={2025: 8_591_659_718},
        ),
        DummyCtx(),
        2025,
    )
    pla_recipients = _compute_column(
        Target(
            name="slc/parents_learning_allowance_recipients",
            variable="parents_learning_allowance",
            source="slc",
            unit=Unit.COUNT,
            geographic_level=GeographicLevel.NATIONAL,
            values={2025: 99_645},
            is_count=True,
        ),
        DummyCtx(),
        2025,
    )
    pla_spend = _compute_column(
        Target(
            name="slc/parents_learning_allowance_spend",
            variable="parents_learning_allowance",
            source="slc",
            unit=Unit.GBP,
            geographic_level=GeographicLevel.NATIONAL,
            values={2025: 181_421_659},
        ),
        DummyCtx(),
        2025,
    )

    assert recipients.tolist() == [0.0, 1.0, 1.0]
    assert spend.tolist() == [0.0, 5_000.0, 7_500.0]
    assert pla_recipients.tolist() == [0.0, 1.0, 1.0]
    assert pla_spend.tolist() == [0.0, 2_024.0, 2_024.0]


def test_student_loan_repayment_target_compute_filters_country_and_plan():
    """Repayment amount targets should filter on modeled plan and country."""
    from policyengine_uk_data.targets.compute.other import (
        compute_student_loan_repayment,
    )

    class DummyCtx:
        class sim:
            @staticmethod
            def calculate(variable, map_to=None):
                if variable == "country" and map_to == "person":
                    return SimpleNamespace(
                        values=np.array(
                            [
                                "ENGLAND",
                                "ENGLAND",
                                "SCOTLAND",
                                "ENGLAND",
                                "WALES",
                            ]
                        )
                    )
                raise AssertionError(f"Unexpected calculate call: {variable}, {map_to}")

        @staticmethod
        def pe_person(variable):
            values = {
                "student_loan_plan": np.array(
                    ["PLAN_1", "PLAN_2", "PLAN_4", "POSTGRADUATE", "PLAN_1"]
                ),
                "student_loan_repayment": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
            }
            return values[variable]

        @staticmethod
        def household_from_person(values):
            return values

    england_total = compute_student_loan_repayment(
        SimpleNamespace(name="slc/student_loan_repayment/england"),
        DummyCtx(),
    )
    england_plan_2 = compute_student_loan_repayment(
        SimpleNamespace(name="slc/student_loan_repayment/england/plan_2"),
        DummyCtx(),
    )
    scotland_total = compute_student_loan_repayment(
        SimpleNamespace(name="slc/student_loan_repayment/scotland"),
        DummyCtx(),
    )

    assert england_total.tolist() == [100.0, 200.0, 0.0, 400.0, 0.0]
    assert england_plan_2.tolist() == [0.0, 200.0, 0.0, 0.0, 0.0]
    assert scotland_total.tolist() == [0.0, 0.0, 300.0, 0.0, 0.0]
