import numpy as np
import pandas as pd
import policyengine_uk
import policyengine_uk_data.datasets.frs as frs_module

from policyengine_uk_data.datasets.frs import (
    add_legacy_benefit_proxies,
    attach_legacy_benefit_proxies_from_frs_person,
    apply_legacy_benefit_proxies,
    create_frs,
    derive_esa_health_condition_proxy,
    derive_esa_support_group_proxy,
    derive_legacy_jobseeker_proxy,
    load_legacy_jobseeker_max_annual_hours,
)


class FakeSim:
    def __init__(self, state_pension_age):
        self._state_pension_age = np.asarray(state_pension_age)

    def calculate(self, variable, period):
        assert variable == "state_pension_age"
        return pd.Series(self._state_pension_age)


def test_legacy_jobseeker_proxy_tracks_unemployed_working_age_low_hours():
    max_annual_hours = load_legacy_jobseeker_max_annual_hours(2025)
    result = derive_legacy_jobseeker_proxy(
        age=np.array([18, 30, 30, 66, 17, 25, 25, 66, 30, 30]),
        employment_status=np.array(
            [
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "STUDENT",
                "CARER",
                "UNEMPLOYED",
                "UNEMPLOYED",
                "UNEMPLOYED",
            ]
        ),
        hours_worked=np.array([0, 12 * 52, 16 * 52, 0, 0, 0, 0, 0, 0, 0]),
        current_education=np.array(
            [
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "TERTIARY",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "UPPER_SECONDARY",
                "NOT_IN_EDUCATION",
            ]
        ),
        employment_status_reported=np.array(
            [True, True, True, True, True, True, True, True, True, False]
        ),
        state_pension_age=np.array([66, 66, 66, 66, 66, 66, 66, 67, 66, 66]),
        max_annual_hours=max_annual_hours,
    )

    assert result.tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]


def test_esa_health_condition_proxy_uses_disabled_employment_states():
    result = derive_esa_health_condition_proxy(
        age=np.array([16, 45, 45, 66, 45]),
        employment_status=np.array(
            [
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "FT_EMPLOYED",
                "LONG_TERM_DISABLED",
                "LONG_TERM_DISABLED",
            ]
        ),
        employment_status_reported=np.array([True, True, True, True, False]),
        state_pension_age=np.array([66, 66, 66, 66, 66]),
    )

    assert result.tolist() == [True, True, False, False, False]


def test_esa_support_group_proxy_is_stricter_subset_of_health_proxy():
    health_proxy = np.array([True, True, True, False, True])
    result = derive_esa_support_group_proxy(
        age=np.array([16, 45, 45, 66, 45]),
        employment_status=np.array(
            [
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "LONG_TERM_DISABLED",
                "FT_EMPLOYED",
                "LONG_TERM_DISABLED",
            ]
        ),
        hours_worked=np.array([0, 0, 12 * 52, 0, 0]),
        esa_health_condition_proxy=health_proxy,
        employment_status_reported=np.array([True, True, True, True, False]),
        state_pension_age=np.array([66, 66, 66, 66, 66]),
    )

    assert result.tolist() == [True, False, False, False, False]


def test_add_legacy_benefit_proxies_wires_all_three_columns():
    pe_person = pd.DataFrame(
        {
            "age": [18, 45, 45, 66],
            "employment_status": [
                "UNEMPLOYED",
                "LONG_TERM_DISABLED",
                "SHORT_TERM_DISABLED",
                "LONG_TERM_DISABLED",
            ],
            "hours_worked": [0, 0, 12 * 52, 0],
            "current_education": [
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
                "NOT_IN_EDUCATION",
            ],
            "is_disabled_for_benefits": [False, True, False, True],
            "is_severely_disabled_for_benefits": [False, False, True, True],
            "esa_income_reported": [0.0, 0.0, 100.0, 0.0],
            "esa_contrib_reported": [0.0, 0.0, 0.0, 0.0],
            "incapacity_benefit_reported": [0.0, 0.0, 0.0, 0.0],
            "sda_reported": [0.0, 0.0, 0.0, 0.0],
        }
    )

    result = add_legacy_benefit_proxies(
        pe_person.copy(),
        employment_status_reported=np.array([True, True, True, False]),
        state_pension_age=np.array([66, 66, 66, 66]),
        legacy_jobseeker_max_annual_hours=load_legacy_jobseeker_max_annual_hours(2025),
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False, False, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, True, True, False]
    assert result["esa_support_group_proxy"].tolist() == [False, True, False, False]


def test_legacy_jobseeker_hours_limit_matches_policyengine_uk_parameter():
    assert load_legacy_jobseeker_max_annual_hours(2025) == 16 * 52


def test_apply_legacy_benefit_proxies_uses_sim_state_pension_age():
    pe_person = pd.DataFrame(
        {
            "age": [66, 66],
            "employment_status": ["UNEMPLOYED", "UNEMPLOYED"],
            "hours_worked": [0, 0],
            "current_education": ["NOT_IN_EDUCATION", "NOT_IN_EDUCATION"],
        }
    )

    result = apply_legacy_benefit_proxies(
        pe_person.copy(),
        FakeSim([66, 67]),
        2025,
        employment_status_reported=np.array([True, True]),
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [False, True]


def test_attach_legacy_benefit_proxies_from_frs_person_uses_empstati_mask():
    pe_person = pd.DataFrame(
        {
            "age": [30, 30],
            "employment_status": ["UNEMPLOYED", "LONG_TERM_DISABLED"],
            "hours_worked": [12 * 52, 0],
            "current_education": ["NOT_IN_EDUCATION", "NOT_IN_EDUCATION"],
        }
    )
    person = pd.DataFrame({"empstati": [1, np.nan]})

    result = attach_legacy_benefit_proxies_from_frs_person(
        pe_person.copy(),
        person,
        FakeSim([66, 66]),
        2025,
    )

    assert result["legacy_jobseeker_proxy"].tolist() == [True, False]
    assert result["esa_health_condition_proxy"].tolist() == [False, False]
    assert result["esa_support_group_proxy"].tolist() == [False, False]


class FakeBenunitPopulation:
    def __init__(self, dataset):
        self.dataset = dataset

    def household(self, variable, period):
        if variable == "region":
            return np.array(["LONDON"])
        if variable == "household_id":
            return np.array([100])
        raise KeyError(variable)


class FakeMicrosimulation:
    def __init__(self, dataset):
        self.dataset = dataset
        self.populations = {"benunit": FakeBenunitPopulation(dataset)}
        self.tax_benefit_system = type(
            "FakeTaxBenefitSystem",
            (),
            {
                "parameters": lambda self, year: type(
                    "FakeParametersRoot",
                    (),
                    {
                        "gov": type(
                            "FakeGov",
                            (),
                            {
                                "dwp": type(
                                    "FakeDwp",
                                    (),
                                    {
                                        "dla": type(
                                            "FakeDla",
                                            (),
                                            {
                                                "self_care": type(
                                                    "FakeSelfCare",
                                                    (),
                                                    {"higher": 1},
                                                )()
                                            },
                                        )(),
                                        "pip": type(
                                            "FakePip",
                                            (),
                                            {
                                                "daily_living": type(
                                                    "FakeDailyLiving",
                                                    (),
                                                    {"enhanced": 1},
                                                )()
                                            },
                                        )(),
                                    },
                                )()
                            },
                        )()
                    },
                )()
            },
        )()

    def calculate(self, variable, year=None):
        if variable == "LHA_category":
            return np.array(["A"])
        if variable == "household_id":
            return np.array([100])
        if variable == "state_pension_age":
            return pd.Series([66])
        raise KeyError(variable)


def test_create_frs_smoke_includes_legacy_proxy_columns(tmp_path, monkeypatch):
    original_read_csv = frs_module.pd.read_csv

    def fake_read_csv(path, *args, **kwargs):
        if str(path).endswith("lha_list_of_rents.csv.gz"):
            return pd.DataFrame(
                {"region": ["LONDON"], "lha_category": ["A"], "brma": ["BRMA1"]}
            )
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(policyengine_uk, "Microsimulation", FakeMicrosimulation)
    monkeypatch.setattr(frs_module.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(frs_module, "load_take_up_rate", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(frs_module, "load_parameter", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(
        frs_module, "sum_to_entity", lambda values, ids, index: np.zeros(len(index))
    )
    monkeypatch.setattr(
        frs_module,
        "sum_from_positive_fields",
        lambda table, fields: np.zeros(len(table)),
    )
    monkeypatch.setattr(
        frs_module,
        "sum_positive_variables",
        lambda variables: (
            np.sum(np.vstack([np.asarray(v) for v in variables]), axis=0)
            if variables
            else 0
        ),
    )
    monkeypatch.setattr(
        frs_module,
        "fill_with_mean",
        lambda table, indicator, amount: np.zeros(len(table)),
    )

    adult = pd.DataFrame(
        [
            {
                "sernum": 100,
                "benunit": 1,
                "person": 1,
                "accssamt": 0,
                "adema": 0,
                "ademaamt": 0,
                "age": 30,
                "age80": 30,
                "cvpay": 0,
                "educft": 0,
                "educqual": 0,
                "eduma": 0,
                "edumaamt": 0,
                "empstati": 8,
                "fsbval": 0,
                "fsfvval": 0,
                "fsmval": 0,
                "fted": 0,
                "heartval": 0,
                "hrpid": 1,
                "inearns": 0,
                "marital": 0,
                "mntamt1": 0,
                "mntamt2": 0,
                "mntus1": 0,
                "mntusam1": 0,
                "redamt": 0,
                "royyr1": 0,
                "seincam2": 0,
                "sex": 1,
                "slrepamt": 0,
                "smpadj": 0,
                "sspadj": 0,
                "tothours": 0,
                "tuborr": 0,
                "typeed2": 0,
                "uperson": 1,
                "allpay2": 0,
                "royyr2": 0,
                "royyr3": 0,
                "royyr4": 0,
                "chamtern": 0,
                "chamttst": 0,
                "apamt": 0,
                "apdamt": 0,
                "pareamt": 0,
                "allpay3": 0,
                "allpay4": 0,
                "grtdir1": 0,
                "grtdir2": 0,
            }
        ]
    )
    child = pd.DataFrame(columns=adult.columns)
    benunit = pd.DataFrame([{"sernum": 100, "benunit": 1, "famtypb2": 1}])
    househol = pd.DataFrame(
        [
            {
                "sernum": 100,
                "adulth": 1,
                "bedroom6": 1,
                "csewamt": 0,
                "ctannual": 0,
                "ctband": 1,
                "ctrebamt": 0,
                "cwatamtd": 0,
                "gross4": 0,
                "gvtregno": 1,
                "hhrent": 0,
                "mortint": 0,
                "ptentyp2": 0,
                "rt2rebam": 0,
                "struins": 0,
                "subrent": 0,
                "tentyp2": 0,
                "typeacc": 0,
                "watsewrt": 0,
                "niratlia": 0,
                **{f"chrgamt{i}": 0 for i in range(1, 10)},
            }
        ]
    )
    raw_tables = {
        "adult": adult,
        "child": child,
        "benunit": benunit,
        "househol": househol,
        "pension": pd.DataFrame(
            columns=[
                "person",
                "sernum",
                "penoth",
                "penpay",
                "poamt",
                "poinc",
                "ptamt",
                "ptinc",
            ]
        ),
        "oddjob": pd.DataFrame(columns=["person", "sernum", "ojamt", "ojnow"]),
        "accounts": pd.DataFrame(
            columns=["person", "sernum", "accint", "acctax", "invtax", "account"]
        ),
        "job": pd.DataFrame(columns=["person", "sernum", "deduc1", "spnamt", "salsac"]),
        "benefits": pd.DataFrame(
            columns=["person", "sernum", "benamt", "benefit", "var2"]
        ),
        "maint": pd.DataFrame(columns=["person", "sernum", "mramt", "mruamt", "mrus"]),
        "penprov": pd.DataFrame(columns=["person", "sernum", "penamt", "stemppen"]),
        "chldcare": pd.DataFrame(
            columns=["person", "sernum", "chamt", "cost", "registrd"]
        ),
        "extchild": pd.DataFrame(columns=["sernum", "nhhamt"]),
        "mortgage": pd.DataFrame(
            columns=["sernum", "borramt", "mortend", "rmamt", "rmort"]
        ),
    }

    for name, table in raw_tables.items():
        table.to_csv(tmp_path / f"{name}.tab", sep="\t", index=False)

    dataset = create_frs(tmp_path, 2025)

    assert {
        "legacy_jobseeker_proxy",
        "esa_health_condition_proxy",
        "esa_support_group_proxy",
    }.issubset(dataset.person.columns)
