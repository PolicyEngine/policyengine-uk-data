import numpy as np
import pandas as pd


class _FakeDataset:
    time_period = 2025


class _FakeSim:
    def __init__(self, *args, **kwargs):
        self.default_calculation_period = 2025

    def calculate(self, variable, *args, **kwargs):
        values = {
            "employment_income": np.array([10_000.0, 30_000.0]),
            "income_tax": np.array([1.0, 1.0]),
            "age": np.array([40, 70]),
            "universal_credit": np.array([0.0, 1.0]),
            "equiv_hbai_household_net_income": np.array([20_000.0, 25_000.0]),
            "equiv_hbai_household_net_income_ahc": np.array([18_000.0, 22_000.0]),
            "tenure_type": np.array(["RENT_PRIVATELY", "OWNED_OUTRIGHT"]),
            "benunit_rent": np.array([12_000.0, 0.0]),
            "country": np.array(["ENGLAND", "SCOTLAND"]),
        }
        return type("Result", (), {"values": values[variable]})()

    def map_result(self, values, source_entity, target_entity):
        return np.asarray(values)


def _fake_la_codes():
    return pd.DataFrame(
        {
            "code": ["E06000001", "W06000001", "S12000001", "N09000001"],
        }
    )


def _patch_common_la_inputs(monkeypatch, tmp_path):
    from policyengine_uk_data.datasets.local_areas.local_authorities import loss

    (_storage := tmp_path / "storage").mkdir()
    _fake_la_codes().to_csv(_storage / "local_authorities_2021.csv", index=False)

    monkeypatch.setattr(loss, "STORAGE_FOLDER", _storage)
    monkeypatch.setattr(loss, "Microsimulation", _FakeSim)
    monkeypatch.setattr(loss, "INCOME_VARIABLES", ["employment_income"])
    monkeypatch.setattr(
        loss,
        "get_la_income_targets",
        lambda: pd.DataFrame(
            {
                "employment_income_amount": [1.0, 1.0, 1.0, 1.0],
                "employment_income_count": [1.0, 1.0, 1.0, 1.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "get_national_income_projections",
        lambda year: pd.DataFrame(
            {
                "total_income_lower_bound": [12_570],
                "total_income_upper_bound": [np.inf],
                "employment_income_amount": [4.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "get_la_age_targets",
        lambda: pd.DataFrame({"age/0_100": [1.0, 1.0, 1.0, 1.0]}),
    )
    monkeypatch.setattr(loss, "get_uk_total_population", lambda year: 4.0)
    monkeypatch.setattr(loss, "get_la_uc_targets", lambda: pd.Series([0, 1, 0, 0]))
    monkeypatch.setattr(
        loss,
        "get_ons_income_uprating_factors",
        lambda year: (1.0, 1.0),
    )
    monkeypatch.setattr(
        loss,
        "load_household_counts",
        lambda: pd.DataFrame(
            {
                "la_code": ["E06000001", "W06000001"],
                "households": [100.0, 200.0],
            }
        ),
    )
    return loss


def test_la_loss_masks_missing_ons_income_cells(monkeypatch, tmp_path):
    loss = _patch_common_la_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        loss,
        "load_ons_la_income",
        lambda: pd.DataFrame(
            {
                "la_code": ["E06000001", "W06000001"],
                "net_income_bhc": [30_000.0, 25_000.0],
                "net_income_ahc": [26_000.0, 21_000.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "load_tenure_data",
        lambda: pd.DataFrame(
            {
                "la_code": ["E06000001"],
                "owned_outright_pct": [30.0],
                "owned_mortgage_pct": [30.0],
                "private_rent_pct": [25.0],
                "social_rent_pct": [15.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "load_private_rents",
        lambda: pd.DataFrame(
            {"area_code": ["E06000001"], "median_annual_rent": [12_000.0]}
        ),
    )

    _, y, _ = loss.create_local_authority_target_matrix(_FakeDataset())

    direct = y["ons/equiv_net_income_bhc"].iloc[:2]
    missing = y["ons/equiv_net_income_bhc"].iloc[2:]
    assert direct.notna().all()
    assert missing.isna().all()


def test_la_loss_masks_missing_tenure_and_rent_cells(monkeypatch, tmp_path):
    loss = _patch_common_la_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(
        loss,
        "load_ons_la_income",
        lambda: pd.DataFrame(
            {
                "la_code": ["E06000001", "W06000001"],
                "net_income_bhc": [30_000.0, 25_000.0],
                "net_income_ahc": [26_000.0, 21_000.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "load_tenure_data",
        lambda: pd.DataFrame(
            {
                "la_code": ["E06000001"],
                "owned_outright_pct": [30.0],
                "owned_mortgage_pct": [30.0],
                "private_rent_pct": [25.0],
                "social_rent_pct": [15.0],
            }
        ),
    )
    monkeypatch.setattr(
        loss,
        "load_private_rents",
        lambda: pd.DataFrame(
            {"area_code": ["E06000001"], "median_annual_rent": [12_000.0]}
        ),
    )

    _, y, _ = loss.create_local_authority_target_matrix(_FakeDataset())

    for column in [
        "tenure/owned_outright",
        "tenure/owned_mortgage",
        "tenure/private_rent",
        "tenure/social_rent",
        "rent/private_rent",
    ]:
        assert pd.notna(y[column].iloc[0]), f"{column}: direct cell should be finite"
        assert y[column].iloc[1:].isna().all(), (
            f"{column}: missing-source cells should be masked"
        )
