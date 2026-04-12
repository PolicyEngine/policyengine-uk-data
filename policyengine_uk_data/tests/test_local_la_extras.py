from types import SimpleNamespace

import pytest

from policyengine_uk_data.targets.sources import local_la_extras


def test_get_ons_income_uprating_factors_uses_legacy_2025_fallback():
    assert local_la_extras.get_ons_income_uprating_factors(2025) == (
        1985.1 / 1467.6,
        103.5 / 84.9,
    )


def test_get_ons_income_uprating_factors_prefers_policyengine_uk_parameters(
    monkeypatch,
):
    params = SimpleNamespace(
        gov=SimpleNamespace(
            economic_assumptions=SimpleNamespace(
                local_authority_targets=SimpleNamespace(
                    ons_income=SimpleNamespace(
                        net_income_bhc_uprating_factor=lambda _: 1.5,
                        housing_costs_uprating_factor=lambda _: 1.25,
                    )
                )
            )
        )
    )

    monkeypatch.setattr(
        local_la_extras,
        "system",
        SimpleNamespace(parameters=params),
    )

    assert local_la_extras.get_ons_income_uprating_factors(2025) == (1.5, 1.25)


def test_get_ons_income_uprating_factors_raises_for_unknown_year(monkeypatch):
    params = SimpleNamespace(
        gov=SimpleNamespace(economic_assumptions=SimpleNamespace())
    )
    monkeypatch.setattr(
        local_la_extras,
        "system",
        SimpleNamespace(parameters=params),
    )

    with pytest.raises(ValueError, match="No ONS LA income uprating factors"):
        local_la_extras.get_ons_income_uprating_factors(2024)
