from policyengine_uk_data.datasets.private_releases import (
    CURRENT_ETB_RELEASE,
    CURRENT_LCFS_RELEASE,
    CURRENT_WAS_RELEASE,
)


def test_current_lcfs_release_points_to_2023_24_ukds_files():
    assert CURRENT_LCFS_RELEASE.name == "lcfs_2023_24"
    assert CURRENT_LCFS_RELEASE.ukds_study_number == 9468
    assert CURRENT_LCFS_RELEASE.doi == "10.5255/UKDA-SN-9468-3"
    assert CURRENT_LCFS_RELEASE.household_tab_filename == "9468_dvhh_ukanon_v2_2023.tab"
    assert (
        CURRENT_LCFS_RELEASE.person_tab_filename == "9468_dvper_ukanon_202324_2023.tab"
    )
    assert CURRENT_LCFS_RELEASE.fuel_price_year == 2023


def test_current_was_release_points_to_round_8_ukds_files():
    assert CURRENT_WAS_RELEASE.name == "was_2006_22"
    assert CURRENT_WAS_RELEASE.latest_round == 8
    assert CURRENT_WAS_RELEASE.ukds_study_number == 7215
    assert CURRENT_WAS_RELEASE.doi == "10.5255/UKDA-SN-7215-20"
    assert (
        CURRENT_WAS_RELEASE.household_tab_filename
        == "7215_was_round_8_hhold_eul_may_2025_230525.tab"
    )


def test_current_etb_release_points_to_2023_24_ukds_files():
    assert CURRENT_ETB_RELEASE.name == "etb_1977_24"
    assert CURRENT_ETB_RELEASE.latest_year == 2024
    assert CURRENT_ETB_RELEASE.default_training_year == 2023
    assert CURRENT_ETB_RELEASE.ukds_study_number == 8856
    assert CURRENT_ETB_RELEASE.doi == "10.5255/UKDA-SN-8856-4"
    assert (
        CURRENT_ETB_RELEASE.household_tab_filename == "8856_householdv2_1977-2024.tab"
    )


def test_consumption_model_metadata_tracks_private_releases():
    from policyengine_uk_data.datasets.imputations.consumption import (
        CONSUMPTION_MODEL_FILENAME,
        get_consumption_model_metadata,
        get_has_fuel_model_metadata,
    )

    metadata = get_consumption_model_metadata()
    has_fuel_metadata = get_has_fuel_model_metadata()

    assert CURRENT_LCFS_RELEASE.name in CONSUMPTION_MODEL_FILENAME
    assert CURRENT_WAS_RELEASE.name in CONSUMPTION_MODEL_FILENAME
    assert metadata["lcfs_release_name"] == CURRENT_LCFS_RELEASE.name
    assert metadata["was_release_name"] == CURRENT_WAS_RELEASE.name
    assert has_fuel_metadata["was_release_name"] == CURRENT_WAS_RELEASE.name


def test_etb_model_metadata_tracks_private_release():
    from policyengine_uk_data.datasets.imputations.services.etb import (
        get_public_services_model_metadata,
    )
    from policyengine_uk_data.datasets.imputations.vat import (
        DEFAULT_ETB_YEAR,
        get_vat_model_metadata,
    )

    vat_metadata = get_vat_model_metadata()
    services_metadata = get_public_services_model_metadata()

    assert DEFAULT_ETB_YEAR == CURRENT_ETB_RELEASE.default_training_year
    assert vat_metadata["etb_release_name"] == CURRENT_ETB_RELEASE.name
    assert services_metadata["etb_release_name"] == CURRENT_ETB_RELEASE.name
