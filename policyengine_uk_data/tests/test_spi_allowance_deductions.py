import pandas as pd


def test_spi_overrides_allowance_deductions_not_policy_parameters(tmp_path):
    from policyengine_uk_data.datasets.spi import create_spi

    row = {
        "SREF": 1,
        "FACT": 1,
        "DIVIDENDS": 0,
        "GIFTAID": 0,
        "GORCODE": 7,
        "INCBBS": 0,
        "INCPROP": 1_000,
        "PAY": 0,
        "EPB": 0,
        "EXPS": 0,
        "PENSION": 0,
        "PSAV_XS": 0,
        "PENSRLF": 0,
        "PROFITS": 1_000,
        "CAPALL": 0,
        "LOSSBF": 0,
        "AGERANGE": 3,
        "SRP": 0,
        "TAX_CRED": 0,
        "MOTHINC": 0,
        "INCPBEN": 0,
        "OSSBEN": 0,
        "TAXTERM": 0,
        "UBISJA": 0,
        "OTHERINC": 0,
        "GIFTINV": 0,
        "OTHERINV": 0,
        "COVNTS": 0,
        "MOTHDED": 0,
        "DEFICIEN": 0,
        "MCAS": 0,
        "BPADUE": 0,
        "MAIND": 0,
    }
    spi_path = tmp_path / "spi.tab"
    pd.DataFrame([row]).to_csv(spi_path, sep="\t", index=False)

    dataset = create_spi(spi_path, fiscal_year=2020, output_file_path=None)

    assert "trading_allowance" not in dataset.person
    assert "property_allowance" not in dataset.person
    assert dataset.person["trading_allowance_deduction"].iat[0] == 0
    assert dataset.person["property_allowance_deduction"].iat[0] == 0
