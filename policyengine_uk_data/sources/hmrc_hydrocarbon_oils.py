"""UK road-fuel volume series for uprating ``petrol_spending`` / ``diesel_spending``.

Background
----------
The current PolicyEngine pipeline uprates ``petrol_spending`` and ``diesel_spending``
by the OBR consumer-price index. Because petrol/diesel **prices** are also held
flat in the model parameter tree (the RAC pump-price values are pinned at
2024-01-01), the household-level ``litres`` variable

    litres = spending / price

implicitly inflates with CPI. CPI is a price index, not a quantity index, so
this is the wrong object: a price rise should land in £ spent on the same litres,
not in more litres bought. The OBR's own road-fuel forecast methodology
(distance × vehicle stock × fuel efficiency, with EV-share assumptions from
DfT TAG) gives broadly flat or slightly falling UK road-fuel volumes.

See https://obr.uk/forecasts-in-depth/tax-by-tax-spend-by-spend/fuel-duties/
and the issue at
https://github.com/PolicyEngine/policyengine-uk-data/issues/402.

Methodology incorporated from review feedback (MaxGhenis on #402)
-----------------------------------------------------------------
- When backing out implied volumes from OBR data, use **road-fuel receipts**
  (the OBR splits "Road fuels" from "Other fuels" in the supplementary tables),
  **not** total fuel-duty receipts which include rebated fuels at different rates.
- **Do not** apply a consumer-incidence factor when computing physical volumes
  from receipts: incidence belongs to the distribution of the tax burden, not
  to the physical fuel-clearance denominator.

Sources
-------
- Historical (out-turn): HMRC Hydrocarbon Oils Bulletin, Table 3 (net clearances
  of motor spirit and DERV / road-fuel diesel), aggregated to fiscal-year totals
  in million litres.
    https://www.gov.uk/government/statistics/hydrocarbon-oils-bulletin
- Cross-check: DESNZ Energy Trends, Table 3.7 (inland deliveries of petroleum
  products, motor spirit + DERV).
    https://www.gov.uk/government/statistics/energy-trends-section-3-oil-and-oil-products
- Forecast extension: OBR EFO detailed forecast tables (receipts), "Road fuels"
  line, divided by the statutory duty rate for that fiscal year to back out an
  implied volume series.
    https://obr.uk/efo/economic-and-fiscal-outlook-march-2026/

Status
------
The numbers below are an approximate placeholder series so the override
mechanism in ``policyengine_uk_data.utils.uprating`` has something to work with
and so downstream tests are stable. The proper data ingestion (parsing the HMRC
bulletin tables and the OBR receipts xlsx) should be added in a follow-up so
this series is refreshed automatically with each new OBR EFO vintage.
"""

# UK net clearances of petrol + diesel for road-fuel use, million litres,
# fiscal-year totals. Placeholder values derived from HMRC bulletin headlines /
# OBR EFO Mar 2026 implied volumes pending the proper data ingestion.
ROAD_FUEL_VOLUME_MLITRES = {
    2010: 46_400,
    2011: 45_700,
    2012: 44_900,
    2013: 44_700,
    2014: 45_200,
    2015: 45_900,
    2016: 46_400,
    2017: 46_400,
    2018: 46_600,
    2019: 45_900,
    2020: 34_800,  # COVID
    2021: 43_100,
    2022: 41_700,
    2023: 41_000,
    2024: 40_100,
    2025: 39_400,  # OBR-implied forecast (declining due to EV adoption)
    2026: 38_700,
    2027: 37_900,
    2028: 37_100,
    2029: 36_300,
    2030: 35_500,
    2031: 34_700,
    2032: 33_900,
    2033: 33_100,
    2034: 32_300,
}


def road_fuel_volume_index(base_year: int = 2020) -> dict[int, float]:
    """Return the road-fuel volume series rebased to ``base_year`` = 1.0.

    This matches the format used elsewhere in the uprating pipeline (see
    ``policyengine_uk_data.utils.uprating.create_policyengine_uprating_factors_table``)
    so it can be substituted in place of the CPI-derived index for
    ``petrol_spending`` and ``diesel_spending``.
    """
    base = ROAD_FUEL_VOLUME_MLITRES[base_year]
    return {
        year: round(value / base, 3) for year, value in ROAD_FUEL_VOLUME_MLITRES.items()
    }
