"""Road-fuel volume series used for petrol and diesel uprating.

The source series is fiscal-year UK petrol plus diesel clearances, measured in
million litres. Historical outturn comes directly from HMRC Hydrocarbon Oils
Bulletin Table 2a, using ``Total petrol`` + ``Total diesel``. Forecast years use
OBR March 2026 fuel-duty receipts, net of non-road fuel receipts, divided by the
standard petrol and diesel duty rate for the fiscal year.

Sources:
- HMRC Hydrocarbon Oils Bulletin tables, July to December 2025 release.
- OBR March 2026 EFO detailed forecast tables: receipts, Table 3.8.
- OBR April 2024 supplementary release, fuel duty receipts by vehicle type,
  for the non-road fuel receipts split.
"""

HMRC_HYDROCARBON_OILS_BULLETIN_URL = (
    "https://www.gov.uk/government/statistics/hydrocarbon-oils-bulletin"
)
OBR_MARCH_2026_RECEIPTS_URL = (
    "https://obr.uk/efo/economic-and-fiscal-outlook-march-2026/"
)
OBR_FUEL_DUTY_BY_VEHICLE_TYPE_URL = (
    "https://obr.uk/docs/dlm_uploads/"
    "Fuel-duty-supplementary-release_receipts-by-vehicle-type.pdf"
)

# Fiscal year beginning in the key, e.g. 2024 means 2024-25.
HMRC_ROAD_FUEL_CLEARANCES_MLITRES = {
    2020: 35_289.7611569628,
    2021: 43_906.907618977,
    2022: 46_653.9535006421,
    2023: 46_386.741837677,
    2024: 46_327.0970704816,
}

# OBR March 2026 EFO detailed forecast tables, receipts Table 3.8,
# "Fuel duties", fiscal years 2025-26 to 2030-31, GBP billions.
OBR_FUEL_DUTY_RECEIPTS_GBP_BN = {
    2025: 24.241874775213375,
    2026: 24.628571426324807,
    2027: 26.545622366266198,
    2028: 26.63575593480781,
    2029: 26.382076806202907,
    2030: 25.740748126281627,
}

# OBR April 2024 supplementary fuel-duty release, "Other fuels" row, GBP
# billions. March 2026 did not republish the road/other split, so use the
# latest explicit split to remove non-road fuel receipts before converting
# receipts to litres. Hold the final published split flat after 2028-29.
NON_ROAD_FUEL_RECEIPTS_GBP_BN = {
    2025: 0.2,
    2026: 0.3,
    2027: 0.3,
    2028: 0.3,
    2029: 0.3,
    2030: 0.3,
}

# Fiscal-year average statutory petrol/diesel duty rates, GBP per litre.
# Rates follow PolicyEngine UK's fuel-duty parameter comments for the staged
# 5p reversal and subsequent RPI uprating.
FISCAL_YEAR_AVERAGE_DUTY_RATE = {
    2025: 0.5295,
    2026: (0.5295 * 153 + 0.5395 * 91 + 0.5595 * 90 + 0.5795 * 31) / 365,
    2027: 0.6010,
    2028: 0.6198,
    2029: 0.6376,
    2030: 0.6562,
}


def forecast_road_fuel_clearances_mlitres() -> dict[int, float]:
    """Convert OBR road-fuel receipts forecasts into physical clearances."""
    return {
        year: (
            OBR_FUEL_DUTY_RECEIPTS_GBP_BN[year] - NON_ROAD_FUEL_RECEIPTS_GBP_BN[year]
        )
        * 1_000
        / FISCAL_YEAR_AVERAGE_DUTY_RATE[year]
        for year in OBR_FUEL_DUTY_RECEIPTS_GBP_BN
    }


def road_fuel_clearances_mlitres(end_year: int | None = None) -> dict[int, float]:
    """Return road-fuel clearances, carrying the last forecast year forward."""
    series = {
        **HMRC_ROAD_FUEL_CLEARANCES_MLITRES,
        **forecast_road_fuel_clearances_mlitres(),
    }
    if end_year is None:
        return series

    last_year = max(series)
    last_value = series[last_year]
    for year in range(last_year + 1, end_year + 1):
        series[year] = last_value
    return series


def road_fuel_volume_index(
    base_year: int = 2020,
    end_year: int | None = None,
) -> dict[int, float]:
    """Return road-fuel clearances rebased to ``base_year`` = 1.0."""
    series = road_fuel_clearances_mlitres(end_year=end_year)
    base = series[base_year]
    return {year: value / base for year, value in series.items()}


def road_fuel_volume_uprating(start_year: int, end_year: int) -> float:
    """Return the relative road-fuel volume change between two fiscal years."""
    index = road_fuel_volume_index(base_year=start_year, end_year=end_year)
    return index[end_year]
