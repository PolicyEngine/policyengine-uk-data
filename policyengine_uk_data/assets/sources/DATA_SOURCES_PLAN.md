# Official data sources implementation plan

This document outlines the plan for converting manually-entered calibration targets to automated fetchers from official government sources.

## Current state

### Implemented (automated fetching)
- **OBR** (35 metrics): Economic and Fiscal Outlook releases with snapshot tracking ✓
  - Tax receipts (income tax, NI, VAT, corporation tax, duties, etc.)
  - Welfare spending (state pension, universal credit, benefits)
  - Council tax by country

### Not implemented (manual data or one-off loads)
- **DWP** (7 metrics): benefit cap, PIP claimants, two-child limit, UC households
- **HMRC** (4 metrics): salary sacrifice contributions and tax relief
- **ONS** (6 metrics): population, households, savings interest, Scotland demographics
- **VOA** (8 metrics): council tax bands A-H
- **NTS** (3 metrics): vehicle ownership rates
- **SSS** (1 metric): Scottish child payment
- **Housing** (2 metrics): private rent, mortgage payments

## Priority 1: DWP Stat-Xplore

**Metrics to implement:**
- `uc_two_child_limit_children` - children affected by two-child limit
- `uc_two_child_limit_households` - households affected by two-child limit
- `pip_dl_standard_claimants` - PIP daily living standard rate
- `pip_dl_enhanced_claimants` - PIP daily living enhanced rate
- `benefit_capped_households` - households affected by benefit cap
- `benefit_cap_total_reduction` - total annual benefit cap reduction
- `scotland_uc_households_child_under_1` - UC households in Scotland with child under 1

**Official source:** [DWP Stat-Xplore Open Data API](https://stat-xplore.dwp.gov.uk/webapi/online-help/Open-Data-API.html)

**API details:**
- REST API based on SuperSTAR 9.5 Open Data API
- Requires API key (free registration at [stat-xplore.dwp.gov.uk](https://stat-xplore.dwp.gov.uk/))
- Provides access to Universal Credit, PIP, benefit cap, and other DWP datasets
- Updated monthly (latest update: 20 January 2026)

**Technical approach:**
1. Register for API key and store in environment variable
2. Explore schema endpoint to identify relevant datasets and dimensions
3. Build queries for each metric using the table endpoint
4. Extract time series data with proper area/date dimensions
5. Store observations with snapshot_date tracking (like OBR)

**Python package available:** [statxplore on PyPI](https://pypi.org/project/statxplore/)

**Implementation file:** `policyengine_uk_data/assets/sources/dwp_stat_xplore.py`

## Priority 2: ONS API (Nomis)

**Metrics to implement:**
- `population` - total population
- `households` - number of households
- `savings_interest_income` - household interest income
- `scotland_babies_under_1` - babies under 1 in Scotland
- `scotland_children_under_16` - children under 16 in Scotland
- `scotland_households_3plus_children` - Scotland households with 3+ children

**Official source:** [Nomis - Official Census and Labour Market Statistics](https://www.nomisweb.co.uk/)

**API details:**
- Free to access, no registration required
- REST API returning HTML, CSV, or JSON
- Contains Census data, Labour Force Survey, population estimates
- [API documentation and workshops](https://www.eventbrite.co.uk/e/ons-local-workshop-how-to-use-the-nomis-api-tickets-1975279484110)

**Technical approach:**
1. Use discovery endpoints to find relevant datasets
2. Build queries for population and household statistics
3. Extract time series by geography (UK, Scotland, regions)
4. For Scotland-specific metrics, filter to Scottish areas
5. Store with appropriate snapshot dates

**R package reference:** [nomisr package](https://docs.evanodell.com/nomisr/) (for API patterns)

**Implementation file:** `policyengine_uk_data/assets/sources/ons_nomis.py`

## Priority 3: VOA council tax statistics

**Metrics to implement:**
- `ct_band_a` through `ct_band_h` - council tax band dwellings

**Official source:** [VOA Council Tax data on data.gov.uk](https://www.data.gov.uk/dataset/b13ddc99-acf2-4b80-856c-56924f959ef1/voa-council-tax--addresses-characteristics-and-attributes-of-properties_1)

**Data format:**
- CSV/Excel downloads (no dedicated API found)
- Published at [GOV.UK VOA statistics](https://www.gov.uk/government/organisations/valuation-office-agency/about/statistics)
- "Council Tax: stock of properties" annual publication

**Technical approach:**
1. Download latest "Council Tax: stock of properties" Excel file
2. Parse tables by band and geography
3. Extract England/Wales totals and regional breakdowns
4. Store as annual snapshots (published September each year)

**Implementation file:** `policyengine_uk_data/assets/sources/voa_council_tax.py`

## Priority 4: Scottish government statistics

**Metrics to implement:**
- `scottish_child_payment` - Scottish child payment expenditure

**Official source:** [statistics.gov.scot](https://statistics.gov.scot/home)

**API details:**
- SPARQL endpoint for linked open data
- [API documentation](https://guides.statistics.gov.scot/category/37-api)
- Open Government Licence, no registration required
- Over 250 datasets from Scottish government

**Additional source:** [Scottish Budget documents](https://www.gov.scot/budget/)
- Annual budget published each January (2026-27 published 13 January 2026)
- Contains Social Security Scotland spending forecasts

**Technical approach:**
1. Query statistics.gov.scot SPARQL endpoint for Social Security spending
2. Extract Scottish child payment expenditure by year
3. Supplement with annual budget document data if needed
4. Store with snapshot_date from publication date

**Python package available:** opendatascot R package (reference for API patterns)

**Implementation file:** `policyengine_uk_data/assets/sources/scotland.py`

## Priority 5: HMRC statistics

**Metrics to implement:**
- `salary_sacrifice_contributions` - total salary sacrifice contributions
- `salary_sacrifice_it_relief_additional` - additional rate tax relief
- `salary_sacrifice_it_relief_basic` - basic rate tax relief
- `salary_sacrifice_it_relief_higher` - higher rate tax relief

**Official source:** [HMRC statistics publications](https://www.gov.uk/government/statistics)

**Data format:**
- No dedicated API for statistics (HMRC Developer Hub APIs are for tax operations)
- Statistics published as Excel/ODS tables on GOV.UK
- Relevant publication: "Income Tax statistics and distributions"

**Technical approach:**
1. Download latest income tax statistics Excel file
2. Parse Table 6.2 or equivalent for salary sacrifice data
3. Extract time series and tax relief breakdowns
4. Store as annual snapshots aligned with publication dates

**Implementation file:** `policyengine_uk_data/assets/sources/hmrc_statistics.py`

## Priority 6: National Travel Survey

**Metrics to implement:**
- `no_vehicle_rate` - share of households with no vehicle
- `one_vehicle_rate` - share of households with one vehicle
- `two_plus_vehicle_rate` - share of households with 2+ vehicles

**Official source:** [National Travel Survey, Department for Transport](https://www.gov.uk/government/collections/national-travel-survey-statistics)

**Data format:**
- Excel/ODS tables published annually
- No API available
- Published on GOV.UK

**Technical approach:**
1. Download latest NTS statistical tables
2. Parse household vehicle ownership tables
3. Extract rates by year
4. Store as annual snapshots

**Implementation file:** `policyengine_uk_data/assets/sources/nts.py`

## Priority 7: Housing statistics

**Metrics to implement:**
- `rent_private` - total private rent payments
- `total_mortgage` - total mortgage payments

**Official source:** [ONS National Accounts](https://www.ons.gov.uk/economy/nationalaccounts)

**Technical approach:**
1. Access via ONS API or direct CSV downloads
2. Extract household expenditure on housing
3. Calculate totals for private rent and mortgage payments
4. Store as quarterly/annual snapshots

**Implementation file:** `policyengine_uk_data/assets/sources/ons_housing.py`

## Implementation roadmap

### Phase 1 (DWP focus)
1. Implement DWP Stat-Xplore fetcher
2. Add API key to environment variables
3. Create Dagster asset for DWP observations
4. Test with existing metrics
5. Verify snapshot tracking works

### Phase 2 (ONS focus)
1. Implement Nomis API fetcher
2. Create assets for population and household metrics
3. Add Scotland-specific queries
4. Test geographic filtering

### Phase 3 (remaining sources)
1. Implement VOA, Scottish government, HMRC, NTS, housing fetchers
2. Most will use direct file downloads rather than APIs
3. Add error handling for broken download links
4. Implement version tracking for publication dates

### Phase 4 (maintenance)
1. Add automated refresh schedules to Dagster
2. Set up monitoring for failed fetches
3. Document API keys and credentials needed
4. Create runbook for troubleshooting

## Technical patterns

All fetchers should follow the OBR pattern:
- Return `list[dict]` with observation schema
- Include `snapshot_date` for tracking forecast/publication evolution
- Include `source` and `source_url` for transparency
- Set `is_forecast` appropriately (future years beyond publication)
- Handle errors gracefully and log warnings
- Use requests with appropriate headers and timeouts

## API keys and credentials

Required for:
- DWP Stat-Xplore: API key from stat-xplore.dwp.gov.uk (free registration)

Not required for:
- ONS Nomis: open access
- Scottish government: open access
- VOA, HMRC, NTS: file downloads from GOV.UK

Store API keys in:
- `.env` file (not committed to git)
- Dagster environment configuration
- Environment variables in deployment

## Sources

- [DWP Stat-Xplore Open Data API](https://stat-xplore.dwp.gov.uk/webapi/online-help/Open-Data-API.html)
- [Nomis API](https://www.nomisweb.co.uk/)
- [VOA Statistics](https://www.gov.uk/government/organisations/valuation-office-agency/about/statistics)
- [statistics.gov.scot](https://statistics.gov.scot/home)
- [HMRC Developer Hub](https://developer.service.hmrc.gov.uk/)
- [ONS API Documentation](https://www.ons.gov.uk/aboutus/whatwedo/programmesandprojects/censusanddatacollectiontransformationprogramme)
