# DWP Stat-Xplore implementation summary

## Implemented metrics (16 benefit caseload statistics)

✅ **Successfully implemented via Stat-Xplore API:**

### State benefits caseload
1. **state_pension_recipients** - State Pension recipients
   - Source: `str:database:SP_New`
   - Latest data: 13,116,578 recipients (2026)

2. **universal_credit_people** - People on Universal Credit
   - Source: `str:database:UC_Monthly`
   - Latest data: 8,400,344 people (2026)

3. **universal_credit_households** - Households on Universal Credit
   - Source: `str:database:UC_Households`
   - Latest data: 6,862,289 households (2026)

4. **pension_credit_claimants** - Pension Credit claimants
   - Source: `str:database:PC_New`
   - Latest data: 1,391,075 claimants (2026)

### Disability and care benefits
5. **attendance_allowance_claimants** - Attendance Allowance claimants
   - Source: `str:database:AA_In_Payment_New`
   - Latest data: 1,749,547 claimants (2026)

6. **dla_claimants** - Disability Living Allowance claimants
   - Source: `str:database:DLA_In_Payment_New`
   - Latest data: 1,409,787 claimants (2026)

7. **pip_dl_standard_claimants** - PIP Daily Living Standard rate claimants
   - Source: `str:database:PIP_Monthly_new`
   - Latest data: 1,685,421 claimants (2026)

8. **pip_dl_enhanced_claimants** - PIP Daily Living Enhanced rate claimants
   - Source: `str:database:PIP_Monthly_new`
   - Latest data: 2,035,617 claimants (2026)

9. **carers_allowance_claimants** - Carers Allowance claimants
   - Source: `str:database:CA_In_Payment_New`
   - Latest data: 986,370 claimants (2026)

### Employment support benefits
10. **jsa_claimants** - Jobseekers Allowance claimants
    - Source: `str:database:JSA`
    - Latest data: 86,079 claimants (2026)

11. **esa_claimants** - Employment and Support Allowance claimants
    - Source: `str:database:ESA_Caseload_new`
    - Latest data: 1,175,221 claimants (2026)

12. **income_support_claimants** - Income Support claimants
    - Source: `str:database:IS`
    - Latest data: 1,419 claimants (2026)

### Housing and other benefits
13. **housing_benefit_claimants** - Housing Benefit claimants
    - Source: `str:database:hb_new`
    - Latest data: 1,596,629 claimants (2026)

14. **winter_fuel_payment_recipients** - Winter Fuel Payment recipients
    - Source: `str:database:WFP`
    - Latest data: 1,267,782 recipients (2026)

### Benefit cap statistics
15. **benefit_capped_households** - Households affected by benefit cap
    - Sources: `str:database:Benefit_Cap_Monthly_2011` (HB) + `str:database:BC_UC_Monthly` (UC)
    - Latest data: 119,347 households (2026)
    - Combines Housing Benefit and Universal Credit capped households

16. **benefit_cap_total_reduction** - Total annual benefit cap reduction
    - Sources: Same as above
    - Latest data: £356,342,548 per year
    - Calculated as: (HB households × HB mean weekly reduction + UC households × UC mean weekly reduction) × 52

❌ **Not available via Stat-Xplore API:**

- **uc_two_child_limit_children** - Children affected by two-child limit (not in Stat-Xplore databases)
- **uc_two_child_limit_households** - Households affected by two-child limit (not in Stat-Xplore databases)
- **scotland_uc_households_child_under_1** - UC households in Scotland with child under 1 (Stat-Xplore only has age bands "0-4", not "under 1")

## Files created/modified

**New files:**
- `dwp_stat_xplore.py` - Main data fetcher with query functions
- `explore_stat_xplore.py` - Interactive exploration tool
- `test_stat_xplore.py` - Connection test script
- `README_STAT_XPLORE.md` - Setup and usage guide
- `STAT_XPLORE_FINDINGS.md` - API exploration findings
- `DATA_SOURCES_PLAN.md` - Overall implementation roadmap
- `DWP_IMPLEMENTATION_SUMMARY.md` - This file

**Modified files:**
- `pyproject.toml` - Added httpx dependency
- `policyengine_uk_data/assets/__init__.py` - Added dwp_stat_xplore_observations asset
- `policyengine_uk_data/assets/sources/__init__.py` - Exported new asset
- `policyengine_uk_data/assets/targets.py` - Integrated dwp_stat_xplore into targets_db pipeline
- `policyengine_uk_data/definitions.py` - Added dwp_stat_xplore_observations to Dagster definitions
- `.env` - Added STAT_XPLORE_API_KEY

## Technical implementation

### Database structure

All databases follow a similar pattern with count measures and optional statistical measures:

**Example: Universal Credit**
- Database: `UC_Monthly`
- Count measure: `str:count:UC_Monthly:V_F_UC_CASELOAD_FULL`
- Latest month: January 2026

**Benefit Cap (special case):**
- Two separate databases:
  - Housing Benefit: `Benefit_Cap_Monthly_2011`
  - Universal Credit: `BC_UC_Monthly`
- Measures:
  - Count of capped households
  - Mean weekly reduction amount (requires MEAN statistical function)
- Data is combined programmatically

### Query patterns

Simple aggregated queries (no dimensions) to get latest snapshot:

```python
{
    "database": "str:database:SP_New",
    "measures": ["str:count:SP_New:V_F_SP_CASELOAD_New"],
    "dimensions": []
}
```

Returns latest month snapshot by default (currently January 2026).

### Code organization

The implementation uses a modular approach:

1. **`StatXploreClient`** - HTTP client wrapper with API key authentication
2. **`query_simple_caseload()`** - Generic function for simple count queries
3. **`query_all_benefit_caseloads()`** - Fetches all 12 major benefit caseloads
4. **`query_pip_claimants()`** - Special handling for PIP Daily Living components
5. **`query_benefit_cap()`** - Combines HB and UC benefit cap data
6. **`dwp_stat_xplore_observations`** - Main Dagster asset that orchestrates all queries

### Integration with Dagster

The `dwp_stat_xplore_observations` asset:
- Requires `STAT_XPLORE_API_KEY` environment variable
- Returns list of 16 observation dicts
- Logs progress and metrics summary
- Handles errors gracefully per metric group
- Closes HTTP client properly
- **Integrated into targets_db pipeline** as input dependency

## Usage

```bash
# Test connection
export STAT_XPLORE_API_KEY=<your_key>
python test_stat_xplore.py

# Explore databases
python explore_stat_xplore.py

# Run as Dagster asset
dagster asset materialize -m policyengine_uk_data.definitions --select dwp_stat_xplore_observations

# Build full targets database (includes Stat-Xplore data)
dagster asset materialize -m policyengine_uk_data.definitions --select targets_db
```

## Current limitations

1. **Single month snapshots only** - Queries return latest month to avoid timeouts
   - Could be enhanced to fetch multiple months with date dimension recodes
   - Would need careful pagination/filtering

2. **No historical time series** - Not implemented yet
   - API supports it but requires date dimension queries
   - Risk of timeouts with broad date ranges

3. **Limited to 16 metrics** - UC two-child limit and Scotland child under 1 not available
   - These require alternative data sources
   - See `DATA_SOURCES_PLAN.md` for other options

## Rate limits

- **Limit:** 10,000 requests per day
- **Current usage:** ~17 requests per execution (12 benefit caseloads + 2 PIP + 4 benefit cap queries)
- **Can run:** ~588 times per day

## Data quality notes

1. **Scotland PIP data** - Adult Disability Payment (ADP) launched in Scotland from March 2022, completed transfer by June 2025. Scotland PIP data may be minimal from mid-2025 onwards.

2. **Benefit cap timing** - Lower cap levels implemented November 2016, phased to January 2017. Data before March 2017 may not fully reflect lower cap levels.

3. **Benefit cap scope** - Combines Housing Benefit and Universal Credit capped households for complete picture.

4. **Income Support decline** - Only 1,419 claimants reflect transition to Universal Credit.

## Integration with targets pipeline

The Stat-Xplore observations are now fully integrated:

1. **Asset dependency**: `targets_db` asset depends on `dwp_stat_xplore_observations` via AssetIn
2. **Data flow**: Stat-Xplore data → `dwp_stat_xplore_observations` → `targets_db` → SQLite database
3. **Complementary data**: Provides caseload statistics to complement OBR spending forecasts

This means running `targets_db` will automatically fetch fresh benefit caseload data from Stat-Xplore and include it in the targets database alongside OBR, ONS, and other official statistics.

## Next steps

If needed:
1. Implement time series fetching with date filtering
2. Add retry logic for timeout handling
3. Implement caching to reduce API calls
4. Add UC two-child limit from separate DWP publications
5. Explore alternative sources for Scotland child under 1 metric
