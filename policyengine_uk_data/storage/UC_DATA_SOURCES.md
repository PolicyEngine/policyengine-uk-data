# Universal Credit data sources

## National payment distribution

Source: Stat-Xplore (DWP)
- Rows: Monthly award amount bands + Households on Universal Credit
- Columns: Family type
- File: `uc_national_payment_dist.xlsx`

## Parliamentary constituency households

### Great Britain data

Source: Stat-Xplore (DWP)
- Rows: Westminster Parliamentary Constituency 2024 + Households on Universal Credit
- File: `uc_pc_households.xlsx`

### Northern Ireland data

Source: Department for Communities Northern Ireland
- URL: https://www.communities-ni.gov.uk/publications/universal-credit-statistics-may-2025
- File: `dfc-ni-uc-stats-supp-tables-may-2025.ods`
- Sheet: 5b
- Data: Household counts by Westminster Parliamentary Constituency 2024

The NI data is combined with the GB data to produce a complete UK-wide parliamentary constituency table.

## Data processing notes

- The "Unknown" constituency category is excluded from the constituency data
- Constituency household counts are scaled to match the national total from the payment distribution data, as the two sources have different totals due to timing and methodology differences
