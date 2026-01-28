# DWP Stat-Xplore integration

## Getting your API key

1. Go to [stat-xplore.dwp.gov.uk](https://stat-xplore.dwp.gov.uk/)
2. Register for a free account
3. Log in and click Account (top right)
4. Find your API key under "Open Data API Access"

## Setting up

Add your API key to `.env` file in the repo root:

```bash
STAT_XPLORE_API_KEY=your_key_here
```

Or export it as an environment variable:

```bash
export STAT_XPLORE_API_KEY=your_key_here
```

## Exploring the API

Use the interactive exploration script to discover database and field IDs:

```bash
cd policyengine_uk_data/assets/sources
python explore_stat_xplore.py
```

This will let you:
- List all available databases (118+ benefit datasets)
- Search for specific databases (e.g., "PIP", "universal credit")
- Explore database schemas (fields, measures, dimensions)
- Test queries and see response structures
- Check API rate limits

## Finding database IDs

Common databases we need:

1. **PIP (Personal Independence Payment)**
   - Search for: "PIP" or "personal independence"
   - Likely ID: `str:database:pip_monthly` or similar

2. **Universal Credit**
   - Search for: "universal credit"
   - Likely ID: `str:database:UC_Monthly`

3. **Benefit Cap**
   - Search for: "benefit cap"
   - Likely ID: `str:database:benefit_cap`

Use the interactive explorer to confirm exact IDs and find available fields.

## Building queries

Once you have database IDs, queries follow this pattern:

```python
query = {
    "database": "str:database:UC_Monthly",
    "measures": [
        "str:count:UC_Monthly:V_F_UC_MONTHLY:HOUSEHOLDS"
    ],
    "dimensions": [
        ["str:field:UC_Monthly:V_F_UC_MONTHLY:MONTH"],  # Time
        ["str:field:UC_Monthly:V_F_UC_MONTHLY:GEOGRAPHY"],  # Area
    ],
    "recodes": {
        "str:field:UC_Monthly:V_F_UC_MONTHLY:GEOGRAPHY": {
            "map": [["Scotland"]],  # Filter to Scotland only
            "total": False
        }
    }
}
```

## Implementing fetchers

After discovering the correct IDs:

1. Update the constants in `dwp_stat_xplore.py`:
   - Database IDs
   - Field IDs
   - Measure IDs

2. Implement the query parsing logic in each query function

3. Test locally with the exploration script

4. Run as Dagster asset to populate the database

## API documentation

- [Stat-Xplore Open Data API](https://stat-xplore.dwp.gov.uk/webapi/online-help/Open-Data-API.html)
- [Schema endpoint](https://stat-xplore.dwp.gov.uk/webapi/online-help/Open-Data-API-Schema.html)
- [Table queries](https://stat-xplore.dwp.gov.uk/webapi/online-help/Open-Data-API-Table.html)

## Rate limits

The API has rate limits. Check them with:

```python
from dwp_stat_xplore import StatXploreClient
client = StatXploreClient(api_key)
print(client.get_rate_limit())
```

Be respectful of rate limits during development and testing.
