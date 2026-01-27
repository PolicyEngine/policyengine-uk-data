"""DWP Stat-Xplore data source.

Downloads benefit statistics from the DWP Stat-Xplore Open Data API.
"""

import os
from datetime import date
from typing import Optional

import httpx
from dagster import asset, AssetExecutionContext


class StatXploreClient:
    """Client for DWP Stat-Xplore API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://stat-xplore.dwp.gov.uk/webapi/rest/v1",
        timeout: float = 120.0,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self._client = httpx.Client(
            base_url=base_url,
            headers={"APIKey": api_key},
            timeout=timeout,
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._client.close()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def get_schema(self, schema_id: Optional[str] = None) -> dict:
        """Get schema information."""
        url = "/schema"
        if schema_id:
            url = f"{url}/{schema_id}"

        response = self._client.get(url)
        response.raise_for_status()
        return response.json()

    def query_table(self, query: dict) -> dict:
        """Execute a table query."""
        response = self._client.post(
            "/table",
            json=query,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        return response.json()

    def get_rate_limit(self) -> dict:
        """Get current rate limit status."""
        response = self._client.get("/rate_limit")
        response.raise_for_status()
        return response.json()


# Database and field identifiers discovered from API exploration

# PIP (Personal Independence Payment)
PIP_DATABASE = "str:database:PIP_Monthly_new"
PIP_COUNT_MEASURE = "str:count:PIP_Monthly_new:V_F_PIP_MONTHLY"
PIP_DL_AWARD_FIELD = "str:field:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE"
PIP_DATE_FIELD = "str:field:PIP_Monthly_new:F_PIP_DATE:DATE2"

# PIP Daily Living Award values
PIP_DL_ENHANCED = "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:1"
PIP_DL_STANDARD = "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:2"
PIP_DL_NIL = "str:value:PIP_Monthly_new:V_F_PIP_MONTHLY:DL_AWARD_TYPE:C_PIP_DL_AWARD_TYPE:3"

# Universal Credit Households
UC_HOUSEHOLDS_DATABASE = "str:database:UC_Households"
UC_HOUSEHOLDS_COUNT_MEASURE = "str:count:UC_Households:V_F_UC_HOUSEHOLDS"
UC_HH_DATE_FIELD = "str:field:UC_Households:F_UC_HH_DATE:DATE_NAME"
UC_HH_YOUNGEST_CHILD_AGE = "str:field:UC_Households:V_F_UC_HOUSEHOLDS:YOUNGEST_CHILD_AGE"

# Benefit Cap - Housing Benefit claimants
BENEFIT_CAP_HB_DATABASE = "str:database:Benefit_Cap_Monthly_2011"
BENEFIT_CAP_HB_COUNT = "str:count:Benefit_Cap_Monthly_2011:V_F_BC_MONTHLY_FULL_2011"
BENEFIT_CAP_HB_AMOUNT = "str:statfn:Benefit_Cap_Monthly_2011:V_F_BC_MONTHLY_FULL_2011:CABENCAP:MEAN"

# Benefit Cap - Universal Credit claimants
BENEFIT_CAP_UC_DATABASE = "str:database:BC_UC_Monthly"
BENEFIT_CAP_UC_COUNT = "str:count:BC_UC_Monthly:V_F_UCFS_MONTHLY"
BENEFIT_CAP_UC_AMOUNT = "str:statfn:BC_UC_Monthly:V_F_UCFS_MONTHLY:weekly_cap:MEAN"


def parse_stat_xplore_response(
    result: dict,
    metric_mappings: dict[str, str],
    snapshot_date: date,
    source: str,
) -> list[dict]:
    """Parse Stat-Xplore API response into observations.

    Args:
        result: API response with cubes and fields
        metric_mappings: Map from label text to metric_code
        snapshot_date: Publication date for this data
        source: Source description
    """
    observations = []

    # Extract measure URI and values
    cubes = result.get("cubes", {})
    if not cubes:
        return observations

    measure_uri = list(cubes.keys())[0]
    cube_data = cubes[measure_uri]
    values = cube_data.get("values", [])

    # Extract field structure
    fields = result.get("fields", [])

    # Parse dimensions and their items
    # values is nested array matching dimension structure
    # For 2D: [[val1, val2], [val3, val4]] = dimension1 x dimension2

    if len(fields) == 1:
        # Single dimension (e.g., just award types)
        field = fields[0]
        items = field.get("items", [])

        for idx, item in enumerate(items):
            label = item.get("labels", [""])[0]
            metric_code = metric_mappings.get(label)

            if metric_code and idx < len(values[0]):
                value = values[0][idx]
                observations.append({
                    "metric_code": metric_code,
                    "area_code": "UK",
                    "valid_year": snapshot_date.year,
                    "snapshot_date": snapshot_date.isoformat(),
                    "value": float(value),
                    "source": source,
                    "source_url": "https://stat-xplore.dwp.gov.uk/",
                    "is_forecast": False,
                })

    elif len(fields) == 2:
        # Two dimensions (e.g., date x award type)
        date_field = fields[0]
        category_field = fields[1]

        date_items = date_field.get("items", [])
        category_items = category_field.get("items", [])

        for date_idx, date_item in enumerate(date_items):
            date_label = date_item.get("labels", [""])[0]

            # Parse year from date label (e.g., "Oct-25" -> 2025)
            try:
                if "-" in date_label:
                    year = 2000 + int(date_label.split("-")[1])
                else:
                    year = int(date_label[:4])
            except (ValueError, IndexError):
                year = snapshot_date.year

            for cat_idx, cat_item in enumerate(category_items):
                label = cat_item.get("labels", [""])[0]
                metric_code = metric_mappings.get(label)

                if metric_code and date_idx < len(values) and cat_idx < len(values[date_idx]):
                    value = values[date_idx][cat_idx]
                    observations.append({
                        "metric_code": metric_code,
                        "area_code": "UK",
                        "valid_year": year,
                        "snapshot_date": snapshot_date.isoformat(),
                        "value": float(value),
                        "source": source,
                        "source_url": "https://stat-xplore.dwp.gov.uk/",
                        "is_forecast": False,
                    })

    return observations


def query_pip_claimants(
    client: StatXploreClient, snapshot_date: date, limit_months: int = 36
) -> list[dict]:
    """Query PIP claimants by daily living component rate.

    Returns observations for:
    - pip_dl_standard_claimants
    - pip_dl_enhanced_claimants

    Args:
        client: Stat-Xplore API client
        snapshot_date: Snapshot date for observations
        limit_months: Number of recent months to fetch (default 36 = 3 years)
    """
    # Query with minimal dimensions to avoid timeout
    # Default query returns latest month only, we need to be selective
    query = {
        "database": PIP_DATABASE,
        "measures": [PIP_COUNT_MEASURE],
        "dimensions": [
            [PIP_DL_AWARD_FIELD],
        ],
    }

    result = client.query_table(query)

    # Map API labels to our metric codes
    metric_mappings = {
        "Daily Living - Enhanced": "pip_dl_enhanced_claimants",
        "Daily Living - Standard": "pip_dl_standard_claimants",
    }

    return parse_stat_xplore_response(
        result,
        metric_mappings,
        snapshot_date,
        "DWP Stat-Xplore PIP Statistics",
    )


def query_simple_caseload(
    client: StatXploreClient,
    snapshot_date: date,
    database_id: str,
    measure_id: str,
    metric_code: str,
    source_name: str,
) -> list[dict]:
    """Query a simple caseload count from a database.

    Args:
        client: Stat-Xplore API client
        snapshot_date: Snapshot date for observations
        database_id: Database ID (e.g., 'str:database:SP_New')
        measure_id: Count measure ID (e.g., 'str:count:SP_New:V_F_SP_CASELOAD_New')
        metric_code: Metric code for output (e.g., 'state_pension_recipients')
        source_name: Human-readable source name
    """
    query = {
        "database": database_id,
        "measures": [measure_id],
        "dimensions": [],
    }

    result = client.query_table(query)
    cubes = result.get("cubes", {})

    if not cubes:
        return []

    value = list(cubes.values())[0]["values"][0]

    return [{
        "metric_code": metric_code,
        "area_code": "UK",
        "valid_year": snapshot_date.year,
        "snapshot_date": snapshot_date.isoformat(),
        "value": float(value),
        "source": source_name,
        "source_url": "https://stat-xplore.dwp.gov.uk/",
        "is_forecast": False,
    }]


def query_all_benefit_caseloads(
    client: StatXploreClient, snapshot_date: date
) -> list[dict]:
    """Query caseload statistics for all major benefits.

    Returns observations for:
    - state_pension_recipients
    - universal_credit_people
    - universal_credit_households
    - pension_credit_claimants
    - attendance_allowance_claimants
    - jsa_claimants
    - esa_claimants
    - housing_benefit_claimants
    - dla_claimants
    - carers_allowance_claimants
    - income_support_claimants
    - winter_fuel_payment_recipients
    """
    all_observations = []

    # Define all benefit caseload queries
    caseload_queries = [
        ("str:database:SP_New", "str:count:SP_New:V_F_SP_CASELOAD_New",
         "state_pension_recipients", "DWP Stat-Xplore State Pension"),
        ("str:database:UC_Monthly", "str:count:UC_Monthly:V_F_UC_CASELOAD_FULL",
         "universal_credit_people", "DWP Stat-Xplore Universal Credit"),
        ("str:database:UC_Households", "str:count:UC_Households:V_F_UC_HOUSEHOLDS",
         "universal_credit_households", "DWP Stat-Xplore Universal Credit"),
        ("str:database:PC_New", "str:count:PC_New:V_F_PC_CASELOAD_New",
         "pension_credit_claimants", "DWP Stat-Xplore Pension Credit"),
        ("str:database:AA_In_Payment_New", "str:count:AA_In_Payment_New:V_F_AA_In_Payment_New",
         "attendance_allowance_claimants", "DWP Stat-Xplore Attendance Allowance"),
        ("str:database:JSA", "str:count:JSA:V_F_JSA",
         "jsa_claimants", "DWP Stat-Xplore Jobseekers Allowance"),
        ("str:database:ESA_Caseload_new", "str:count:ESA_Caseload_new:V_F_ESA_NEW",
         "esa_claimants", "DWP Stat-Xplore Employment and Support Allowance"),
        ("str:database:hb_new", "str:count:hb_new:V_F_HB_NEW",
         "housing_benefit_claimants", "DWP Stat-Xplore Housing Benefit"),
        ("str:database:DLA_In_Payment_New", "str:count:DLA_In_Payment_New:V_F_DLA_In_Payment_New",
         "dla_claimants", "DWP Stat-Xplore Disability Living Allowance"),
        ("str:database:CA_In_Payment_New", "str:count:CA_In_Payment_New:V_F_CA_In_Payment_New",
         "carers_allowance_claimants", "DWP Stat-Xplore Carers Allowance"),
        ("str:database:IS", "str:count:IS:V_F_IS",
         "income_support_claimants", "DWP Stat-Xplore Income Support"),
        ("str:database:WFP", "str:count:WFP:V_F_WFP",
         "winter_fuel_payment_recipients", "DWP Stat-Xplore Winter Fuel Payment"),
    ]

    for db_id, measure_id, metric_code, source in caseload_queries:
        observations = query_simple_caseload(
            client, snapshot_date, db_id, measure_id, metric_code, source
        )
        all_observations.extend(observations)

    return all_observations


def query_benefit_cap(
    client: StatXploreClient, snapshot_date: date
) -> list[dict]:
    """Query benefit cap statistics from both HB and UC databases.

    Returns observations for:
    - benefit_capped_households (HB + UC)
    - benefit_cap_total_reduction (HB + UC)
    """
    # Query HB caseload
    query_hb = {
        "database": BENEFIT_CAP_HB_DATABASE,
        "measures": [BENEFIT_CAP_HB_COUNT],
        "dimensions": [],
    }

    result_hb = client.query_table(query_hb)
    cubes_hb = result_hb.get("cubes", {})
    hb_households = 0
    if cubes_hb:
        values = list(cubes_hb.values())[0].get("values", [])
        hb_households = float(values[0]) if values else 0

    # Query HB mean amount
    query_hb_amount = {
        "database": BENEFIT_CAP_HB_DATABASE,
        "measures": [BENEFIT_CAP_HB_AMOUNT],
        "dimensions": [],
    }

    result_hb_amount = client.query_table(query_hb_amount)
    cubes_hb_amount = result_hb_amount.get("cubes", {})
    hb_mean_weekly = 0
    if cubes_hb_amount:
        values = list(cubes_hb_amount.values())[0].get("values", [])
        hb_mean_weekly = float(values[0]) if values else 0

    # Query UC caseload
    query_uc = {
        "database": BENEFIT_CAP_UC_DATABASE,
        "measures": [BENEFIT_CAP_UC_COUNT],
        "dimensions": [],
    }

    result_uc = client.query_table(query_uc)
    cubes_uc = result_uc.get("cubes", {})
    uc_households = 0
    if cubes_uc:
        values = list(cubes_uc.values())[0].get("values", [])
        uc_households = float(values[0]) if values else 0

    # Query UC mean amount
    query_uc_amount = {
        "database": BENEFIT_CAP_UC_DATABASE,
        "measures": [BENEFIT_CAP_UC_AMOUNT],
        "dimensions": [],
    }

    result_uc_amount = client.query_table(query_uc_amount)
    cubes_uc_amount = result_uc_amount.get("cubes", {})
    uc_mean_weekly = 0
    if cubes_uc_amount:
        values = list(cubes_uc_amount.values())[0].get("values", [])
        uc_mean_weekly = float(values[0]) if values else 0

    # Combine HB and UC
    total_households = hb_households + uc_households
    total_weekly_reduction = (hb_households * hb_mean_weekly) + (uc_households * uc_mean_weekly)
    total_annual_reduction = total_weekly_reduction * 52

    observations = [
        {
            "metric_code": "benefit_capped_households",
            "area_code": "UK",
            "valid_year": snapshot_date.year,
            "snapshot_date": snapshot_date.isoformat(),
            "value": total_households,
            "source": "DWP Stat-Xplore Benefit Cap Statistics",
            "source_url": "https://stat-xplore.dwp.gov.uk/",
            "is_forecast": False,
        },
        {
            "metric_code": "benefit_cap_total_reduction",
            "area_code": "UK",
            "valid_year": snapshot_date.year,
            "snapshot_date": snapshot_date.isoformat(),
            "value": total_annual_reduction,
            "source": "DWP Stat-Xplore Benefit Cap Statistics",
            "source_url": "https://stat-xplore.dwp.gov.uk/",
            "is_forecast": False,
        },
    ]

    return observations


@asset(group_name="targets")
def dwp_stat_xplore_observations(context: AssetExecutionContext) -> list[dict]:
    """Download and parse DWP Stat-Xplore benefit statistics.

    Fetches caseload statistics for all major benefits:
    - State Pension recipients
    - Universal Credit people and households
    - Pension Credit claimants
    - Attendance Allowance claimants
    - Jobseekers Allowance claimants
    - Employment and Support Allowance claimants
    - Housing Benefit claimants
    - Disability Living Allowance claimants
    - Personal Independence Payment claimants (Daily Living components)
    - Carers Allowance claimants
    - Income Support claimants
    - Winter Fuel Payment recipients
    - Benefit cap statistics (households capped and total reduction)

    Note: UC two-child limit and Scotland child under 1 metrics are not available
    via Stat-Xplore API and require separate data sources.

    Requires STAT_XPLORE_API_KEY environment variable.
    """
    api_key = os.getenv("STAT_XPLORE_API_KEY")
    if not api_key:
        context.log.error("STAT_XPLORE_API_KEY environment variable not set")
        return []

    client = StatXploreClient(api_key)
    snapshot_date = date.today()

    context.log.info(f"Querying DWP Stat-Xplore API (snapshot: {snapshot_date})...")

    all_observations = []

    # Check rate limit
    try:
        rate_limit = client.get_rate_limit()
        remaining = rate_limit.get("remaining", "unknown")
        limit = rate_limit.get("limit", "unknown")
        context.log.info(f"Rate limit: {remaining}/{limit} requests remaining")
    except Exception as e:
        context.log.warning(f"Could not fetch rate limit: {e}")

    # Query each metric group
    queries = [
        ("All benefit caseloads", query_all_benefit_caseloads),
        ("PIP claimants", query_pip_claimants),
        ("Benefit cap", query_benefit_cap),
    ]

    for name, query_func in queries:
        try:
            context.log.info(f"Querying {name}...")
            observations = query_func(client, snapshot_date)
            all_observations.extend(observations)
            context.log.info(f"  ✓ {name}: {len(observations)} observations")
        except Exception as e:
            context.log.error(f"  ✗ Failed to query {name}: {e}")
            import traceback
            context.log.error(traceback.format_exc())
            continue

    # Close the HTTP client
    client.close()

    context.log.info(f"\nTotal: {len(all_observations)} DWP Stat-Xplore observations")

    # Summary by metric
    if all_observations:
        metrics = {}
        for obs in all_observations:
            m = obs["metric_code"]
            metrics[m] = metrics.get(m, 0) + 1

        context.log.info("\nObservations by metric:")
        for metric, count in sorted(metrics.items()):
            context.log.info(f"  {metric}: {count} observations")

    return all_observations


# Exploration utility for development
def explore_schema(api_key: str, schema_id: Optional[str] = None):
    """Explore Stat-Xplore schema to find database and field IDs.

    Usage:
        from policyengine_uk_data.assets.sources.dwp_stat_xplore import explore_schema
        explore_schema("your_api_key_here")
    """
    client = StatXploreClient(api_key)
    schema = client.get_schema(schema_id)

    print(f"\nSchema ID: {schema.get('id', 'root')}")
    print(f"Label: {schema.get('label', 'N/A')}")
    print(f"Type: {schema.get('type', 'N/A')}")

    if "itemType" in schema:
        print(f"Item Type: {schema['itemType']}")

    if "items" in schema:
        print(f"\nItems ({len(schema['items'])}):")
        for item in schema["items"]:
            item_type = item.get("type", "unknown")
            item_id = item.get("id", "unknown")
            item_label = item.get("label", "unknown")
            print(f"  [{item_type}] {item_label}")
            print(f"    ID: {item_id}")

    return schema
