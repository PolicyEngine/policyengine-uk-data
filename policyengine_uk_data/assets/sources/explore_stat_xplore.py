"""Interactive script to explore DWP Stat-Xplore schema and test queries.

Usage:
    export STAT_XPLORE_API_KEY=your_key_here
    python -m policyengine_uk_data.assets.sources.explore_stat_xplore
"""

import os
import sys
import json
from datetime import date

from dwp_stat_xplore import StatXploreClient


def print_schema_tree(schema: dict, indent: int = 0):
    """Pretty print schema structure."""
    prefix = "  " * indent

    schema_type = schema.get("type", "unknown")
    schema_id = schema.get("id", "")
    label = schema.get("label", "unknown")

    # Color coding by type
    type_colors = {
        "FOLDER": "\033[94m",  # Blue
        "DATABASE": "\033[92m",  # Green
        "FIELD": "\033[93m",  # Yellow
        "VALUESET": "\033[95m",  # Magenta
    }
    reset = "\033[0m"
    color = type_colors.get(schema_type, "")

    print(f"{prefix}{color}[{schema_type}]{reset} {label}")

    if schema_id and schema_type in ["DATABASE", "FIELD", "VALUESET"]:
        print(f"{prefix}  ID: {schema_id}")

    if "items" in schema:
        for item in schema["items"]:
            print_schema_tree(item, indent + 1)


def search_databases(schema: dict, search_term: str = "") -> list[dict]:
    """Find all databases matching search term."""
    databases = []

    if schema.get("type") == "DATABASE":
        label = schema.get("label", "").lower()
        if not search_term or search_term.lower() in label:
            databases.append(schema)

    if "items" in schema:
        for item in schema["items"]:
            databases.extend(search_databases(item, search_term))

    return databases


def explore_database(client: StatXploreClient, database_id: str):
    """Explore a specific database schema."""
    print(f"\n{'='*80}")
    print(f"Database: {database_id}")
    print(f"{'='*80}\n")

    schema = client.get_schema(database_id)

    print(f"Label: {schema.get('label', 'N/A')}")
    print(f"Description: {schema.get('description', 'N/A')}\n")

    # Find fields and measures
    fields = []
    measures = []

    def extract_items(s):
        if s.get("type") == "FIELD":
            fields.append(s)
        elif s.get("type") == "MEASURE":
            measures.append(s)
        elif s.get("type") == "VALUESET":
            measures.append(s)

        if "items" in s:
            for item in s["items"]:
                extract_items(item)

    extract_items(schema)

    print(f"Fields ({len(fields)}):")
    for field in fields:
        print(f"  • {field.get('label', 'N/A')}")
        print(f"    ID: {field.get('id', 'N/A')}")
        if "description" in field:
            print(f"    Description: {field['description']}")
        print()

    print(f"\nMeasures ({len(measures)}):")
    for measure in measures:
        print(f"  • {measure.get('label', 'N/A')}")
        print(f"    ID: {measure.get('id', 'N/A')}")
        if "description" in measure:
            print(f"    Description: {measure['description']}")
        print()


def test_query(client: StatXploreClient, database_id: str):
    """Test a simple query on a database."""
    print(f"\nTesting query on {database_id}...")

    # Simple query to get latest snapshot
    query = {
        "database": database_id,
        "measures": ["str:count"],  # Generic count measure
        "dimensions": [],  # No dimensions for simplest query
    }

    try:
        result = client.query_table(query)
        print("\nQuery successful!")
        print(f"Result structure: {json.dumps(result, indent=2)[:500]}...")

        # Try to extract some sample data
        if "cubes" in result:
            print(f"\nNumber of cubes: {len(result['cubes'])}")
        if "fields" in result:
            print(f"Number of fields: {len(result['fields'])}")

        return result
    except Exception as e:
        print(f"Query failed: {e}")
        return None


def interactive_mode(client: StatXploreClient):
    """Interactive exploration mode."""
    print("\n" + "="*80)
    print("DWP Stat-Xplore Interactive Explorer")
    print("="*80)

    while True:
        print("\nOptions:")
        print("  1. List all databases")
        print("  2. Search databases")
        print("  3. Explore database schema")
        print("  4. Test query")
        print("  5. Check rate limit")
        print("  0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            print("\nFetching all databases...")
            schema = client.get_schema()
            databases = search_databases(schema)
            print(f"\nFound {len(databases)} databases:\n")
            for i, db in enumerate(databases, 1):
                print(f"{i}. {db.get('label', 'N/A')}")
                print(f"   ID: {db.get('id', 'N/A')}\n")

        elif choice == "2":
            search_term = input("Search term: ").strip()
            schema = client.get_schema()
            databases = search_databases(schema, search_term)
            print(f"\nFound {len(databases)} matching databases:\n")
            for i, db in enumerate(databases, 1):
                print(f"{i}. {db.get('label', 'N/A')}")
                print(f"   ID: {db.get('id', 'N/A')}\n")

        elif choice == "3":
            db_id = input("Database ID: ").strip()
            explore_database(client, db_id)

        elif choice == "4":
            db_id = input("Database ID: ").strip()
            test_query(client, db_id)

        elif choice == "5":
            try:
                rate_limit = client.get_rate_limit()
                print(f"\nRate limit: {json.dumps(rate_limit, indent=2)}")
            except Exception as e:
                print(f"Error: {e}")

        else:
            print("Invalid choice")


def main():
    """Main entry point."""
    api_key = os.getenv("STAT_XPLORE_API_KEY")

    if not api_key:
        print("Error: STAT_XPLORE_API_KEY environment variable not set")
        print("\nGet your API key from: https://stat-xplore.dwp.gov.uk/")
        print("Then set it with: export STAT_XPLORE_API_KEY=your_key_here")
        sys.exit(1)

    client = StatXploreClient(api_key)

    print("Testing API connection...")
    try:
        rate_limit = client.get_rate_limit()
        print(f"✓ Connected successfully")
        print(f"Rate limit: {rate_limit}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)

    # Run interactive mode
    interactive_mode(client)


if __name__ == "__main__":
    main()
