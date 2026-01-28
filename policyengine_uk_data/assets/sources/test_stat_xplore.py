"""Quick test script for DWP Stat-Xplore API connectivity.

Usage:
    export STAT_XPLORE_API_KEY=your_key_here
    python test_stat_xplore.py
"""

import os
import sys
import json

from dwp_stat_xplore import StatXploreClient


def test_connection(api_key: str):
    """Test basic API connectivity."""
    print("Testing DWP Stat-Xplore API connection...")

    client = StatXploreClient(api_key)

    # Test 1: Get rate limit
    print("\n1. Testing rate limit endpoint...")
    try:
        rate_limit = client.get_rate_limit()
        print(f"   ✓ Rate limit: {json.dumps(rate_limit, indent=2)}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 2: Get root schema
    print("\n2. Testing schema endpoint...")
    try:
        schema = client.get_schema()
        print(f"   ✓ Root schema retrieved")
        print(f"   Label: {schema.get('label', 'N/A')}")
        print(f"   Type: {schema.get('type', 'N/A')}")
        if "items" in schema:
            print(f"   Items: {len(schema['items'])} folders/databases")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    # Test 3: Search for benefit-related databases
    print("\n3. Searching for key databases...")
    try:
        root = client.get_schema()
        databases = []

        def find_databases(item):
            if item.get("type") == "DATABASE":
                databases.append(item)
            if "items" in item:
                for child in item["items"]:
                    find_databases(child)

        find_databases(root)

        print(f"   ✓ Found {len(databases)} databases total")

        # Look for our key databases
        targets = ["pip", "universal credit", "benefit cap", "uc"]
        found = []

        for db in databases:
            label = db.get("label", "").lower()
            for target in targets:
                if target in label:
                    found.append(db)
                    break

        print(f"\n   Key databases found:")
        for db in found[:10]:  # Show first 10
            print(f"   • {db.get('label', 'N/A')}")
            print(f"     ID: {db.get('id', 'N/A')}")

    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    print("\n✓ All tests passed!")
    print("\nNext steps:")
    print("1. Run explore_stat_xplore.py to interactively explore databases")
    print("2. Use the explorer to find exact database and field IDs")
    print("3. Update dwp_stat_xplore.py with correct IDs")
    print("4. Implement query parsing logic")

    return True


def main():
    """Main entry point."""
    api_key = os.getenv("STAT_XPLORE_API_KEY")

    if not api_key:
        print("Error: STAT_XPLORE_API_KEY environment variable not set")
        print("\nGet your API key from: https://stat-xplore.dwp.gov.uk/")
        print("Then set it with: export STAT_XPLORE_API_KEY=your_key_here")
        sys.exit(1)

    success = test_connection(api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
