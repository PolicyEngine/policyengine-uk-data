"""Dagster asset for the calibration targets database."""

from pathlib import Path

from dagster import asset, AssetExecutionContext

from policyengine_uk_data.targets import TargetsDB
from policyengine_uk_data.targets.seed import seed_all

TARGETS_DB_PATH = Path(__file__).parent.parent / "targets" / "targets.db"


@asset(group_name="targets")
def targets_db(context: AssetExecutionContext) -> dict:
    """Build the calibration targets SQLite database.

    Seeds from:
    - tax_benefit.csv (OBR/DWP aggregates)
    - demographics.csv (ONS population)
    - council_tax_bands_2024.csv (VOA)
    - Hardcoded targets with proper metadata

    Returns dict with database path and stats.
    """
    context.log.info("Seeding calibration targets database...")

    seed_all(TARGETS_DB_PATH)

    db = TargetsDB(TARGETS_DB_PATH)
    stats = db.stats()

    context.log.info(
        f"Database seeded: {stats['observations']} observations, "
        f"{stats['metrics']} metrics, {stats['areas']} areas"
    )

    return {
        "path": str(TARGETS_DB_PATH),
        "observations": stats["observations"],
        "metrics": stats["metrics"],
        "areas": stats["areas"],
        "categories": stats.get("categories", []),
    }
