"""Calibration targets database with bitemporal tracking.

Key concepts:
- Area: Geographic hierarchy (UK → country → region → constituency)
- Metric: What's being measured (income_tax, population, etc.)
- Observation: A value with two time dimensions:
  - valid_year: The year the statistic applies to
  - snapshot_date: When this information was published

Example usage:
    from policyengine_uk_data.targets import TargetsDB

    db = TargetsDB()

    # Get latest value for income tax 2026
    value = db.get_value("income_tax", "UK", 2026)

    # Get trajectory as of March 2024
    trajectory = db.get_trajectory("income_tax", "UK", as_of=date(2024, 3, 15))

    # See how forecasts changed over time
    revisions = db.get_revision_history("income_tax", "UK", 2026)
"""

from policyengine_uk_data.targets.database import TargetsDB
from policyengine_uk_data.targets.models import Area, Metric, Observation

__all__ = ["TargetsDB", "Area", "Metric", "Observation"]
