"""CLI for viewing and debugging calibration targets.

Usage:
    python -m policyengine_uk_data.targets.cli trajectory income_tax --as-of 2024-03-15
    python -m policyengine_uk_data.targets.cli history income_tax 2026
    python -m policyengine_uk_data.targets.cli areas
    python -m policyengine_uk_data.targets.cli metrics --category obr
    python -m policyengine_uk_data.targets.cli stats
"""

import argparse
from datetime import date

from rich.console import Console
from rich.table import Table

from policyengine_uk_data.targets import TargetsDB

console = Console()


def cmd_trajectory(args):
    """Show forecast trajectory for a metric."""
    db = TargetsDB()
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()

    trajectory = db.get_trajectory(args.metric, args.area, as_of=as_of)

    if not trajectory:
        console.print(f"[red]No data for {args.metric} in {args.area}[/red]")
        return

    metric = db.get_metric(args.metric)
    metric_name = metric.name if metric else args.metric

    table = Table(title=f"{metric_name} ({args.area}) as of {as_of}")
    table.add_column("Year", style="green")
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Type", style="cyan")

    for year, value in trajectory.items():
        obs = db.get_latest(args.metric, args.area, year, as_of=as_of)
        obs_type = "forecast" if obs and obs.is_forecast else "actual"
        display = obs.display_value() if obs else f"{value:,.0f}"
        table.add_row(str(year), display, obs_type)

    console.print(table)


def cmd_history(args):
    """Show how forecasts changed over time for a metric/year."""
    db = TargetsDB()

    revisions = db.get_revision_history(args.metric, args.area, args.year)

    if not revisions:
        console.print(f"[red]No data for {args.metric} {args.year}[/red]")
        return

    metric = db.get_metric(args.metric)
    metric_name = metric.name if metric else args.metric

    table = Table(title=f"Revision history: {metric_name} {args.year}")
    table.add_column("Snapshot", style="green")
    table.add_column("Value", style="yellow", justify="right")
    table.add_column("Source")
    table.add_column("Type", style="cyan")

    for obs in revisions:
        obs_type = "forecast" if obs.is_forecast else "actual"
        table.add_row(
            str(obs.snapshot_date),
            obs.display_value(),
            obs.source,
            obs_type,
        )

    console.print(table)


def cmd_areas(args):
    """List geographic areas."""
    db = TargetsDB()
    areas = db.list_areas(area_type=args.type)

    table = Table(title=f"Areas ({len(areas)})")
    table.add_column("Code", style="cyan")
    table.add_column("Name")
    table.add_column("Type", style="green")
    table.add_column("Parent")

    for area in sorted(areas, key=lambda a: (a.area_type, a.code)):
        table.add_row(area.code, area.name, area.area_type, area.parent_code or "-")

    console.print(table)


def cmd_metrics(args):
    """List metric definitions."""
    db = TargetsDB()
    metrics = db.list_metrics(category=args.category)

    table = Table(title=f"Metrics ({len(metrics)})")
    table.add_column("Code", style="cyan")
    table.add_column("Name")
    table.add_column("Category", style="green")
    table.add_column("Unit")

    for m in sorted(metrics, key=lambda x: (x.category, x.code)):
        table.add_row(m.code, m.name, m.category, m.unit)

    console.print(table)


def cmd_stats(args):
    """Show database statistics."""
    db = TargetsDB()
    stats = db.stats()

    console.print("\n[bold]Targets database statistics[/bold]")
    console.print(f"  Observations: {stats.get('observations', 0)}")
    console.print(f"  Metrics: {stats.get('metrics', 0)}")
    console.print(f"  Areas: {stats.get('areas', 0)}")

    if stats.get("valid_years"):
        console.print(f"  Valid years: {min(stats['valid_years'])}-{max(stats['valid_years'])}")

    if stats.get("snapshot_dates"):
        console.print(f"  Snapshots: {min(stats['snapshot_dates'])} to {max(stats['snapshot_dates'])}")

    if stats.get("categories"):
        console.print(f"\n[bold]Categories:[/bold]")
        for cat in stats["categories"]:
            metrics = db.list_metrics(category=cat)
            console.print(f"  {cat}: {len(metrics)} metrics")

    if stats.get("area_types"):
        console.print(f"\n[bold]Area types:[/bold]")
        for at in stats["area_types"]:
            areas = db.list_areas(area_type=at)
            console.print(f"  {at}: {len(areas)} areas")


def cmd_compare(args):
    """Compare trajectories at two different snapshot dates."""
    db = TargetsDB()
    as_of_1 = date.fromisoformat(args.snapshot1)
    as_of_2 = date.fromisoformat(args.snapshot2)

    traj1 = db.get_trajectory(args.metric, args.area, as_of=as_of_1)
    traj2 = db.get_trajectory(args.metric, args.area, as_of=as_of_2)

    if not traj1 and not traj2:
        console.print(f"[red]No data for {args.metric}[/red]")
        return

    metric = db.get_metric(args.metric)
    metric_name = metric.name if metric else args.metric

    table = Table(title=f"{metric_name}: {as_of_1} vs {as_of_2}")
    table.add_column("Year", style="green")
    table.add_column(f"As of {as_of_1}", style="yellow", justify="right")
    table.add_column(f"As of {as_of_2}", style="cyan", justify="right")
    table.add_column("Change", justify="right")

    all_years = sorted(set(traj1.keys()) | set(traj2.keys()))
    for year in all_years:
        v1 = traj1.get(year)
        v2 = traj2.get(year)

        s1 = f"{v1/1e9:.1f}bn" if v1 else "-"
        s2 = f"{v2/1e9:.1f}bn" if v2 else "-"

        if v1 and v2:
            change = (v2 - v1) / v1 * 100
            change_str = f"{change:+.1f}%" if abs(change) > 0.1 else "~"
        else:
            change_str = "-"

        table.add_row(str(year), s1, s2, change_str)

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Calibration targets database CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # trajectory command
    traj_parser = subparsers.add_parser("trajectory", help="Show forecast trajectory")
    traj_parser.add_argument("metric", help="Metric code")
    traj_parser.add_argument("--area", default="UK", help="Area code")
    traj_parser.add_argument("--as-of", help="Snapshot date (YYYY-MM-DD)")
    traj_parser.set_defaults(func=cmd_trajectory)

    # history command
    hist_parser = subparsers.add_parser("history", help="Show revision history")
    hist_parser.add_argument("metric", help="Metric code")
    hist_parser.add_argument("year", type=int, help="Target year")
    hist_parser.add_argument("--area", default="UK", help="Area code")
    hist_parser.set_defaults(func=cmd_history)

    # areas command
    areas_parser = subparsers.add_parser("areas", help="List areas")
    areas_parser.add_argument("--type", help="Filter by area type")
    areas_parser.set_defaults(func=cmd_areas)

    # metrics command
    metrics_parser = subparsers.add_parser("metrics", help="List metrics")
    metrics_parser.add_argument("--category", help="Filter by category")
    metrics_parser.set_defaults(func=cmd_metrics)

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Show database stats")
    stats_parser.set_defaults(func=cmd_stats)

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare trajectories at two snapshots")
    compare_parser.add_argument("metric", help="Metric code")
    compare_parser.add_argument("snapshot1", help="First snapshot date (YYYY-MM-DD)")
    compare_parser.add_argument("snapshot2", help="Second snapshot date (YYYY-MM-DD)")
    compare_parser.add_argument("--area", default="UK", help="Area code")
    compare_parser.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
