"""FastAPI backend for the targets database explorer."""

from datetime import date
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from policyengine_uk_data.targets import TargetsDB

DB_PATH = Path(__file__).parent.parent.parent / "targets" / "targets.db"
db = TargetsDB(DB_PATH)

app = FastAPI(
    title="PolicyEngine UK targets explorer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AreaResponse(BaseModel):
    code: str
    name: str
    area_type: str
    parent_code: str | None
    children: list["AreaResponse"] = []


class MetricResponse(BaseModel):
    code: str
    name: str
    category: str
    unit: str


class ObservationResponse(BaseModel):
    metric_code: str
    area_code: str
    valid_year: int
    snapshot_date: date
    value: float
    source: str
    source_url: str | None
    is_forecast: bool


class TimeSeriesPoint(BaseModel):
    year: int
    value: float
    is_forecast: bool


@app.get("/api/health")
def health():
    """Health check."""
    stats = db.stats()
    return {"status": "healthy", "observations": stats["observations"]}


@app.get("/api/stats")
def get_stats():
    """Database statistics."""
    from collections import Counter
    from policyengine_uk_data.targets.models import Observation, Metric
    from sqlmodel import Session, select, func

    with Session(db.engine) as session:
        # Get counts
        obs_count = session.exec(select(func.count()).select_from(Observation)).one()
        metric_count = session.exec(select(func.count()).select_from(Metric)).one()
        area_count = len(db.list_areas())

        # Get year range
        years = session.exec(
            select(func.min(Observation.valid_year), func.max(Observation.valid_year))
        ).one()

        # Get categories with counts
        metrics = session.exec(select(Metric)).all()
        category_counts = Counter(m.category for m in metrics)

        # Get sources with counts
        source_counts_result = session.exec(
            select(Observation.source, func.count())
            .group_by(Observation.source)
        ).all()
        source_counts = {s[0]: s[1] for s in source_counts_result}

        return {
            "total_observations": obs_count,
            "total_metrics": metric_count,
            "total_areas": area_count,
            "year_range": [years[0] or 2020, years[1] or 2030],
            "categories": dict(category_counts),
            "sources": source_counts,
        }


@app.get("/api/areas", response_model=list[AreaResponse])
def get_areas(area_type: str | None = Query(None), flat: bool = Query(False)):
    """List areas."""
    areas = db.list_areas(area_type=area_type)

    if flat:
        return [
            AreaResponse(
                code=a.code,
                name=a.name,
                area_type=a.area_type,
                parent_code=a.parent_code,
                children=[],
            )
            for a in areas
        ]

    area_map = {
        a.code: AreaResponse(
            code=a.code,
            name=a.name,
            area_type=a.area_type,
            parent_code=a.parent_code,
            children=[],
        )
        for a in areas
    }

    roots = []
    for a in areas:
        if a.parent_code and a.parent_code in area_map:
            area_map[a.parent_code].children.append(area_map[a.code])
        elif not a.parent_code:
            roots.append(area_map[a.code])

    return roots


@app.get("/api/areas/{code}", response_model=AreaResponse)
def get_area(code: str):
    """Get single area."""
    area = db.get_area(code)
    if not area:
        raise HTTPException(status_code=404, detail=f"Area {code} not found")

    children = db.get_children(code)

    return AreaResponse(
        code=area.code,
        name=area.name,
        area_type=area.area_type,
        parent_code=area.parent_code,
        children=[
            AreaResponse(
                code=c.code,
                name=c.name,
                area_type=c.area_type,
                parent_code=c.parent_code,
                children=[],
            )
            for c in children
        ],
    )


@app.get("/api/metrics", response_model=list[MetricResponse])
def get_metrics(category: str | None = Query(None)):
    """List metrics."""
    metrics = db.list_metrics(category=category)
    return [
        MetricResponse(code=m.code, name=m.name, category=m.category, unit=m.unit)
        for m in metrics
    ]


@app.get("/api/metrics/{code}", response_model=MetricResponse)
def get_metric(code: str):
    """Get single metric."""
    metric = db.get_metric(code)
    if not metric:
        raise HTTPException(status_code=404, detail=f"Metric {code} not found")
    return MetricResponse(
        code=metric.code, name=metric.name, category=metric.category, unit=metric.unit
    )


@app.get("/api/observations", response_model=list[ObservationResponse])
def get_observations(
    metric_code: str | None = Query(None),
    area_code: str | None = Query(None),
    valid_year: int | None = Query(None),
    category: str | None = Query(None),
    is_forecast: bool | None = Query(None),
):
    """Query observations."""
    observations = db.query_observations(
        metric_code=metric_code,
        area_code=area_code,
        valid_year=valid_year,
        category=category,
        is_forecast=is_forecast,
    )
    return [
        ObservationResponse(
            metric_code=o.metric_code,
            area_code=o.area_code,
            valid_year=o.valid_year,
            snapshot_date=o.snapshot_date,
            value=o.value,
            source=o.source,
            source_url=o.source_url,
            is_forecast=o.is_forecast,
        )
        for o in observations
    ]


@app.get("/api/timeseries", response_model=list[TimeSeriesPoint])
@app.get("/api/observations/timeseries", response_model=list[TimeSeriesPoint])
def get_timeseries(
    metric_code: str = Query(...),
    area_code: str = Query("UK"),
    as_of: date | None = Query(None),
    snapshot_date: date | None = Query(None),
):
    """Get time series."""
    # Use snapshot_date if provided, otherwise as_of
    effective_date = snapshot_date or as_of
    trajectory = db.get_trajectory(metric_code, area_code, as_of=effective_date)

    latest = db.get_latest(metric_code, area_code, as_of=effective_date)
    if not latest:
        return []

    return [
        TimeSeriesPoint(
            year=year,
            value=value,
            is_forecast=year > date.today().year,
        )
        for year, value in trajectory.items()
    ]


@app.get("/api/observations/snapshots")
def get_snapshots(metric_code: str | None = Query(None)):
    """Get available snapshot dates."""
    from policyengine_uk_data.targets.models import Observation
    from sqlmodel import Session, select, func

    with Session(db.engine) as session:
        query = select(Observation.snapshot_date).distinct()
        if metric_code:
            query = query.where(Observation.metric_code == metric_code)

        dates = session.exec(query.order_by(Observation.snapshot_date.desc())).all()

        return [
            {"date": str(d), "label": str(d)}
            for d in dates
        ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
