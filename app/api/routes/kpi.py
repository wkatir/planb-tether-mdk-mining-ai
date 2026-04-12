import duckdb

from fastapi import APIRouter, Depends

from app.api.dependencies import get_db
from app.api.schemas import FleetOverview, DeviceRanking

router = APIRouter()


@router.get("/fleet-overview", response_model=FleetOverview)
def get_fleet_overview(
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> FleetOverview:
    sql = """
        SELECT
            COUNT(DISTINCT device_id) as total_devices,
            AVG(temperature) as avg_temperature,
            AVG(hash_rate) as avg_hash_rate,
            AVG(efficiency) as avg_efficiency,
            SUM(power_draw) as total_power_draw,
            SUM(energy_cost) as total_energy_cost,
            AVG(efficiency) * 1000 / AVG(temperature) as fleet_te_score
        FROM telemetry
    """
    df = db.execute(sql).fetchdf()
    row = df.iloc[0]
    return FleetOverview(
        total_devices=int(row["total_devices"]),
        avg_temperature=float(row["avg_temperature"]),
        avg_hash_rate=float(row["avg_hash_rate"]),
        avg_efficiency=float(row["avg_efficiency"]),
        total_power_draw=float(row["total_power_draw"]),
        total_energy_cost=float(row["total_energy_cost"]),
        fleet_te_score=float(row["fleet_te_score"]),
    )


@router.get("/device-ranking")
def get_device_ranking(
    limit: int = 10,
    db: duckdb.DuckDBPyConnection = Depends(get_db),
) -> list[DeviceRanking]:
    sql = f"""
        SELECT
            device_id,
            AVG(efficiency) * 1000 / AVG(temperature) as te_score
        FROM telemetry
        GROUP BY device_id
        ORDER BY te_score DESC
        LIMIT {limit}
    """
    df = db.execute(sql).fetchdf()
    rankings = []
    for i, row in enumerate(df.itertuples(), 1):
        rankings.append(
            DeviceRanking(
                device_id=row.device_id,
                te_score=float(row.te_score),
                rank=i,
            )
        )
    return rankings
