from fastapi import APIRouter, Depends
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from app.api.dependencies import get_db
from app.api.schemas import TelemetryResponse

router = APIRouter()


@router.get("/latest", response_model=TelemetryResponse)
def get_latest_telemetry(
    limit: int = 100,
    db: Connection = Depends(get_db),
) -> TelemetryResponse:
    sql = text("""
        SELECT * FROM telemetry
        ORDER BY timestamp DESC
        LIMIT :limit
    """)
    df = pd.read_sql(sql, db, params={"limit": limit})
    records = df.to_dict(orient="records")
    return TelemetryResponse(records=records, count=len(records))


@router.get("/devices")
def get_devices(db: Connection = Depends(get_db)) -> list[dict]:
    sql = text("SELECT DISTINCT device_id FROM telemetry ORDER BY device_id")
    df = pd.read_sql(sql, db)
    return df.to_dict(orient="records")


@router.get("/stats/{device_id}")
def get_device_stats(device_id: str, db: Connection = Depends(get_db)) -> dict:
    sql = text("""
        SELECT
            device_id,
            AVG(chip_temperature_c) as avg_temperature,
            AVG(asic_hashrate_th) as avg_hash_rate,
            SUM(energy_price_kwh) as total_energy_cost,
            COUNT(*) as record_count
        FROM telemetry
        WHERE device_id = :device_id
        GROUP BY device_id
    """)
    df = pd.read_sql(sql, db, params={"device_id": device_id})
    if df.empty:
        return {"error": "Device not found"}
    return df.to_dict(orient="records")[0]
