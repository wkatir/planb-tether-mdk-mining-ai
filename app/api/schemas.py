from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TelemetryRecord(BaseModel):
    timestamp: datetime
    device_id: str
    temperature: float = Field(
        ge=0, le=120, description="Device temperature in Celsius"
    )
    power_draw: float = Field(ge=0, le=10000, description="Power consumption in Watts")
    hash_rate: float = Field(ge=0, le=1000, description="Hash rate in TH/s")
    efficiency: float = Field(ge=0, le=100, description="Efficiency in J/TH")
    uptime_hours: float = Field(ge=0, description="Uptime in hours")
    ambient_temp: float = Field(
        ge=-20, le=60, description="Ambient temperature in Celsius"
    )
    energy_cost: float = Field(ge=0, description="Energy cost per kWh")


class TelemetryResponse(BaseModel):
    records: list[TelemetryRecord]
    count: int


class KPISummary(BaseModel):
    device_id: str
    avg_temperature: float = Field(description="Average temperature")
    avg_hash_rate: float = Field(description="Average hash rate")
    total_energy_cost: float = Field(description="Total energy cost")
    uptime_hours: float = Field(description="Total uptime hours")
    te_score: float = Field(description="Thermal Efficiency score")


class FleetOverview(BaseModel):
    total_devices: int
    avg_temperature: float
    avg_hash_rate: float
    avg_efficiency: float
    total_power_draw: float
    total_energy_cost: float
    fleet_te_score: float


class DeviceRanking(BaseModel):
    device_id: str
    te_score: float
    rank: int


class HealthStatus(BaseModel):
    status: str
    message: Optional[str] = None


class ControlAction(BaseModel):
    status: str
    message: Optional[str] = None
