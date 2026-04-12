from fastapi import APIRouter

from app.api.schemas import HealthStatus

router = APIRouter()


@router.get("/", response_model=HealthStatus)
def health_check() -> HealthStatus:
    return HealthStatus(status="placeholder", message="Health check endpoint")
