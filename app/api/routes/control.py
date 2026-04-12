from fastapi import APIRouter

from app.api.schemas import ControlAction

router = APIRouter()


@router.post("/action", response_model=ControlAction)
def control_action(action: str) -> ControlAction:
    return ControlAction(status="placeholder", message="Control action endpoint")
