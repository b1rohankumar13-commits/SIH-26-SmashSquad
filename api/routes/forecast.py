from fastapi import APIRouter

router = APIRouter(prefix="/forecast", tags=["forecast"])

@router.get("/current")
def current_forecast():
    return {"status": "not_generated", "horizon_days": 10}
