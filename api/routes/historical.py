from fastapi import APIRouter

router = APIRouter(prefix="/historical", tags=["historical"])

@router.get("/{init_date}")
def historical_forecast(init_date: str):
    return {"init_date": init_date, "status": "not_generated"}
