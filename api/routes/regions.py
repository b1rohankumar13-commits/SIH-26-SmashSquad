"""Regional catalogue API routes."""

from fastapi import APIRouter

from api.schemas import CatalogueResponse
from api.services.prediction_store import catalogue_values, load_current_snapshot


router = APIRouter(prefix="/regions", tags=["regions"])


@router.get("", response_model=CatalogueResponse)
def list_regions() -> CatalogueResponse:
    snapshot = load_current_snapshot()
    if snapshot.frame is None:
        return CatalogueResponse(status=snapshot.status, message=snapshot.message)
    return CatalogueResponse(
        status="available",
        items=catalogue_values(snapshot.frame, "region_id"),
    )
