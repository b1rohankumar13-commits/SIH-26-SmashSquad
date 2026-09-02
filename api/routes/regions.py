from fastapi import APIRouter

router = APIRouter(prefix="/regions", tags=["regions"])

@router.get("")
def list_regions():
    return {"regions": [], "note": "Load approved India boundary definitions."}
