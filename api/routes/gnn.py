"""Routes for externally generated GNN predictions."""

from fastapi import APIRouter, HTTPException, status

from api.schemas import GNNPublishRequest, GNNPublishResponse, GNNStatusResponse
from api.services.gnn_store import latest_gnn_file, publish_gnn_records


router = APIRouter(prefix="/gnn", tags=["gnn"])


@router.get("/status", response_model=GNNStatusResponse)
def gnn_status() -> GNNStatusResponse:
    latest = latest_gnn_file()
    return GNNStatusResponse(
        status="available" if latest else "not_generated",
        latest_file=latest.name if latest else None,
    )


@router.post(
    "/predictions",
    response_model=GNNPublishResponse,
    status_code=status.HTTP_201_CREATED,
)
def publish_gnn_predictions(payload: GNNPublishRequest) -> GNNPublishResponse:
    try:
        result = publish_gnn_records(
            [record.model_dump(mode="python") for record in payload.records],
            run_id=payload.run_id,
            model_id=payload.model_id,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    dashboard_ready = result.dashboard_file is not None
    return GNNPublishResponse(
        status="published" if dashboard_ready else "stored",
        run_id=payload.run_id,
        model_id=payload.model_id,
        record_count=result.record_count,
        dashboard_ready=dashboard_ready,
        stored_file=result.branch_file.name,
        dashboard_file=result.dashboard_file.name if result.dashboard_file else None,
        message=(
            "Final probabilities published for the dashboard."
            if dashboard_ready
            else "GNN branch probabilities stored; calibrated/fused probabilities are still required."
        ),
    )
