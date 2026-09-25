"""Tests for externally supplied GNN predictions."""

from api.services.gnn_store import publish_gnn_records


def _records(*, final: bool):
    rows = [
        {
            "grid_id": "20.0_85.0",
            "lead_day": lead,
            "latitude": 20.0,
            "longitude": 85.0,
            "gnn_probability": 0.2 + lead / 100,
        }
        for lead in (1, 2)
    ]
    if final:
        for row in rows:
            row["overall_bust_probability"] = row["gnn_probability"]
    return rows


def test_branch_predictions_do_not_publish_unfused_dashboard_values(tmp_path):
    result = publish_gnn_records(
        _records(final=False),
        run_id="20260914_00",
        model_id="gnn-v1",
        branch_directory=tmp_path / "gnn",
        dashboard_directory=tmp_path / "dashboard",
    )
    assert result.branch_file.exists()
    assert result.dashboard_file is None


def test_final_probabilities_are_published_for_dashboard(tmp_path):
    result = publish_gnn_records(
        _records(final=True),
        run_id="20260914_00",
        model_id="gnn-v1",
        branch_directory=tmp_path / "gnn",
        dashboard_directory=tmp_path / "dashboard",
    )
    assert result.branch_file.exists()
    assert result.dashboard_file.exists()
