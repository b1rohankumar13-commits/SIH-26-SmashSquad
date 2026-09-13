import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.models.graphnet.grid_graph import build_grid_graph
from src.models.graphnet.model import GridGraphNet


def _small_graph():
    return build_grid_graph(np.arange(10.0, 18.0), np.arange(70.0, 80.0), connectivity=8)


def test_forward_shape_and_probability_range():
    graph = _small_graph()
    model = GridGraphNet(graph, in_channels=12, hidden=16).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 9, graph.n_lat, graph.n_lon, 12))
    assert out.shape == (2, 9, graph.n_lat, graph.n_lon, 1)
    assert torch.isfinite(out).all()
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_gradients_flow_to_all_parameters():
    graph = _small_graph()
    model = GridGraphNet(graph, in_channels=12, hidden=16)
    model(torch.randn(1, 9, graph.n_lat, graph.n_lon, 12)).mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_rejects_wrong_grid_shape():
    model = GridGraphNet(_small_graph(), in_channels=12, hidden=8)
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(1, 9, 5, 5, 12))


def test_single_lead_sequence_works():
    graph = _small_graph()
    model = GridGraphNet(graph, in_channels=12, hidden=8).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, graph.n_lat, graph.n_lon, 12))
    assert out.shape == (1, 1, graph.n_lat, graph.n_lon, 1)
