import pytest

torch = pytest.importorskip("torch")
from src.models.graphnet.mesh import build_multi_mesh
from src.models.graphnet.multimesh_model import MultiMeshGraphNet, _combined_edge_index

SMALL_DOMAIN = {"south": 10.0, "north": 18.0, "west": 70.0, "east": 80.0}


def _small_mesh():
    return build_multi_mesh(n_levels=3, domain=SMALL_DOMAIN)


def test_combined_edge_index_spans_all_levels():
    mesh = _small_mesh()
    edge_index, offsets, total = _combined_edge_index(mesh)
    assert total == sum(l.num_nodes for l in mesh.levels)
    assert edge_index.max() < total and edge_index.min() >= 0
    assert edge_index.shape[1] > mesh.levels[0].num_edges


def test_forward_shape_and_probability_range():
    mesh = _small_mesh()
    l0 = mesh.levels[0]
    model = MultiMeshGraphNet(mesh, in_channels=12, hidden=16, processor_layers=3).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 9, l0.n_lat, l0.n_lon, 12))
    assert out.shape == (2, 9, l0.n_lat, l0.n_lon, 1)
    assert torch.isfinite(out).all()
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_gradients_reach_processor_and_head():
    mesh = _small_mesh()
    l0 = mesh.levels[0]
    model = MultiMeshGraphNet(mesh, in_channels=12, hidden=16, processor_layers=3)
    model(torch.randn(1, 9, l0.n_lat, l0.n_lon, 12)).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_matches_output_contract():
    mesh = _small_mesh()
    l0 = mesh.levels[0]
    model = MultiMeshGraphNet(mesh, in_channels=12, hidden=8, processor_layers=2).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 9, l0.n_lat, l0.n_lon, 12))
    assert out.shape[-1] == 1 and out.shape[1] == 9
