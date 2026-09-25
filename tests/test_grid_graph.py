import numpy as np

from src.models.graphnet.grid_graph import build_grid_graph, canonical_centres


def _edge_set(graph):
    return {(int(a), int(b)) for a, b in zip(graph.edge_index[0], graph.edge_index[1])}


def test_canonical_grid_is_66_by_70():
    lats, lons = canonical_centres()
    assert lats.size == 66 and lons.size == 70
    assert lats[0] == 5.25 and lats[-1] == 37.75
    assert lons[0] == 65.25 and lons[-1] == 99.75


def test_node_ordering_is_row_major():
    g = build_grid_graph(np.array([10.0, 11.0, 12.0]), np.array([70.0, 71.0, 72.0, 73.0]), connectivity=4)
    assert g.num_nodes == 12
    assert g.node_row[5] == 1 and g.node_col[5] == 1
    assert g.node_lat[5] == 11.0 and g.node_lon[5] == 71.0


def test_four_connectivity_neighbour_counts():
    g = build_grid_graph(np.arange(3.0, 6.0), np.arange(70.0, 74.0), connectivity=4)
    assert sum(1 for a, _ in g.edge_index.T if a == 0) == 2
    assert sum(1 for a, _ in g.edge_index.T if a == 5) == 4


def test_eight_connectivity_interior_has_eight():
    g = build_grid_graph(np.arange(3.0, 6.0), np.arange(70.0, 74.0), connectivity=8)
    assert sum(1 for a, _ in g.edge_index.T if a == 5) == 8


def test_edges_are_symmetric_and_distances_positive():
    g = build_grid_graph(np.arange(5.0, 9.0), np.arange(70.0, 75.0), connectivity=8)
    edges = _edge_set(g)
    assert all((b, a) in edges for a, b in edges)
    assert g.edge_distance_km.shape[0] == g.num_edges
    assert np.all(g.edge_distance_km > 0)
    assert not any(a == b for a, b in edges)


def test_full_india_grid_shape():
    g = build_grid_graph(connectivity=8)
    assert g.num_nodes == 66 * 70 == 4620
    assert 8 * g.num_nodes * 0.8 < g.num_edges < 8 * g.num_nodes
