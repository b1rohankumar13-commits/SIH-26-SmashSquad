import numpy as np

from src.models.graphnet.mesh import build_multi_mesh


def test_levels_coarsen_and_shrink():
    mesh = build_multi_mesh(n_levels=4)
    assert mesh.steps == [0.5, 1.0, 2.0, 4.0]
    counts = [lvl.num_nodes for lvl in mesh.levels]
    assert counts[0] == 66 * 70
    assert counts == sorted(counts, reverse=True)
    assert counts[1] == 33 * 35


def test_every_fine_node_has_a_valid_parent():
    mesh = build_multi_mesh(n_levels=3)
    for k, parent in enumerate(mesh.parents):
        fine, coarse = mesh.levels[k], mesh.levels[k + 1]
        assert parent.shape == (fine.num_nodes,)
        assert parent.min() >= 0 and parent.max() < coarse.num_nodes


def test_parent_is_geographically_nearest():
    mesh = build_multi_mesh(n_levels=2)
    fine, coarse = mesh.levels[0], mesh.levels[1]
    for i in [0, 100, 2000, fine.num_nodes - 1]:
        d = (coarse.node_lat - fine.node_lat[i]) ** 2 + (coarse.node_lon - fine.node_lon[i]) ** 2
        assert mesh.parents[0][i] == int(np.argmin(d))


def test_inter_edge_index_directions():
    mesh = build_multi_mesh(n_levels=2)
    up = mesh.inter_edge_index(0, direction="up")
    down = mesh.inter_edge_index(0, direction="down")
    n_fine = mesh.levels[0].num_nodes
    assert up.shape == (2, n_fine) and down.shape == (2, n_fine)
    assert np.array_equal(up[0], down[1]) and np.array_equal(up[1], down[0])
    assert up[0].max() < n_fine


def test_single_level_has_no_parents():
    mesh = build_multi_mesh(n_levels=1)
    assert mesh.n_levels == 1 and mesh.parents == []
