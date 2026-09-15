"""Grid-graph over the 0.5-degree India grid (nodes row-major to match grid_id)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DOMAIN = {"south": 5.0, "north": 38.0, "west": 65.0, "east": 100.0}

_NEIGHBOUR_OFFSETS = {
    4: [(-1, 0), (1, 0), (0, -1), (0, 1)],
    8: [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
}


def canonical_centres(step: float = 0.5, domain: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    domain = domain or DOMAIN
    lats = np.arange(domain["south"] + step / 2, domain["north"], step)
    lons = np.arange(domain["west"] + step / 2, domain["east"], step)
    return lats, lons


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    radius = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(inner))


@dataclass(frozen=True)
class GridGraph:
    n_lat: int
    n_lon: int
    edge_index: np.ndarray
    node_lat: np.ndarray
    node_lon: np.ndarray
    node_row: np.ndarray
    node_col: np.ndarray
    edge_distance_km: np.ndarray

    @property
    def num_nodes(self) -> int:
        return self.n_lat * self.n_lon

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_torch(self):
        import torch

        edge_index = torch.as_tensor(self.edge_index, dtype=torch.long)
        pos = torch.as_tensor(np.stack([self.node_lat, self.node_lon], axis=1), dtype=torch.float32)
        edge_attr = torch.as_tensor(self.edge_distance_km[:, None], dtype=torch.float32)
        return edge_index, pos, edge_attr


def build_grid_graph(
    lats: np.ndarray | None = None,
    lons: np.ndarray | None = None,
    *,
    connectivity: int = 8,
) -> GridGraph:
    if connectivity not in _NEIGHBOUR_OFFSETS:
        raise ValueError("connectivity must be 4 or 8")
    if lats is None or lons is None:
        lats, lons = canonical_centres()
    lats = np.asarray(lats, dtype=np.float64)
    lons = np.asarray(lons, dtype=np.float64)
    n_lat, n_lon = lats.size, lons.size
    if n_lat == 0 or n_lon == 0:
        raise ValueError("lats and lons must be non-empty")

    node_row, node_col = np.divmod(np.arange(n_lat * n_lon), n_lon)
    node_lat, node_lon = lats[node_row], lons[node_col]

    src, dst = [], []
    for row in range(n_lat):
        for col in range(n_lon):
            here = row * n_lon + col
            for d_row, d_col in _NEIGHBOUR_OFFSETS[connectivity]:
                nr, nc = row + d_row, col + d_col
                if 0 <= nr < n_lat and 0 <= nc < n_lon:
                    src.append(here)
                    dst.append(nr * n_lon + nc)

    edge_index = np.array([src, dst], dtype=np.int64)
    edge_distance_km = _haversine_km(
        node_lat[edge_index[0]], node_lon[edge_index[0]],
        node_lat[edge_index[1]], node_lon[edge_index[1]],
    )
    return GridGraph(
        n_lat=n_lat, n_lon=n_lon, edge_index=edge_index,
        node_lat=node_lat, node_lon=node_lon,
        node_row=node_row, node_col=node_col, edge_distance_km=edge_distance_km,
    )
