"""Multi-resolution mesh: grid levels at 0.5, 1, 2, 4 deg with fine->coarse parent maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .grid_graph import DOMAIN, GridGraph, build_grid_graph, canonical_centres


@dataclass(frozen=True)
class MultiMesh:
    levels: list[GridGraph]          # finest first
    parents: list[np.ndarray]        # parents[k][i] = coarse node in level k+1 for fine node i
    steps: list[float]

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def inter_edge_index(self, fine_level: int, *, direction: str = "up") -> np.ndarray:
        if not 0 <= fine_level < self.n_levels - 1:
            raise ValueError(f"fine_level must be in [0, {self.n_levels - 2}]")
        parent = self.parents[fine_level]
        fine_nodes = np.arange(parent.size)
        if direction == "up":
            return np.stack([fine_nodes, parent]).astype(np.int64)
        if direction == "down":
            return np.stack([parent, fine_nodes]).astype(np.int64)
        raise ValueError("direction must be 'up' or 'down'")


def build_multi_mesh(
    *, n_levels: int = 4, base_step: float = 0.5, connectivity: int = 8,
    domain: dict | None = None,
) -> MultiMesh:
    if n_levels < 1:
        raise ValueError("n_levels must be >= 1")
    domain = domain or DOMAIN

    levels, steps = [], []
    for k in range(n_levels):
        step = base_step * (2 ** k)
        lats, lons = canonical_centres(step, domain)
        if lats.size < 2 or lons.size < 2:
            break
        levels.append(build_grid_graph(lats, lons, connectivity=connectivity))
        steps.append(step)

    parents = []
    for k in range(len(levels) - 1):
        fine, coarse = levels[k], levels[k + 1]
        coarse_points = np.stack([coarse.node_lat, coarse.node_lon], axis=1)
        fine_points = np.stack([fine.node_lat, fine.node_lon], axis=1)
        _, parent = cKDTree(coarse_points).query(fine_points, k=1)
        parents.append(parent.astype(np.int64))

    return MultiMesh(levels=levels, parents=parents, steps=steps)
