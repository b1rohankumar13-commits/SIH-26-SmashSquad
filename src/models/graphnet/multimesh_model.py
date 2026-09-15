"""Multi-resolution GraphSAGE processor + TCN head; same I/O contract as GridGraphNet."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .mesh import MultiMesh


def _combined_edge_index(mesh: MultiMesh) -> tuple[np.ndarray, list[int], int]:
    sizes = [level.num_nodes for level in mesh.levels]
    offsets = np.cumsum([0] + sizes)[:-1].tolist()
    total = int(sum(sizes))

    blocks = [level.edge_index + offsets[k] for k, level in enumerate(mesh.levels)]
    for k in range(mesh.n_levels - 1):
        up = mesh.inter_edge_index(k, direction="up").copy()
        up[0] += offsets[k]; up[1] += offsets[k + 1]
        down = mesh.inter_edge_index(k, direction="down").copy()
        down[0] += offsets[k + 1]; down[1] += offsets[k]
        blocks.extend([up, down])
    return np.concatenate(blocks, axis=1).astype(np.int64), offsets, total


class MultiMeshGraphNet(nn.Module):
    def __init__(
        self,
        mesh: MultiMesh,
        *,
        in_channels: int = 12,
        hidden: int = 64,
        processor_layers: int = 4,
        tcn_kernel: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_kernel % 2 == 0:
            raise ValueError("tcn_kernel must be odd to preserve the lead length")
        level0 = mesh.levels[0]
        self.n_lat, self.n_lon = level0.n_lat, level0.n_lon
        self.n_level0 = level0.num_nodes

        edge_index, _, total = _combined_edge_index(mesh)
        self.num_nodes_total = total
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))

        self.input_proj = nn.Linear(in_channels, hidden)
        self.processor = nn.ModuleList(SAGEConv(hidden, hidden) for _ in range(processor_layers))
        self.spatial_dropout = nn.Dropout(dropout)

        self.tcn = nn.ModuleList(
            nn.Conv1d(hidden, hidden, tcn_kernel, padding=d * (tcn_kernel - 1) // 2, dilation=d)
            for d in tcn_dilations
        )
        self.temporal_dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self._edge_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

    def _batched_edge_index(self, snapshots: int, device: torch.device) -> torch.Tensor:
        key = (snapshots, device)
        cached = self._edge_cache.get(key)
        if cached is None:
            edges = self.edge_index.size(1)
            repeated = self.edge_index.repeat(1, snapshots)
            offsets = (torch.arange(snapshots, device=self.edge_index.device)
                       * self.num_nodes_total).repeat_interleave(edges)
            cached = (repeated + offsets).to(device)
            self._edge_cache[key] = cached
        return cached

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5 or x.shape[2] != self.n_lat or x.shape[3] != self.n_lon:
            raise ValueError(f"Expected [B, L, {self.n_lat}, {self.n_lon}, C], got {tuple(x.shape)}")
        batch, leads, n_lat, n_lon, channels = x.shape
        snapshots, n0, ntot = batch * leads, self.n_level0, self.num_nodes_total

        level0_feats = self.input_proj(x.reshape(snapshots, n0, channels))
        feats = level0_feats.new_zeros(snapshots, ntot, level0_feats.size(-1))
        feats[:, :n0, :] = level0_feats

        nodewise = feats.reshape(snapshots * ntot, -1)
        edge_index = self._batched_edge_index(snapshots, x.device)
        for conv in self.processor:
            nodewise = self.spatial_dropout(torch.relu(conv(nodewise, edge_index)))
        hidden = nodewise.size(-1)

        level0 = nodewise.reshape(snapshots, ntot, hidden)[:, :n0, :]

        temporal = (level0.reshape(batch, leads, n0, hidden)
                    .permute(0, 2, 3, 1).reshape(batch * n0, hidden, leads))
        for conv in self.tcn:
            temporal = temporal + self.temporal_dropout(torch.relu(conv(temporal)))
        temporal = temporal.reshape(batch, n0, hidden, leads).permute(0, 3, 1, 2)

        return torch.sigmoid(self.head(temporal)).reshape(batch, leads, n_lat, n_lon, 1)
