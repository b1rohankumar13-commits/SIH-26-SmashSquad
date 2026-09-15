"""GraphSAGE spatial encoder + TCN temporal head over the flat grid-graph."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv

from .grid_graph import GridGraph


class GridGraphNet(nn.Module):
    def __init__(
        self,
        graph: GridGraph,
        *,
        in_channels: int = 12,
        hidden: int = 64,
        sage_layers: int = 2,
        tcn_kernel: int = 3,
        tcn_dilations: tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        if tcn_kernel % 2 == 0:
            raise ValueError("tcn_kernel must be odd to preserve the lead length")
        self.n_lat, self.n_lon, self.num_nodes = graph.n_lat, graph.n_lon, graph.num_nodes

        edge_index, _, _ = graph.to_torch()
        self.register_buffer("edge_index", edge_index)

        self.sage = nn.ModuleList()
        dim = in_channels
        for _ in range(sage_layers):
            self.sage.append(SAGEConv(dim, hidden))
            dim = hidden
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
            offsets = (torch.arange(snapshots, device=self.edge_index.device) * self.num_nodes
                       ).repeat_interleave(edges)
            cached = (repeated + offsets).to(device)
            self._edge_cache[key] = cached
        return cached

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Encoder output (SAGE + TCN) as per-cell hidden features [B, L, lat, lon, hidden]."""
        if x.dim() != 5 or x.shape[2] != self.n_lat or x.shape[3] != self.n_lon:
            raise ValueError(f"Expected [B, L, {self.n_lat}, {self.n_lon}, C], got {tuple(x.shape)}")
        batch, leads, n_lat, n_lon, channels = x.shape
        snapshots, nodes = batch * leads, self.num_nodes

        nodewise = x.reshape(snapshots * nodes, channels)
        edge_index = self._batched_edge_index(snapshots, x.device)
        for conv in self.sage:
            nodewise = self.spatial_dropout(torch.relu(conv(nodewise, edge_index)))
        hidden = nodewise.size(-1)

        temporal = (nodewise.reshape(batch, leads, nodes, hidden)
                    .permute(0, 2, 3, 1).reshape(batch * nodes, hidden, leads))
        for conv in self.tcn:
            temporal = temporal + self.temporal_dropout(torch.relu(conv(temporal)))
        return temporal.reshape(batch, nodes, hidden, leads).permute(0, 3, 1, 2).reshape(
            batch, leads, n_lat, n_lon, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.features(x)))
