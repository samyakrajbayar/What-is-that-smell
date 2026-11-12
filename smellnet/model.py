"""Pure-PyTorch port of OpenPOM's MPNN-POM.

Layer-for-layer equivalent to openpom/models/mpnn_pom.py, with DGL's batching
and message passing replaced by explicit index operations:

    atoms -> Linear+ReLU -> [NNConv(residual) -> ReLU -> GRU] x steps
          -> radius-0 edge/atom folding -> Set2Set -> FFN -> 138 odor logits

The feed-forward head also returns the 256-d Principal Odor Map embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import DEFAULT_CONFIG, N_TASKS
from .featurizer import ATOM_FDIM, BOND_FDIM, MolGraph


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
@dataclass
class GraphBatch:
    atom_features: torch.Tensor  # (N, 134)
    edge_index: torch.Tensor  # (2, E)
    bond_features: torch.Tensor  # (E, 6)
    batch: torch.Tensor  # (N,) graph index per atom
    num_graphs: int

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            self.atom_features.to(device),
            self.edge_index.to(device),
            self.bond_features.to(device),
            self.batch.to(device),
            self.num_graphs,
        )


def collate_graphs(graphs: Sequence[MolGraph]) -> GraphBatch:
    if not graphs:
        raise ValueError("cannot collate an empty batch")

    atom_feats: List[np.ndarray] = []
    edge_indices: List[np.ndarray] = []
    bond_feats: List[np.ndarray] = []
    batches: List[np.ndarray] = []
    offset = 0
    for graph_idx, graph in enumerate(graphs):
        atom_feats.append(graph.atom_features)
        edge_indices.append(graph.edge_index + offset)
        bond_feats.append(graph.bond_features)
        batches.append(np.full(graph.num_atoms, graph_idx, dtype=np.int64))
        offset += graph.num_atoms

    return GraphBatch(
        atom_features=torch.from_numpy(np.concatenate(atom_feats, axis=0)),
        edge_index=torch.from_numpy(np.concatenate(edge_indices, axis=1)),
        bond_features=torch.from_numpy(np.concatenate(bond_feats, axis=0)),
        batch=torch.from_numpy(np.concatenate(batches, axis=0)),
        num_graphs=len(graphs),
    )


def _segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = values.new_zeros((size, values.shape[1]))
    out.index_add_(0, index, values)
    return out


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------
class NNConvLayer(nn.Module):
    """DGL NNConv: m_ij = W(e_ij) * h_j, summed at the destination.

    With residual=True the destination node representation is added back,
    exactly as in openpom's CustomMPNNGNN (NNConv(residual=True)).
    """

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        edge_func: nn.Module,
        aggregator_type: str = "sum",
        residual: bool = True,
    ) -> None:
        super().__init__()
        if aggregator_type not in {"sum", "mean", "max"}:
            raise ValueError(f"unknown aggregator_type: {aggregator_type}")
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.edge_func = edge_func
        self.aggregator_type = aggregator_type
        self.residual = residual
        self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        weights = self.edge_func(edge_feats).view(-1, self.in_feats, self.out_feats)
        messages = torch.bmm(node_feats[src].unsqueeze(1), weights).squeeze(1)
        num_nodes = node_feats.shape[0]

        if self.aggregator_type == "sum":
            aggregated = _segment_sum(messages, dst, num_nodes)
        elif self.aggregator_type == "mean":
            counts = torch.zeros(num_nodes, device=node_feats.device).index_add_(
                0, dst, torch.ones_like(dst, dtype=node_feats.dtype)
            )
            aggregated = _segment_sum(messages, dst, num_nodes) / counts.clamp(min=1.0).unsqueeze(-1)
        else:  # max
            aggregated = messages.new_full((num_nodes, self.out_feats), -1e9)
            index = dst.view(-1, 1).expand_as(messages)
            aggregated.scatter_reduce_(0, index, messages, reduce="amax", include_self=False)
            has_edge = torch.zeros(num_nodes, dtype=torch.bool, device=node_feats.device)
            has_edge[dst] = True
            aggregated = torch.where(has_edge.unsqueeze(-1), aggregated, torch.zeros_like(aggregated))

        out = aggregated + self.bias
        if self.residual:
            out = out + node_feats
        return out


class MPNNBlock(nn.Module):
    """dgllife MPNNGNN with OpenPOM's residual/aggregator options."""

    def __init__(
        self,
        node_in_feats: int = ATOM_FDIM,
        edge_in_feats: int = BOND_FDIM,
        node_out_feats: int = 100,
        edge_hidden_feats: int = 75,
        num_step_message_passing: int = 5,
        residual: bool = True,
        aggregator_type: str = "sum",
    ) -> None:
        super().__init__()
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats), nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats),
        )
        self.gnn_layer = NNConvLayer(
            node_out_feats,
            node_out_feats,
            edge_network,
            aggregator_type=aggregator_type,
            residual=residual,
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)
        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(node_feats, edge_index, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats


class Set2Set(nn.Module):
    """DGL Set2Set readout, batched with explicit segment operations."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers)

    def forward(
        self, feat: torch.Tensor, batch: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        device = feat.device
        dtype = feat.dtype
        h = (
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )
        q_star = feat.new_zeros((batch_size, self.output_dim))

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)
            e = (feat * q[batch]).sum(dim=-1, keepdim=True)

            # per-graph softmax over atoms
            graph_max = e.new_full((batch_size, 1), float("-inf"))
            graph_max.scatter_reduce_(
                0, batch.view(-1, 1), e, reduce="amax", include_self=False
            )
            graph_max = torch.nan_to_num(graph_max, neginf=0.0)
            exp_e = torch.exp(e - graph_max[batch])
            denom = exp_e.new_zeros((batch_size, 1)).index_add_(0, batch, exp_e)
            alpha = exp_e / denom[batch].clamp(min=1e-12)

            readout = _segment_sum(feat * alpha, batch, batch_size)
            q_star = torch.cat([q, readout], dim=-1)

        return q_star


class PositionwiseFFN(nn.Module):
    """OpenPOM's FFN with batch norm, dropout and POM embedding output."""

    def __init__(
        self,
        d_input: int,
        d_hidden_list: Sequence[int],
        d_output: int,
        activation: str = "relu",
        dropout_p: float = 0.0,
        dropout_at_input_no_act: bool = False,
    ) -> None:
        super().__init__()
        activations = {
            "relu": nn.ReLU,
            "leakyrelu": lambda: nn.LeakyReLU(0.1),
            "prelu": nn.PReLU,
            "tanh": nn.Tanh,
            "selu": nn.SELU,
            "elu": nn.ELU,
            "linear": nn.Identity,
        }
        if activation not in activations:
            raise ValueError(f"unknown activation: {activation}")
        self.activation = activations[activation]()
        self.dropout_p = float(dropout_p)
        self.dropout_at_input_no_act = dropout_at_input_no_act
        self.n_layers = len(d_hidden_list) + 1

        sizes = [d_input, *d_hidden_list, d_output]
        self.linears = nn.ModuleList(
            [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(size) for size in d_hidden_list]
        )
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return None, self.linears[0](self.dropout(x))
            return None, self.dropout(self.activation(self.linears[0](x)))

        if self.dropout_at_input_no_act:
            x = self.dropout(x)

        for i in range(self.n_layers - 2):
            x = self.dropout(self.activation(self.batchnorms[i](self.linears[i](x))))

        embeddings = self.linears[self.n_layers - 2](x)
        x = self.dropout(
            self.activation(self.batchnorms[self.n_layers - 2](embeddings))
        )
        output = self.linears[-1](x)
        return embeddings, output


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class MPNNPOM(nn.Module):
    """Principal Odor Map MPNN, ported from OpenPOM's MPNNPOM."""

    def __init__(
        self,
        n_tasks: int = N_TASKS,
        node_out_feats: int = 100,
        edge_hidden_feats: int = 75,
        edge_out_feats: int = 100,
        num_step_message_passing: int = 5,
        mpnn_residual: bool = True,
        message_aggregator_type: str = "sum",
        readout_type: str = "set2set",
        num_step_set2set: int = 3,
        num_layer_set2set: int = 2,
        ffn_hidden_list: Sequence[int] = (392, 392),
        ffn_embeddings: int = 256,
        ffn_activation: str = "relu",
        ffn_dropout_p: float = 0.12,
        ffn_dropout_at_input_no_act: bool = False,
        number_atom_features: int = ATOM_FDIM,
        number_bond_features: int = BOND_FDIM,
    ) -> None:
        super().__init__()
        self.n_tasks = n_tasks
        self.ffn_embeddings = ffn_embeddings
        self.readout_type = readout_type

        self.mpnn = MPNNBlock(
            node_in_feats=number_atom_features,
            edge_in_feats=number_bond_features,
            node_out_feats=node_out_feats,
            edge_hidden_feats=edge_hidden_feats,
            num_step_message_passing=num_step_message_passing,
            residual=mpnn_residual,
            aggregator_type=message_aggregator_type,
        )
        self.project_edge_feats = nn.Sequential(
            nn.Linear(number_bond_features, edge_out_feats), nn.ReLU()
        )

        if readout_type == "set2set":
            self.readout_set2set = Set2Set(
                input_dim=node_out_feats + edge_out_feats,
                n_iters=num_step_set2set,
                n_layers=num_layer_set2set,
            )
            ffn_input = 2 * (node_out_feats + edge_out_feats)
        elif readout_type == "global_sum_pooling":
            ffn_input = node_out_feats + edge_out_feats
        else:
            raise ValueError(f"unknown readout_type: {readout_type}")

        d_hidden_list = list(ffn_hidden_list)
        if ffn_embeddings is not None:
            d_hidden_list = d_hidden_list + [ffn_embeddings]

        self.ffn = PositionwiseFFN(
            d_input=ffn_input,
            d_hidden_list=d_hidden_list,
            d_output=n_tasks,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act,
        )

        self.config = {
            "n_tasks": n_tasks,
            "node_out_feats": node_out_feats,
            "edge_hidden_feats": edge_hidden_feats,
            "edge_out_feats": edge_out_feats,
            "num_step_message_passing": num_step_message_passing,
            "mpnn_residual": mpnn_residual,
            "message_aggregator_type": message_aggregator_type,
            "readout_type": readout_type,
            "num_step_set2set": num_step_set2set,
            "num_layer_set2set": num_layer_set2set,
            "ffn_hidden_list": list(ffn_hidden_list),
            "ffn_embeddings": ffn_embeddings,
            "ffn_activation": ffn_activation,
            "ffn_dropout_p": ffn_dropout_p,
            "ffn_dropout_at_input_no_act": ffn_dropout_at_input_no_act,
            "number_atom_features": number_atom_features,
            "number_bond_features": number_bond_features,
        }

    def _readout(self, batch_graph: GraphBatch, node_encodings: torch.Tensor) -> torch.Tensor:
        edge_embeds = self.project_edge_feats(batch_graph.bond_features)
        src, dst = batch_graph.edge_index
        folded = torch.cat([node_encodings[src], edge_embeds], dim=-1)
        node_set = _segment_sum(folded, dst, node_encodings.shape[0])

        if self.readout_type == "set2set":
            return self.readout_set2set(node_set, batch_graph.batch, batch_graph.num_graphs)
        return _segment_sum(node_set, batch_graph.batch, batch_graph.num_graphs)

    def forward(
        self, batch_graph: GraphBatch
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_encodings = self.mpnn(
            batch_graph.atom_features,
            batch_graph.edge_index,
            batch_graph.bond_features,
        )
        molecular_encodings = self._readout(batch_graph, node_encodings)
        if self.readout_type == "global_sum_pooling":
            molecular_encodings = F.softmax(molecular_encodings, dim=1)

        embeddings, logits = self.ffn(molecular_encodings)
        proba = torch.sigmoid(logits)
        return proba, logits, embeddings


def build_model(config: Optional[dict] = None, **overrides) -> MPNNPOM:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    merged.update(overrides)
    return MPNNPOM(**merged)


def load_model(path, device: torch.device = torch.device("cpu")) -> Tuple[MPNNPOM, dict]:
    """Load a smellnet checkpoint, returning the model and metadata."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint.get("config"))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    metadata = {k: v for k, v in checkpoint.items() if k not in {"state_dict", "config"}}
    metadata["config"] = model.config
    return model, metadata
