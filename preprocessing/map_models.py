"""Inference-only SPARC-Map architectures compatible with released checkpoints."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def build_knn_graph(coords: torch.Tensor, k: int, exclude_self: bool = True) -> torch.Tensor:
    distances = torch.cdist(coords, coords)
    if exclude_self:
        distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k, dim=1, largest=False).indices


class CoordEncoder(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, relative_coordinates: torch.Tensor) -> torch.Tensor:
        return self.mlp(relative_coordinates)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized = self.norm1(tokens)
        padding_mask = ~attn_mask if attn_mask is not None else None
        attention, _ = self.attn(normalized, normalized, normalized, key_padding_mask=padding_mask)
        tokens = tokens + attention
        return tokens + self.mlp(self.norm2(tokens))


class OriginalSpatialTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.coord_encoder = CoordEncoder(hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, center_emb, neighbor_embs, rel_coords, attn_mask=None):
        batch_size = neighbor_embs.shape[0]
        center = self.input_proj(center_emb).unsqueeze(1)
        neighbors = self.input_proj(neighbor_embs) + self.coord_encoder(rel_coords)
        tokens = torch.cat([center, neighbors], dim=1)
        full_mask = None
        if attn_mask is not None:
            full_mask = torch.cat([
                torch.ones(batch_size, 1, dtype=torch.bool, device=attn_mask.device), attn_mask
            ], dim=1)
        for block in self.transformer_blocks:
            tokens = block(tokens, attn_mask=full_mask)
        return self.head(tokens[:, 0])


class MeanResidSpatialTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256,
                 num_layers: int = 3, num_heads: int = 8, dropout: float = 0.1,
                 num_organs: int = 0, num_techs: int = 0):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.coord_encoder = CoordEncoder(hidden_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.use_organ = num_organs > 0
        self.use_tech = num_techs > 0
        if self.use_organ:
            self.organ_embedding = nn.Embedding(num_organs + 1, hidden_dim)
        if self.use_tech:
            self.tech_embedding = nn.Embedding(num_techs + 1, hidden_dim)
        self.mean_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(), nn.Linear(hidden_dim // 2, output_dim),
        )
        self.resid_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, center_emb, neighbor_embs, rel_coords, attn_mask=None,
                organ_id=None, tech_id=None):
        batch_size = neighbor_embs.shape[0]
        center = self.input_proj(center_emb).unsqueeze(1)
        neighbors = self.input_proj(neighbor_embs) + self.coord_encoder(rel_coords)
        extra = []
        if self.use_organ and organ_id is not None:
            extra.append(self.organ_embedding(organ_id).unsqueeze(1))
        if self.use_tech and tech_id is not None:
            extra.append(self.tech_embedding(tech_id).unsqueeze(1))
        prefix = extra + [center]
        tokens = torch.cat(prefix + [neighbors], dim=1)
        full_mask = None
        if attn_mask is not None:
            full_mask = torch.cat([
                torch.ones(batch_size, len(prefix), dtype=torch.bool, device=attn_mask.device), attn_mask
            ], dim=1)
        for block in self.transformer_blocks:
            tokens = block(tokens, attn_mask=full_mask)
        output = tokens[:, len(extra)]
        return self.resid_head(output), self.mean_head(output)


def denormalize_predictions_array(predictions: np.ndarray, norm_stats, label_norm: str,
                                  program_names, organ=None) -> np.ndarray:
    if label_norm == "none" or norm_stats is None:
        return predictions
    output = predictions.copy()
    if label_norm == "per_program":
        return output * norm_stats["std"] + norm_stats["mean"]
    if label_norm == "per_program_robust":
        return output * norm_stats["iqr"] + norm_stats["median"]
    if label_norm == "per_patient":
        return output
    if label_norm == "per_organ_per_program":
        stats = norm_stats.get("organ_stats", {}).get(
            organ, norm_stats.get("organ_stats", {}).get("__global__")
        )
        if stats is not None:
            return output * stats["std"] + stats["mean"]
        return output
    raise ValueError(f"Unsupported checkpoint label normalization: {label_norm}")
