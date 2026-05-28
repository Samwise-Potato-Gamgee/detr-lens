"""
Deformable sampling location extraction.

Deformable DETR computes a set of reference points per query, then predicts
offsets around those reference points. The actual sampling locations are:

    sampling_loc = reference_point + offset * scale

The hook captures the *normalized* sampling locations in [0, 1] image space.

Shape from model: (bs, n_heads, n_queries, n_levels, n_points, 2)
  where dim -1 is (x, y) in normalized [0,1] coords.
"""

import torch
import numpy as np
from typing import Optional


def extract_sampling_locations(
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    level_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Extract normalized (x, y) sampling locations for a query.

    Args:
        model_output: dict from model wrapper
        layer_idx:    decoder layer index
        head_idx:     head index (None = all heads)
        query_idx:    object query index
        level_idx:    feature level index (None = all levels)

    Returns:
        np.ndarray of shape (N, 2) with (x, y) in [0, 1] coords.
        N = n_heads * n_levels * n_points (or subset if head/level selected).
    """
    offsets_list = model_output.get("sampling_offsets", [])
    if not offsets_list:
        return np.empty((0, 2))

    # shape: (bs, n_heads, n_queries, n_levels, n_points, 2)
    locs = offsets_list[layer_idx].squeeze(0)  # (heads, queries, levels, points, 2)
    locs = locs[:, query_idx, :, :, :]         # (heads, levels, points, 2)

    if head_idx is not None:
        locs = locs[head_idx:head_idx+1]       # (1, levels, points, 2)
    if level_idx is not None:
        locs = locs[:, level_idx:level_idx+1]  # (heads, 1, points, 2)

    # Flatten to (N, 2)
    return locs.reshape(-1, 2).numpy()


def extract_sampling_locations_per_head(
    model_output: dict,
    layer_idx: int = -1,
    query_idx: int = 0,
) -> dict[int, np.ndarray]:
    """
    Return sampling locations grouped by head index.

    Returns:
        dict mapping head_idx -> np.ndarray of shape (n_levels * n_points, 2)
    """
    offsets_list = model_output.get("sampling_offsets", [])
    if not offsets_list:
        return {}

    locs = offsets_list[layer_idx].squeeze(0)  # (heads, queries, levels, points, 2)
    locs = locs[:, query_idx, :, :, :]         # (heads, levels, points, 2)
    n_heads = locs.shape[0]

    return {
        h: locs[h].reshape(-1, 2).numpy()
        for h in range(n_heads)
    }


def get_n_sampling_points(model_output: dict, layer_idx: int = -1) -> tuple[int, int, int]:
    """Return (n_heads, n_levels, n_points) or (0,0,0) if unavailable."""
    offsets_list = model_output.get("sampling_offsets", [])
    if not offsets_list:
        return (0, 0, 0)
    locs = offsets_list[layer_idx].squeeze(0)
    return locs.shape[0], locs.shape[2], locs.shape[3]
