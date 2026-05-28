"""
Cross-attention extraction utilities.

Both Deformable DETR and RT-DETR use multi-scale deformable attention with shape:
    (batch, n_queries, n_heads, n_levels, n_points)

The returned flat 1-D array has length n_heads * n_levels * n_points.
"""

import torch
import numpy as np
from typing import Optional


def _squeeze_batch(layer: torch.Tensor) -> np.ndarray:
    """Remove batch dim and convert to numpy. Shape: (n_queries, n_heads, n_levels, n_points)."""
    arr = layer.squeeze(0)
    return arr.numpy() if hasattr(arr, "numpy") else np.array(arr)


def extract_cross_attention(
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Extract flattened cross-attention weights for a specific query.

    Returns:
        1-D np.ndarray of shape (n_levels * n_points,) when head_idx is set,
        or (n_heads * n_levels * n_points,) when head_idx is None (all heads concatenated).
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        raise ValueError("No cross-attention weights found.")

    layer = _squeeze_batch(weights_list[layer_idx])   # (n_queries, n_heads, n_levels, n_points)
    attn = layer[query_idx]                            # (n_heads, n_levels, n_points)

    if head_idx is not None:
        return attn[head_idx].reshape(-1)              # (n_levels * n_points,)
    else:
        return attn.mean(0).reshape(-1)                # (n_levels * n_points,)  — head-averaged


def extract_all_heads(
    model_output: dict,
    layer_idx: int = -1,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Return attention weights for all heads.

    Returns:
        np.ndarray of shape (n_heads, n_levels * n_points)
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        raise ValueError("No cross-attention weights found.")

    layer = _squeeze_batch(weights_list[layer_idx])    # (n_queries, n_heads, n_levels, n_points)
    attn = layer[query_idx]                            # (n_heads, n_levels, n_points)
    n_heads = attn.shape[0]
    return attn.reshape(n_heads, -1)                   # (n_heads, n_levels * n_points)


def get_n_layers(model_output: dict) -> int:
    return len(model_output.get("cross_attn_weights", []))


def get_n_heads(model_output: dict, layer_idx: int = -1) -> int:
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        return 0
    # shape: (bs, n_queries, n_heads, n_levels, n_points)
    return weights_list[layer_idx].shape[2]
