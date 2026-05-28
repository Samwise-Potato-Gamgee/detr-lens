"""
Dense (standard) attention extraction for DAB-DETR.

DAB-DETR cross-attention weights shape: (bs, n_heads, n_queries, seq_len)
where seq_len = H_feat * W_feat.  Functions here return spatial (H_feat, W_feat)
arrays directly — no Gaussian splatting needed.
"""

import math
import numpy as np
from typing import Optional


def _get_feat_hw(model_output: dict, seq_len: int) -> tuple[int, int]:
    hw = model_output.get("feat_hw")
    if hw is not None:
        return hw
    h = int(math.isqrt(seq_len))
    while h > 1 and seq_len % h != 0:
        h -= 1
    return (h, seq_len // h)


def _to_numpy(t) -> np.ndarray:
    return t.numpy() if hasattr(t, "numpy") else np.array(t)


def extract_dense_cross_attention(
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Extract a cross-attention map as a (H_feat, W_feat) float32 array.

    Attention weights shape: (bs, n_heads, n_queries, seq_len)
    Squeezes batch, selects query and optional head, reshapes to (H, W),
    normalizes to [0, 1].
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        hw = model_output.get("feat_hw", (1, 1))
        return np.zeros(hw, dtype=np.float32)

    # (n_heads, n_queries, seq_len)
    layer = _to_numpy(weights_list[layer_idx].squeeze(0))
    attn = layer[:, query_idx, :]   # (n_heads, seq_len)

    if head_idx is not None:
        flat = attn[head_idx].astype(np.float32)
    else:
        flat = attn.mean(0).astype(np.float32)

    H, W = _get_feat_hw(model_output, flat.shape[0])
    spatial = flat.reshape(H, W)

    vmin, vmax = spatial.min(), spatial.max()
    if vmax > vmin:
        spatial = (spatial - vmin) / (vmax - vmin)
    else:
        spatial = np.zeros_like(spatial)

    return spatial


def extract_dense_rollout(
    model_output: dict,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Residual rollout across all decoder cross-attention layers.

    Accumulates A_rollout = 0.5*uniform + 0.5*A_layer iteratively, giving a
    holistic view of what the query aggregates across the full decoder stack.

    Returns (H_feat, W_feat) float32 array normalized to [0, 1].
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        hw = model_output.get("feat_hw", (1, 1))
        return np.zeros(hw, dtype=np.float32)

    seq_len = weights_list[0].shape[-1]
    hw = _get_feat_hw(model_output, seq_len)
    rollout: Optional[np.ndarray] = None

    for w in weights_list:
        layer = _to_numpy(w.squeeze(0))            # (n_heads, n_queries, seq_len)
        attn = layer[:, query_idx, :]              # (n_heads, seq_len)
        if head_idx is not None:
            flat = attn[head_idx].astype(np.float32)
        else:
            flat = attn.mean(0).astype(np.float32)

        flat = flat / (flat.sum() + 1e-8)
        if rollout is None:
            rollout = flat
        else:
            a = 0.5 * np.ones_like(flat) / len(flat) + 0.5 * flat
            rollout = rollout * a
            rollout /= rollout.sum() + 1e-8

    H, W = hw
    spatial = rollout.reshape(H, W)
    vmax = spatial.max()
    if vmax > 0:
        spatial = spatial / vmax
    return spatial.astype(np.float32)


def extract_dense_head_diversity(
    model_output: dict,
    layer_idx: int = -1,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Pairwise cosine similarity between attention heads.

    Returns (n_heads, n_heads) float32 array with values in [0, 1].
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        return np.eye(1, dtype=np.float32)

    layer = _to_numpy(weights_list[layer_idx].squeeze(0))  # (n_heads, n_queries, seq_len)
    attn = layer[:, query_idx, :]                          # (n_heads, seq_len)

    norms = np.linalg.norm(attn, axis=1, keepdims=True) + 1e-8
    normed = attn / norms
    return (normed @ normed.T).astype(np.float32)
