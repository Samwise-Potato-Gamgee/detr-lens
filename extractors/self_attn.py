"""
Self-attention extraction utilities.

Self-attention weights shape: (batch, n_queries, n_heads, n_queries)
Note: RT-DETR/Deformable DETR with sdpa may not return self-attn weights;
in that case an empty array is returned gracefully.
"""

import numpy as np
from typing import Optional


def extract_self_attention(
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
) -> np.ndarray:
    """
    Extract self-attention weights for a specific query token.

    Returns:
        np.ndarray of shape (n_queries,) — attention weights this query places
        on all other queries, or empty array if unavailable.
    """
    weights_list = model_output.get("self_attn_weights", [])
    if not weights_list:
        return np.array([])

    layer = weights_list[layer_idx]
    arr = layer.squeeze(0)
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    else:
        import numpy as np_
        arr = np_.array(arr)

    # Shape can be (n_queries, n_heads, n_queries) or (n_heads, n_queries, n_queries)
    if arr.ndim == 3:
        if arr.shape[0] > arr.shape[2]:
            # Assume (n_queries, n_heads, n_queries)
            attn = arr[query_idx]          # (n_heads, n_queries)
        else:
            # Assume (n_heads, n_queries, n_queries)
            attn = arr[:, query_idx, :]    # (n_heads, n_queries)
    else:
        return np.array([])

    if head_idx is not None:
        attn = attn[head_idx]
    else:
        attn = attn.mean(0)

    return attn
