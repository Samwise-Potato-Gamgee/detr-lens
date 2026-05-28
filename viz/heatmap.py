"""
Heatmap renderer for attention weights.

Supports:
  - Single-layer, single-head visualization
  - Average across heads
  - Attention rollout across decoder layers
  - Head diversity heatmap

All functions return PIL.Image objects.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from PIL import Image
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_heatmap_array(
    attn: np.ndarray,
    h: int, w: int,
    interp: str = "bilinear",
) -> np.ndarray:
    """
    Resize a flat/2-D attention array to (h, w) and normalise to [0, 1].

    attn can be:
      - 1-D (key_len,)  → needs spatial reshape
      - 2-D (H, W)      → used directly
    If key_len is a product of spatial dims we reshape, otherwise we just
    interpolate the flat sequence as a square.
    """
    if attn.ndim == 1:
        side = int(np.round(np.sqrt(attn.shape[0])))
        if side * side == attn.shape[0]:
            attn = attn.reshape(side, side)
        else:
            # Can't reshape cleanly — average over levels/points first
            attn = attn.reshape(-1)
            side = int(np.ceil(np.sqrt(attn.shape[0])))
            pad = side * side - attn.shape[0]
            if pad:
                attn = np.concatenate([attn, np.zeros(pad)])
            attn = attn.reshape(side, side)

    # Normalise
    vmin, vmax = attn.min(), attn.max()
    if vmax > vmin:
        attn = (attn - vmin) / (vmax - vmin)
    else:
        attn = np.zeros_like(attn)

    # Resize to target resolution using PIL
    pil = Image.fromarray((attn * 255).astype(np.uint8)).resize(
        (w, h), Image.BILINEAR
    )
    return np.array(pil, dtype=np.float32) / 255.0


def _colorise(arr: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    """Apply a matplotlib colormap to a [0,1] array → RGBA uint8."""
    cmap = cm.get_cmap(cmap_name)
    return (cmap(arr) * 255).astype(np.uint8)


def _blend(image: Image.Image, heatmap_rgba: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend heatmap onto image."""
    base = np.array(image.convert("RGBA"), dtype=np.float32)
    overlay = heatmap_rgba.astype(np.float32)
    blended = (1 - alpha) * base + alpha * overlay
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)).convert("RGB")


# ── deformable attn weight aggregation ───────────────────────────────────────

def _aggregate_deformable_weights(
    weights: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Convert deformable attention weights to a spatial heatmap.

    weights shape: (n_heads, n_levels, n_points) OR (n_levels, n_points)
    Returns: 2-D heatmap (img_h, img_w).

    Since deformable attention has discrete sampling points (not a dense grid),
    we cannot directly reshape to spatial.  We return a uniform image and rely
    on offset_grid.py for the spatial view.  Here we produce a per-point weight
    image by splatting each point with a Gaussian.

    weights: flattened (n_levels*n_points,) → uniform heatmap proxy.
    """
    # Just return a uniform representation; the rich spatial view is offset_grid
    out = np.ones((img_h, img_w), dtype=np.float32) * weights.mean()
    return out


# ── public API ────────────────────────────────────────────────────────────────

def _dense_to_image_heatmap(spatial: np.ndarray, H: int, W: int) -> np.ndarray:
    """Resize a (H_feat, W_feat) float32 map to (H, W) using bilinear interpolation."""
    pil = Image.fromarray((spatial * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def attention_to_heatmap(
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    attn_type: str = "cross",   # "cross" | "self"
) -> np.ndarray:
    """
    Extract and reshape attention weights to a 2-D heatmap array [0,1].

    Dense path (DAB-DETR, feat_hw present): weights (bs, n_heads, n_queries, seq_len)
    reshaped to (H_feat, W_feat) then bilinear-upsampled to image resolution.

    Deformable path (Deformable DETR / RT-DETR): weights (bs, n_queries, n_heads,
    n_levels, n_points) splatted with Gaussian kernels at sampling locations.

    Returns: np.ndarray shape (H, W) in [0,1].
    """
    H, W = model_output["image_size"]

    # ── Dense path (DAB-DETR) ─────────────────────────────────────────────────
    if "feat_hw" in model_output and attn_type == "cross":
        from extractors.dense_attn import extract_dense_cross_attention
        spatial = extract_dense_cross_attention(model_output, layer_idx, head_idx, query_idx)
        return _dense_to_image_heatmap(spatial, H, W)

    key = "cross_attn_weights" if attn_type == "cross" else "self_attn_weights"
    weights_list = model_output.get(key, [])

    if not weights_list:
        return np.zeros((H, W), dtype=np.float32)

    layer = weights_list[layer_idx]   # (bs, n_queries, n_heads, n_levels, n_points)
    # Remove batch dim
    layer = layer.squeeze(0).numpy() if hasattr(layer, "numpy") else np.array(layer.squeeze(0))

    # layer: (n_queries, n_heads, n_levels, n_points)
    attn = layer[query_idx]           # (n_heads, n_levels, n_points)

    if head_idx is not None:
        attn = attn[head_idx]         # (n_levels, n_points)
    else:
        attn = attn.mean(0)           # (n_levels, n_points)

    # Sum over levels, normalise
    attn_flat = attn.reshape(-1)      # (n_levels*n_points,)

    # Build spatial heatmap by splatting weights at sampling locations
    sampling = model_output.get("sampling_offsets", [])
    if sampling:
        locs = sampling[layer_idx]    # (bs, n_queries, n_heads, n_levels, n_points, 2)
        locs = locs.squeeze(0).numpy() if hasattr(locs, "numpy") else np.array(locs.squeeze(0))
        # locs: (n_queries, n_heads, n_levels, n_points, 2) — normalized [0,1]
        query_locs = locs[query_idx]  # (n_heads, n_levels, n_points, 2)

        if head_idx is not None:
            query_locs = query_locs[head_idx:head_idx+1]
            w_sel = attn[head_idx:head_idx+1] if head_idx is not None else attn
        else:
            w_sel = attn.mean(0, keepdims=True).repeat(query_locs.shape[0], axis=0) if head_idx is None else attn[head_idx:head_idx+1]
            w_sel = layer[query_idx].mean(0)  # (n_levels, n_points)

        # Splat all points
        canvas = np.zeros((H, W), dtype=np.float32)
        pts_x = (query_locs[..., 0].reshape(-1) * W).astype(int)
        pts_y = (query_locs[..., 1].reshape(-1) * H).astype(int)
        wts = attn_flat / (attn_flat.sum() + 1e-8)

        # Splat with Gaussian kernel radius proportional to image size
        radius = max(H, W) // 20
        from .heatmap_utils import splat_gaussian
        canvas = splat_gaussian(canvas, pts_x, pts_y, wts, radius)
        return canvas

    # Fallback: return normalised flat weights reshaped
    return _to_heatmap_array(attn_flat, H, W)


def render_heatmap(
    image: Image.Image,
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    attn_type: str = "cross",
    cmap: str = "jet",
    alpha: float = 0.5,
    show_boxes: bool = True,
    dpi: int = 150,
) -> Image.Image:
    """
    Render attention heatmap overlaid on the input image.

    Returns: PIL.Image RGB
    """
    H, W = model_output["image_size"]
    heatmap = attention_to_heatmap(model_output, layer_idx, head_idx, query_idx, attn_type)
    rgba = _colorise(heatmap, cmap)
    result = _blend(image.resize((W, H)), rgba, alpha)

    if show_boxes and len(model_output.get("boxes", [])) > 0:
        result = _draw_boxes(result, model_output, query_idx)

    return result


def render_rollout(
    image: Image.Image,
    model_output: dict,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    cmap: str = "turbo",
    alpha: float = 0.5,
    show_boxes: bool = True,
) -> Image.Image:
    """
    Attention rollout: accumulate attention maps across all decoder layers.

    Dense path (DAB-DETR): residual rollout on (H_feat, W_feat) maps, bilinear
    resize to image resolution.

    Deformable path: same accumulation with Gaussian splatting at sampling locs.
    """
    H, W = model_output["image_size"]
    weights_list = model_output.get("cross_attn_weights", [])

    # ── Dense path (DAB-DETR) ─────────────────────────────────────────────────
    if "feat_hw" in model_output:
        from extractors.dense_attn import extract_dense_rollout
        spatial = extract_dense_rollout(model_output, head_idx, query_idx)
        canvas = _dense_to_image_heatmap(spatial, H, W)
        rgba = _colorise(canvas, cmap)
        result = _blend(image.resize((W, H)), rgba, alpha)
        if show_boxes:
            result = _draw_boxes(result, model_output, query_idx)
        return result

    if not weights_list:
        return image.copy()

    rollout = None
    for w in weights_list:
        w = w.squeeze(0).numpy() if hasattr(w, "numpy") else np.array(w.squeeze(0))
        # (n_queries, n_heads, n_levels, n_points)
        q_w = w[query_idx]    # (n_heads, n_levels, n_points)
        if head_idx is not None:
            q_w = q_w[head_idx]
        else:
            q_w = q_w.mean(0)  # (n_levels, n_points)
        flat = q_w.reshape(-1)
        flat = flat / (flat.sum() + 1e-8)

        if rollout is None:
            rollout = flat
        else:
            # Residual-based rollout: A_rollout = 0.5*I + 0.5*A then accumulate
            a = 0.5 * np.ones_like(flat) / len(flat) + 0.5 * flat
            rollout = rollout * a
            rollout /= rollout.sum() + 1e-8

    # Build spatial heatmap using sampling locations of last layer
    sampling = model_output.get("sampling_offsets", [])
    canvas = np.zeros((H, W), dtype=np.float32)

    if sampling and rollout is not None:
        locs = sampling[-1].squeeze(0)
        if hasattr(locs, "numpy"):
            locs = locs.numpy()
        else:
            locs = np.array(locs)
        q_locs = locs[query_idx]  # (n_heads, n_levels, n_points, 2)
        if head_idx is not None:
            q_locs = q_locs[head_idx:head_idx+1]
        pts_x = (q_locs[..., 0].reshape(-1) * W).astype(int)
        pts_y = (q_locs[..., 1].reshape(-1) * H).astype(int)

        radius = max(H, W) // 15
        from .heatmap_utils import splat_gaussian
        canvas = splat_gaussian(canvas, pts_x, pts_y, rollout, radius)
    elif rollout is not None:
        canvas = _to_heatmap_array(rollout, H, W)

    rgba = _colorise(canvas, cmap)
    result = _blend(image.resize((W, H)), rgba, alpha)
    if show_boxes:
        result = _draw_boxes(result, model_output, query_idx)
    return result


def render_head_diversity(
    model_output: dict,
    layer_idx: int = -1,
    query_idx: int = 0,
    dpi: int = 100,
) -> Image.Image:
    """
    Compute pairwise cosine similarity between all attention heads and render
    as a small heatmap image.  Supports both dense (DAB-DETR) and deformable models.
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        H, W = model_output["image_size"]
        return Image.fromarray(np.zeros((H // 4, W // 4, 3), dtype=np.uint8))

    # ── Dense path (DAB-DETR) ─────────────────────────────────────────────────
    if "feat_hw" in model_output:
        from extractors.dense_attn import extract_dense_head_diversity
        sim = extract_dense_head_diversity(model_output, layer_idx, query_idx)
        n_heads = sim.shape[0]
    else:
        # Deformable path
        layer = weights_list[layer_idx].squeeze(0)   # (n_queries, n_heads, n_levels, n_points)
        if hasattr(layer, "numpy"):
            layer = layer.numpy()
        else:
            layer = np.array(layer)
        q = layer[query_idx]    # (n_heads, n_levels, n_points)
        n_heads = q.shape[0]
        flat = q.reshape(n_heads, -1)   # (n_heads, n_levels*n_points)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        normed = flat / norms
        sim = normed @ normed.T    # (n_heads, n_heads)

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=dpi)
    im = ax.imshow(sim, vmin=0, vmax=1, cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"Head cosine similarity (layer {layer_idx})", fontsize=9)
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_heads))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(buf, mode="RGBA").convert("RGB")


def compute_head_diversity_score(model_output: dict, layer_idx: int = -1, query_idx: int = 0) -> float:
    """
    Return mean pairwise cosine similarity of attention heads (lower = more diverse).
    Supports both dense (DAB-DETR) and deformable models.
    """
    weights_list = model_output.get("cross_attn_weights", [])
    if not weights_list:
        return 0.0

    if "feat_hw" in model_output:
        from extractors.dense_attn import extract_dense_head_diversity
        sim = extract_dense_head_diversity(model_output, layer_idx, query_idx)
    else:
        layer = weights_list[layer_idx].squeeze(0)
        if hasattr(layer, "numpy"):
            layer = layer.numpy()
        else:
            layer = np.array(layer)
        q = layer[query_idx]
        n_heads = q.shape[0]
        flat = q.reshape(n_heads, -1)
        norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
        normed = flat / norms
        sim = normed @ normed.T

    mask = np.triu(np.ones_like(sim, dtype=bool), k=1)
    return float(sim[mask].mean())


# ── box drawing ───────────────────────────────────────────────────────────────

def _draw_boxes(image: Image.Image, model_output: dict, query_idx: int) -> Image.Image:
    """Draw detection boxes; highlight the selected query's box."""
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(image.copy())
    img = image.copy()
    draw = ImageDraw.Draw(img)
    H, W = img.size[1], img.size[0]

    boxes = model_output.get("boxes", [])
    scores = model_output.get("scores", [])

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.tolist()
        color = "yellow" if i == query_idx else "white"
        width = 3 if i == query_idx else 1
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        if i == query_idx:
            draw.text((x1, max(0, y1 - 12)), f"{score:.2f}", fill="yellow")

    return img
