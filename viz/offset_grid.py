"""
Deformable sampling-offset visualizer.

For each object query we know the reference point (implicit in the sampling
location tensor) and the predicted offsets. We draw:
  - A small circle at each sampling location
  - An arrow from the image centre-of-mass of the query's sampling cloud to
    each point, colored by attention weight
  - Optional: draw one panel per attention head, laid out in a grid

All rendering uses PIL + matplotlib (no cv2).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from PIL import Image
from typing import Optional


# colour palette per head (up to 16 heads)
_HEAD_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4", "#f032e6",
    "#bfef45", "#fabed4", "#469990", "#dcbeff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3",
]


def render_sampling_offsets(
    image: Image.Image,
    model_output: dict,
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    show_arrows: bool = True,
    arrow_scale: float = 1.0,
    point_size: float = 40.0,
    alpha_points: float = 0.9,
    alpha_image: float = 0.6,
    dpi: int = 150,
) -> Image.Image:
    """
    Render deformable sampling locations as colored dots and optional arrows.

    When head_idx is None all heads are shown with distinct colours.
    When head_idx is set only that head is shown.

    Dots are sized by their attention weight (larger = higher weight).
    If show_arrows=True, arrows radiate from the centroid of each head's
    sampling cloud to each individual point.

    Returns: PIL.Image RGB
    """
    sampling = model_output.get("sampling_offsets", [])
    if not sampling:
        return image.copy()

    attn_list = model_output.get("cross_attn_weights", [])
    H, W = model_output["image_size"]

    locs = sampling[layer_idx].squeeze(0)   # (n_queries, n_heads, n_levels, n_points, 2)
    locs = locs.numpy() if hasattr(locs, "numpy") else np.array(locs)
    q_locs = locs[query_idx]                # (n_heads, n_levels, n_points, 2)

    # Get attention weights for this query/layer
    if attn_list:
        attn = attn_list[layer_idx].squeeze(0)
        attn = attn.numpy() if hasattr(attn, "numpy") else np.array(attn)
        q_attn = attn[query_idx]            # (n_heads, n_levels, n_points)
    else:
        q_attn = np.ones(q_locs.shape[:3])  # uniform

    n_heads = q_locs.shape[0]

    # Select heads to show
    if head_idx is not None:
        heads = [head_idx]
    else:
        heads = list(range(n_heads))

    # Set up figure
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax.imshow(np.array(image.resize((W, H))), alpha=alpha_image)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    legend_handles = []

    for hi in heads:
        color = _HEAD_COLORS[hi % len(_HEAD_COLORS)]
        # Shape: (n_levels, n_points, 2)
        pts = q_locs[hi]          # (n_levels, n_points, 2)
        wts = q_attn[hi]          # (n_levels, n_points)

        # Normalize weights for sizing
        wts_flat = wts.reshape(-1)
        wts_norm = wts_flat / (wts_flat.max() + 1e-8)

        pts_x = pts[..., 0].reshape(-1) * W    # pixel x
        pts_y = pts[..., 1].reshape(-1) * H    # pixel y

        # Centroid of this head's cloud
        cx = pts_x.mean()
        cy = pts_y.mean()

        if show_arrows:
            for px, py, wn in zip(pts_x, pts_y, wts_norm):
                dx = px - cx
                dy = py - cy
                dist = np.sqrt(dx**2 + dy**2) + 1e-6
                # Draw arrow: starts from centroid, ends at point
                ax.annotate(
                    "",
                    xy=(px, py),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle=f"-|>,head_width={0.2 * arrow_scale},head_length={0.3 * arrow_scale}",
                        color=color,
                        alpha=float(0.4 + 0.5 * wn),
                        lw=float(0.8 + 1.0 * wn) * arrow_scale,
                    ),
                    annotation_clip=False,
                )

        # Draw sampling points, sized by weight
        sizes = (point_size * 0.5 + point_size * 2.0 * wts_norm)
        ax.scatter(
            pts_x, pts_y,
            s=sizes,
            c=color,
            alpha=alpha_points,
            zorder=5,
            linewidths=0.5,
            edgecolors="white",
        )

        if head_idx is None:
            patch = mpatches.Patch(color=color, label=f"Head {hi}")
            legend_handles.append(patch)

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=6,
            markerscale=0.8,
            framealpha=0.7,
        )

    ax.set_title(
        f"Deformable sampling locations · layer {layer_idx} · query {query_idx}",
        fontsize=8,
        pad=4,
    )

    plt.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(buf, mode="RGBA").convert("RGB")


def render_per_head_grid(
    image: Image.Image,
    model_output: dict,
    layer_idx: int = -1,
    query_idx: int = 0,
    point_size: float = 40.0,
    dpi: int = 100,
) -> Image.Image:
    """
    Render one panel per attention head in a grid layout.

    Returns a PIL.Image containing all heads.
    """
    sampling = model_output.get("sampling_offsets", [])
    if not sampling:
        return image.copy()

    locs = sampling[layer_idx].squeeze(0)
    locs = locs.numpy() if hasattr(locs, "numpy") else np.array(locs)
    q_locs = locs[query_idx]               # (n_heads, n_levels, n_points, 2)

    attn_list = model_output.get("cross_attn_weights", [])
    if attn_list:
        attn = attn_list[layer_idx].squeeze(0)
        attn = attn.numpy() if hasattr(attn, "numpy") else np.array(attn)
        q_attn = attn[query_idx]
    else:
        q_attn = np.ones(q_locs.shape[:3])

    H, W = model_output["image_size"]
    n_heads = q_locs.shape[0]
    ncols = 4
    nrows = int(np.ceil(n_heads / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.0), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    img_arr = np.array(image.resize((W, H)))

    for hi in range(n_heads):
        ax = axes[hi]
        ax.imshow(img_arr, alpha=0.5)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        ax.set_title(f"H{hi}", fontsize=7)

        color = _HEAD_COLORS[hi % len(_HEAD_COLORS)]
        pts = q_locs[hi]                  # (n_levels, n_points, 2)
        wts = q_attn[hi].reshape(-1)
        wts_norm = wts / (wts.max() + 1e-8)
        pts_x = pts[..., 0].reshape(-1) * W
        pts_y = pts[..., 1].reshape(-1) * H
        sizes = point_size * 0.5 + point_size * 1.5 * wts_norm
        ax.scatter(pts_x, pts_y, s=sizes, c=color, alpha=0.85, zorder=5,
                   linewidths=0.3, edgecolors="white")

    for hi in range(n_heads, len(axes)):
        axes[hi].set_visible(False)

    plt.suptitle(
        f"Sampling locs per head · layer {layer_idx} · query {query_idx}",
        fontsize=9,
    )
    plt.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return Image.fromarray(buf, mode="RGBA").convert("RGB")
