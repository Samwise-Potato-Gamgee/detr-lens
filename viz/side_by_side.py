"""
Side-by-side comparison panel for two model outputs.

Renders a clean multi-panel image comparing Deformable DETR and RT-DETR
attention visualizations for the same input image.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional


def render_side_by_side(
    image: Image.Image,
    output_a: dict,
    output_b: dict,
    label_a: str = "Deformable DETR",
    label_b: str = "RT-DETR",
    layer_idx: int = -1,
    head_idx: Optional[int] = None,
    query_idx: int = 0,
    mode: str = "heatmap",   # "heatmap" | "offsets" | "rollout"
    cmap: str = "jet",
    alpha: float = 0.5,
    dpi: int = 150,
) -> Image.Image:
    """
    Render a 2-column comparison panel.

    Args:
        image:    original input image
        output_a: model output dict from model A
        output_b: model output dict from model B
        mode:     which visualization to use for each panel
        dpi:      output resolution

    Returns: PIL.Image RGB
    """
    from .heatmap import render_heatmap, render_rollout
    from .offset_grid import render_sampling_offsets

    def _render(output: dict, label: str) -> Image.Image:
        if mode == "heatmap":
            return render_heatmap(
                image, output,
                layer_idx=layer_idx, head_idx=head_idx,
                query_idx=query_idx, cmap=cmap, alpha=alpha,
                show_boxes=True,
            )
        elif mode == "rollout":
            return render_rollout(
                image, output,
                head_idx=head_idx, query_idx=query_idx,
                cmap="turbo", alpha=alpha, show_boxes=True,
            )
        elif mode == "offsets":
            return render_sampling_offsets(
                image, output,
                layer_idx=layer_idx, head_idx=head_idx,
                query_idx=query_idx, dpi=dpi,
            )
        else:
            return image.copy()

    img_a = _render(output_a, label_a)
    img_b = _render(output_b, label_b)

    # Resize both to same height
    target_h = max(img_a.height, img_b.height)
    if img_a.height != target_h:
        ratio = target_h / img_a.height
        img_a = img_a.resize((int(img_a.width * ratio), target_h), Image.LANCZOS)
    if img_b.height != target_h:
        ratio = target_h / img_b.height
        img_b = img_b.resize((int(img_b.width * ratio), target_h), Image.LANCZOS)

    label_h = 28
    total_w = img_a.width + img_b.width + 6
    total_h = target_h + label_h

    panel = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    panel.paste(img_a, (0, label_h))
    panel.paste(img_b, (img_a.width + 6, label_h))

    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((img_a.width // 2 - 60, 6), label_a, fill="white", font=font)
    draw.text((img_a.width + 6 + img_b.width // 2 - 40, 6), label_b, fill="white", font=font)

    return panel


def render_full_comparison(
    image: Image.Image,
    output_a: dict,
    output_b: dict,
    label_a: str = "Deformable DETR",
    label_b: str = "RT-DETR",
    layer_idx: int = -1,
    query_idx: int = 0,
    dpi: int = 120,
) -> Image.Image:
    """
    Render a 3×2 grid: (heatmap | offsets | rollout) × (model A | model B).
    """
    from .heatmap import render_heatmap, render_rollout
    from .offset_grid import render_sampling_offsets

    modes = ["heatmap", "offsets", "rollout"]
    labels_row = ["Attention Heatmap", "Sampling Offsets", "Attention Rollout"]

    rows_a, rows_b = [], []
    for mode in modes:
        if mode == "heatmap":
            va = render_heatmap(image, output_a, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
            vb = render_heatmap(image, output_b, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
        elif mode == "offsets":
            va = render_sampling_offsets(image, output_a, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
            vb = render_sampling_offsets(image, output_b, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
        elif mode == "rollout":
            va = render_rollout(image, output_a, query_idx=query_idx)
            vb = render_rollout(image, output_b, query_idx=query_idx)
        rows_a.append(va)
        rows_b.append(vb)

    # Standardize sizes
    row_h = max(max(r.height for r in rows_a), max(r.height for r in rows_b))
    row_w_a = max(r.width for r in rows_a)
    row_w_b = max(r.width for r in rows_b)

    def _pad(img, w, h):
        out = Image.new("RGB", (w, h), (30, 30, 30))
        out.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        return out

    label_h = 24
    cell_h = row_h + label_h
    total_w = row_w_a + row_w_b + 60  # margin
    total_h = cell_h * len(modes) + 40

    panel = Image.new("RGB", (total_w, total_h), (20, 20, 20))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(panel)
    try:
        font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_bold = ImageFont.load_default()
        font = font_bold

    # Column headers
    draw.text((row_w_a // 2 - 60, 8), label_a, fill="white", font=font_bold)
    draw.text((row_w_a + 60 + row_w_b // 2 - 40, 8), label_b, fill="white", font=font_bold)

    y_off = 30
    for i, (row_label, ra, rb) in enumerate(zip(labels_row, rows_a, rows_b)):
        ra = _pad(ra, row_w_a, row_h)
        rb = _pad(rb, row_w_b, row_h)
        draw.text((6, y_off + row_h // 2), row_label, fill="#aaaaaa", font=font)
        panel.paste(ra, (50, y_off))
        panel.paste(rb, (50 + row_w_a + 10, y_off))
        y_off += cell_h

    return panel
