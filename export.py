#!/usr/bin/env python3
"""
export.py — publication-quality export of attention visualizations.

Exports:
  - PNG at 300 DPI (for print/paper figures)
  - Animated GIF sweeping decoder layers (for README/supplementary)

Usage:
    python export.py --input path/to/image.jpg --model both --output exports/
    python export.py --input data/coco/val2017 --n 5 --gif --output exports/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.deformable_detr import DeformableDETRWrapper
from models.rtdetr import RTDETRWrapper
from viz.heatmap import render_heatmap, render_rollout
from viz.offset_grid import render_sampling_offsets
from viz.side_by_side import render_side_by_side


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRINT_DPI = 300
GIF_DPI = 100
GIF_FRAME_MS = 500


def collect_images(folder: str, n: int) -> list[Path]:
    p = Path(folder)
    if p.is_file():
        return [p]
    return sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])[:n]


def _render_at_dpi(image: Image.Image, out: dict, dpi: int,
                   layer_idx: int, query_idx: int, mode: str) -> Image.Image:
    """Render a visualization at the requested DPI."""
    if mode == "heatmap":
        return render_heatmap(image, out, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
    elif mode == "rollout":
        return render_rollout(image, out, query_idx=query_idx)
    elif mode == "offsets":
        return render_sampling_offsets(image, out, layer_idx=layer_idx, query_idx=query_idx, dpi=dpi)
    return image.copy()


def export_png(
    image: Image.Image,
    outputs: dict,
    out_dir: Path,
    stem: str,
    layer_idx: int,
    query_idx: int,
    dpi: int = PRINT_DPI,
):
    """Export PNG files at publication DPI."""
    for mode in ["heatmap", "rollout", "offsets"]:
        for mname, out in outputs.items():
            vis = _render_at_dpi(image, out, dpi, layer_idx, query_idx, mode)
            fname = out_dir / f"{stem}_{mname.replace(' ', '_')}_{mode}_{dpi}dpi.png"
            # Save with DPI metadata
            vis.save(str(fname), dpi=(dpi, dpi))

    if "Deformable DETR" in outputs and "RT-DETR" in outputs:
        s = render_side_by_side(
            image,
            outputs["Deformable DETR"],
            outputs["RT-DETR"],
            layer_idx=layer_idx,
            query_idx=query_idx,
            mode="heatmap",
            dpi=dpi,
        )
        s.save(str(out_dir / f"{stem}_comparison_{dpi}dpi.png"), dpi=(dpi, dpi))


def export_gif(
    image: Image.Image,
    outputs: dict,
    out_dir: Path,
    stem: str,
    query_idx: int,
    mode: str = "heatmap",
    fps: int = 2,
):
    """Export animated GIF sweeping all decoder layers."""
    n_layers = max(len(o["cross_attn_weights"]) for o in outputs.values()) if outputs else 0
    if n_layers == 0:
        return

    for mname, out in outputs.items():
        frames = []
        n = len(out["cross_attn_weights"])
        for li in range(n):
            vis = _render_at_dpi(image, out, GIF_DPI, li, query_idx, mode)
            # Add layer label
            from PIL import ImageDraw, ImageFont
            draw_img = vis.copy()
            draw = ImageDraw.Draw(draw_img)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            draw.text((8, 8), f"Layer {li}", fill="white", font=font)
            draw.text((9, 9), f"Layer {li}", fill="black", font=font)  # shadow
            frames.append(draw_img.convert("P", palette=Image.ADAPTIVE, colors=256))

        fname = out_dir / f"{stem}_{mname.replace(' ', '_')}_{mode}_layers.gif"
        frames[0].save(
            str(fname),
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=int(1000 / fps),
            optimize=True,
        )
        print(f"  Saved GIF: {fname.name} ({n} frames)")


def main():
    parser = argparse.ArgumentParser(description="Export publication-quality attention visualizations")
    parser.add_argument("--input", required=True, help="Image or folder")
    parser.add_argument("--model", default="both",
                        choices=["deformable-detr", "rt-detr", "both"])
    parser.add_argument("--output", default="exports", help="Output directory")
    parser.add_argument("--n", type=int, default=5, help="Max images to export")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=PRINT_DPI, help="PNG DPI")
    parser.add_argument("--gif", action="store_true", help="Also export animated GIFs")
    parser.add_argument("--gif-mode", default="heatmap",
                        choices=["heatmap", "rollout", "offsets"],
                        help="Visualization mode for GIF frames")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input, args.n)
    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"Loading model(s)...")
    models: dict[str, object] = {}
    if args.model in ("deformable-detr", "both"):
        w = DeformableDETRWrapper()
        w.load()
        models["Deformable DETR"] = w
    if args.model in ("rt-detr", "both"):
        w = RTDETRWrapper()
        w.load()
        models["RT-DETR"] = w

    print(f"Exporting {len(images)} image(s) → {out_dir}/")

    for img_path in tqdm(images, unit="img"):
        image = Image.open(img_path).convert("RGB")
        stem = img_path.stem
        outputs: dict[str, dict] = {}

        for mname, wrapper in models.items():
            try:
                outputs[mname] = wrapper.forward(image)
            except Exception as e:
                print(f"  [error] {mname}: {e}")

        if outputs:
            export_png(image, outputs, out_dir, stem, args.layer, args.query, args.dpi)
            if args.gif:
                export_gif(image, outputs, out_dir, stem, args.query, args.gif_mode)

    torch.cuda.empty_cache()
    print("Export complete.")


if __name__ == "__main__":
    main()
