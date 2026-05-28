#!/usr/bin/env python3
"""
compare.py — batch attention visualization CLI.

Usage:
    python compare.py --input data/coco/val2017 --model RT-DETR --n 10
    python compare.py --input path/to/images --model both --query 0
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from models.deformable_detr import DeformableDETRWrapper
from models.rtdetr import RTDETRWrapper
from viz.heatmap import render_heatmap, render_rollout, render_head_diversity
from viz.offset_grid import render_sampling_offsets, render_per_head_grid
from viz.side_by_side import render_side_by_side


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: str, n: int) -> list[Path]:
    p = Path(folder)
    if p.is_file():
        return [p]
    imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS])
    return imgs[:n]


def run_and_save(
    img_path: Path,
    models: dict,
    out_dir: Path,
    layer_idx: int,
    query_idx: int,
    head_idx,
):
    image = Image.open(img_path).convert("RGB")
    stem = img_path.stem
    outputs: dict[str, dict] = {}

    for mname, wrapper in models.items():
        try:
            outputs[mname] = wrapper.forward(image)
        except Exception as e:
            print(f"  [error] {mname} on {img_path.name}: {e}")
            continue

    for mname, out in outputs.items():
        prefix = out_dir / f"{stem}_{mname.replace(' ', '_')}"

        # Heatmap
        try:
            h = render_heatmap(image, out, layer_idx=layer_idx, query_idx=query_idx)
            h.save(str(prefix) + "_heatmap.png")
        except Exception as e:
            print(f"  heatmap error: {e}")

        # Rollout
        try:
            r = render_rollout(image, out, query_idx=query_idx)
            r.save(str(prefix) + "_rollout.png")
        except Exception as e:
            print(f"  rollout error: {e}")

        # Sampling offsets
        try:
            o = render_sampling_offsets(image, out, layer_idx=layer_idx, query_idx=query_idx)
            o.save(str(prefix) + "_offsets.png")
        except Exception as e:
            print(f"  offsets error: {e}")

        # Head diversity
        try:
            d = render_head_diversity(out, layer_idx=layer_idx, query_idx=query_idx)
            d.save(str(prefix) + "_head_diversity.png")
        except Exception as e:
            print(f"  head_diversity error: {e}")

    # Side-by-side if both models available
    if "Deformable DETR" in outputs and "RT-DETR" in outputs:
        try:
            s = render_side_by_side(
                image,
                outputs["Deformable DETR"],
                outputs["RT-DETR"],
                layer_idx=layer_idx,
                query_idx=query_idx,
                mode="heatmap",
            )
            s.save(str(out_dir / f"{stem}_side_by_side.png"))
        except Exception as e:
            print(f"  side_by_side error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Batch attention visualization")
    parser.add_argument("--input", required=True, help="Image folder or single image path")
    parser.add_argument("--model", default="both",
                        choices=["deformable-detr", "rt-detr", "both"],
                        help="Which model(s) to run")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--n", type=int, default=20, help="Max number of images to process")
    parser.add_argument("--layer", type=int, default=-1, help="Decoder layer index")
    parser.add_argument("--query", type=int, default=0, help="Object query index")
    parser.add_argument("--head", type=int, default=None, help="Attention head (default: average)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input, args.n)
    if not images:
        print("No images found.")
        sys.exit(1)

    print(f"Found {len(images)} image(s). Loading model(s)...")

    models: dict[str, object] = {}
    if args.model in ("deformable-detr", "both"):
        w = DeformableDETRWrapper()
        w.load()
        models["Deformable DETR"] = w
    if args.model in ("rt-detr", "both"):
        w = RTDETRWrapper()
        w.load()
        models["RT-DETR"] = w

    print(f"Models loaded. Processing {len(images)} image(s) → {out_dir}/")

    for img_path in tqdm(images, unit="img"):
        run_and_save(img_path, models, out_dir, args.layer, args.query, args.head)

    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
