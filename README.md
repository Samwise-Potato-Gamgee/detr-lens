# detr-lens

Interactive attention visualization toolkit for **Deformable DETR** and **RT-DETR**.

Drop any image into the web app and compare attention maps side-by-side across both models — including a unique mode for visualizing deformable sampling offsets as directional arrows on the image.

---

## Features

| Mode | Description |
|------|-------------|
| **Attention Heatmap** | Cross-attention weights overlaid as a color heatmap. Supports per-layer, per-head, and head-averaged views. |
| **Deformable Sampling Offsets** | The unique feature: shows where in the image each attention head is actually looking, rendered as colored dots and arrows radiating from the query centroid. One color per head. |
| **Attention Rollout** | Accumulates attention across all decoder layers using the residual rollout method. Gives a holistic view of what each query aggregates from the encoder. |
| **Head Diversity** | Pairwise cosine-similarity heatmap between all attention heads in a layer. Low similarity = heads are specialized. |
| **Side-by-Side** | Compare Deformable DETR and RT-DETR simultaneously on the same image and query. |

---

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd detr-lens

# Create venv (Python 3.12 required)
python3.12 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt

# Download COCO val2017 (optional — needed for built-in examples)
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip val2017.zip && unzip annotations_trainval2017.zip
```

Model weights are downloaded automatically from HuggingFace Hub on first use:
- **Deformable DETR**: `SenseTime/deformable-detr`
- **RT-DETR**: `PekingU/rtdetr_r50vd`

---

## Launch the App

```bash
python app.py
```

Opens at `http://localhost:7860`. Upload any image or click one of the COCO examples.

### Controls

| Control | Description |
|---------|-------------|
| **Model** | Deformable DETR, RT-DETR, or Both (side-by-side) |
| **Visualization mode** | Heatmap / Sampling Offsets / Rollout / Head Diversity / Side-by-Side |
| **Decoder layer** | Negative index: -1 = last (most refined), -6 = first |
| **Attention head** | All (average) or a specific head index |
| **Object query index** | Which of the 300 object queries to visualize. Queries are not sorted by confidence — try different indices to find active detections. |
| **Colormap** | jet, turbo, viridis, plasma, hot, coolwarm |
| **Heatmap opacity** | Blend strength of the attention overlay |

---

## CLI Tools

### compare.py — batch visualization

Runs all visualization modes on a folder of images and saves PNGs to `results/`.

```bash
# Run RT-DETR on first 20 images in the COCO val set
python compare.py --input data/coco/val2017 --model rt-detr --n 20

# Run both models
python compare.py --input data/coco/val2017 --model both --n 10 --output results/

# All options
python compare.py \
    --input  <folder or image>   # required
    --model  rt-detr|deformable-detr|both  (default: both)
    --output results/            # output directory
    --n      20                  # max images
    --layer  -1                  # decoder layer index
    --query  0                   # object query index
    --head   None                # head index (default: average)
```

### export.py — publication-quality export

Exports PNGs at 300 DPI and optional animated GIFs sweeping all decoder layers.

```bash
# Export 5 images at 300 DPI
python export.py --input data/coco/val2017 --n 5 --dpi 300 --output exports/

# Also export animated GIFs (layers sweep)
python export.py --input data/coco/val2017 --n 5 --gif --gif-mode heatmap --output exports/

# All options
python export.py \
    --input  <folder or image>
    --model  rt-detr|deformable-detr|both  (default: both)
    --output exports/
    --n      5
    --dpi    300
    --layer  -1
    --query  0
    --gif                        # also save animated GIFs
    --gif-mode heatmap|rollout|offsets
```

---

## Visualization Modes — Details

### Attention Heatmap

Cross-attention weights are extracted per decoder layer and head via `register_forward_hook`. For deformable attention models, the weights have shape `(n_queries, n_heads, n_levels, n_points)`. We splat each sampling point onto the image canvas with a Gaussian kernel weighted by its attention value, producing a smooth spatial heatmap.

### Deformable Sampling Offsets

The key insight of Deformable DETR (and RT-DETR) is that instead of attending to all encoder positions, each query predicts a small set of reference points (sampling locations) in image space. These locations are extracted via a `register_forward_pre_hook` on the inner CUDA multi-scale deformable attention op, capturing the `(bs, n_queries, n_heads, n_levels, n_points, 2)` normalized coordinate tensor before the weighted aggregation.

Each head gets a distinct color. Dot sizes reflect attention weight. Arrows radiate from the per-head centroid to each sampling point, showing the "reach" of each head.

### Attention Rollout

Iterates across all decoder layers, accumulating attention maps using residual rollout:
```
A_rollout = 0.5 * I + 0.5 * A_layer
rollout    = rollout ⊗ A_rollout  (normalized)
```
This gives a view of the full decoder stack's receptive field for a given query.

### Head Diversity

Computes pairwise cosine similarity between all heads' flattened attention vectors in a given layer. Lower similarity = heads are more specialized. The scalar mean off-diagonal similarity is reported in the UI status line.

---

## Examples

Side-by-side comparisons of Deformable DETR (left) vs RT-DETR (right) on COCO val2017.

### Image 000000000139 — Heatmap
![heatmap](examples/000000000139_heatmap.png)

### Image 000000000139 — Sampling Offsets
![offsets](examples/000000000139_offsets.png)

### Image 000000000285 — Heatmap
![heatmap](examples/000000000285_heatmap.png)

### Image 000000000285 — Sampling Offsets
![offsets](examples/000000000285_offsets.png)

### Image 000000000632 — Rollout
![rollout](examples/000000000632_rollout.png)

---

## Project Structure

```
detr-lens/
├── app.py                   # Gradio web app
├── compare.py               # Batch visualization CLI
├── export.py                # Publication-quality export CLI
├── requirements.txt
├── models/
│   ├── deformable_detr.py   # Hook-based Deformable DETR wrapper
│   └── rtdetr.py            # Hook-based RT-DETR wrapper
├── extractors/
│   ├── cross_attn.py        # Cross-attention extraction utilities
│   ├── self_attn.py         # Self-attention extraction utilities
│   └── deformable_offsets.py# Sampling location extraction
├── viz/
│   ├── heatmap.py           # Heatmap + rollout + head diversity rendering
│   ├── heatmap_utils.py     # Gaussian splatting helper
│   ├── offset_grid.py       # Sampling offset arrow visualization
│   └── side_by_side.py      # Dual-model comparison panels
├── data/coco/               # COCO val2017 images + annotations
├── examples/                # Pre-generated example outputs
└── notebooks/
    └── exploration.ipynb    # Interactive exploration notebook
```

---

## Hardware Requirements

Designed for a single **RTX 4060 Ti (8–16 GB)**. Inference only — no training. Both models fit comfortably in VRAM simultaneously. If you hit memory limits, load one model at a time via the app's model selector.

---

## Technical Notes

- All hook registration is non-invasive — model weights and architecture are never modified.
- The inner multi-scale deformable attention CUDA op (`MultiScaleDeformableAttention`) receives sampling locations as a positional argument, captured via `register_forward_pre_hook`.
- No `cv2` dependency — all image I/O and rendering uses PIL and matplotlib.
- Python 3.12 · PyTorch 2.x · transformers ≥ 4.40 · Gradio 4+
