"""
detr-lens Gradio app.

Interactive comparison of Deformable DETR and RT-DETR attention maps.
Run with: .venv/bin/python app.py
"""

import os
import gc
import torch
import gradio as gr
import numpy as np
from PIL import Image
from typing import Optional

from models.deformable_detr import DeformableDETRWrapper
from models.rtdetr import RTDETRWrapper
from models.dab_detr import DABDETRWrapper
from viz.heatmap import render_heatmap, render_rollout, render_head_diversity, compute_head_diversity_score
from viz.offset_grid import render_sampling_offsets, render_per_head_grid
from viz.side_by_side import render_side_by_side

# ── global model singletons (loaded on first use) ────────────────────────────

_deformable_detr: Optional[DeformableDETRWrapper] = None
_rtdetr: Optional[RTDETRWrapper] = None
_dab_detr: Optional[DABDETRWrapper] = None


def get_model(name: str):
    global _deformable_detr, _rtdetr, _dab_detr
    if name == "Deformable DETR":
        if _deformable_detr is None:
            _deformable_detr = DeformableDETRWrapper()
        _deformable_detr.load()
        return _deformable_detr
    elif name == "DAB-DETR":
        if _dab_detr is None:
            _dab_detr = DABDETRWrapper()
        _dab_detr.load()
        return _dab_detr
    else:
        if _rtdetr is None:
            _rtdetr = RTDETRWrapper()
        _rtdetr.load()
        return _rtdetr


# ── inference cache (avoids re-running the model on every slider move) ───────

_cache: dict = {}


def _cache_key(image: Image.Image, model_name: str) -> str:
    return f"{id(image)}_{model_name}"


def run_model(image: Image.Image, model_name: str) -> dict:
    key = _cache_key(image, model_name)
    if key not in _cache:
        if len(_cache) > 4:
            _cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
        model = get_model(model_name)
        _cache[key] = model.forward(image)
    return _cache[key]


# ── main inference + visualisation function ──────────────────────────────────

def process(
    input_image,
    model_choice: str,
    viz_mode: str,
    layer_idx: int,
    head_choice: str,
    query_idx: int,
    cmap: str,
    alpha: float,
):
    if input_image is None:
        return None, None, "Upload an image to begin."

    image = Image.fromarray(input_image).convert("RGB") if not isinstance(input_image, Image.Image) else input_image

    head_idx = None if head_choice == "All (average)" else int(head_choice.replace("Head ", ""))

    if model_choice == "Both":
        models_to_run = ["Deformable DETR", "RT-DETR"]
    elif model_choice == "Compare all three":
        models_to_run = ["Deformable DETR", "RT-DETR", "DAB-DETR"]
    else:
        models_to_run = [model_choice]

    outputs: dict[str, dict] = {}
    for mname in models_to_run:
        try:
            outputs[mname] = run_model(image, mname)
        except Exception as e:
            return None, None, f"Error running {mname}: {e}"

    info_lines = []
    for mname, out in outputs.items():
        n_det = len(out["boxes"])
        n_layers = len(out["cross_attn_weights"])
        if out["cross_attn_weights"]:
            # Dense (DAB-DETR): shape (bs, n_heads, n_queries, seq_len) — heads at axis 1
            # Deformable: shape (bs, n_queries, n_heads, n_levels, n_points) — heads at axis 2
            w0 = out["cross_attn_weights"][0]
            n_heads = w0.shape[1] if "feat_hw" in out else w0.shape[2]
        else:
            n_heads = 0
        div = compute_head_diversity_score(out)
        info_lines.append(
            f"**{mname}** — {n_det} detections, {n_layers} decoder layers, "
            f"{n_heads} heads, head diversity: {div:.3f}"
        )
    # Clamp layer_idx
    max_layers = max(len(o["cross_attn_weights"]) for o in outputs.values()) if outputs else 6
    layer = max(-max_layers, min(int(layer_idx), max_layers - 1))

    # ── render ──
    vis_main = None
    vis_diversity = None

    if model_choice == "Both" and "Deformable DETR" in outputs and "RT-DETR" in outputs:
        vis_main = render_side_by_side(
            image,
            outputs["Deformable DETR"],
            outputs["RT-DETR"],
            layer_idx=layer,
            head_idx=head_idx,
            query_idx=query_idx,
            mode="offsets" if viz_mode == "Sampling Offsets" else
                 "rollout" if viz_mode == "Rollout" else "heatmap",
            cmap=cmap,
            alpha=alpha,
        )
        div_a = render_head_diversity(outputs["Deformable DETR"], layer_idx=layer, query_idx=query_idx)
        div_b = render_head_diversity(outputs["RT-DETR"], layer_idx=layer, query_idx=query_idx)
        total_w = div_a.width + div_b.width + 10
        total_h = max(div_a.height, div_b.height)
        panel = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        panel.paste(div_a, (0, 0))
        panel.paste(div_b, (div_a.width + 10, 0))
        vis_diversity = panel

    elif model_choice == "Compare all three" and len(outputs) >= 2:
        panels = []
        for mname in ["Deformable DETR", "RT-DETR", "DAB-DETR"]:
            if mname not in outputs:
                continue
            out = outputs[mname]
            p = render_heatmap(image, out, layer_idx=layer, head_idx=head_idx,
                               query_idx=query_idx, cmap=cmap, alpha=alpha)
            panels.append((mname, p))
        vis_main = _three_panel(panels)
        divs = [render_head_diversity(outputs[m], layer_idx=layer, query_idx=query_idx)
                for m in ["Deformable DETR", "RT-DETR", "DAB-DETR"] if m in outputs]
        total_w = sum(d.width for d in divs) + 10 * (len(divs) - 1)
        total_h = max(d.height for d in divs)
        panel = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        x = 0
        for d in divs:
            panel.paste(d, (x, 0))
            x += d.width + 10
        vis_diversity = panel

    else:
        mname = models_to_run[0]
        out = outputs[mname]

        if viz_mode == "Sampling Offsets" and "feat_hw" in out:
            vis_main = _dab_no_offsets_placeholder(image)
            info_lines.append(
                "\n> **Note:** Sampling offset visualization is not available for DAB-DETR "
                "(uses dense attention, not deformable attention)."
            )
        elif viz_mode == "Heatmap":
            vis_main = render_heatmap(image, out, layer_idx=layer, head_idx=head_idx,
                                      query_idx=query_idx, cmap=cmap, alpha=alpha)
        elif viz_mode == "Rollout":
            vis_main = render_rollout(image, out, head_idx=head_idx,
                                      query_idx=query_idx, alpha=alpha)
        elif viz_mode == "Sampling Offsets":
            vis_main = render_sampling_offsets(image, out, layer_idx=layer,
                                               head_idx=head_idx, query_idx=query_idx)
        elif viz_mode == "Head Diversity":
            vis_main = render_per_head_grid(image, out, layer_idx=layer, query_idx=query_idx)
        elif viz_mode == "Side-by-Side":
            vis_main = render_side_by_side(image, out, out,
                                           label_a=f"{mname} (layer {layer})",
                                           label_b=f"{mname} (rollout)",
                                           layer_idx=layer, head_idx=head_idx,
                                           query_idx=query_idx)

        vis_diversity = render_head_diversity(out, layer_idx=layer, query_idx=query_idx)

    info = "\n\n".join(info_lines)
    return (
        np.array(vis_main) if vis_main else None,
        np.array(vis_diversity) if vis_diversity else None,
        info,
    )


# ── UI helpers ───────────────────────────────────────────────────────────────

def _three_panel(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    """Render a 3-column comparison panel with labels."""
    from PIL import ImageDraw, ImageFont
    label_h = 28
    target_h = max(p.height for _, p in panels)
    imgs = []
    for name, p in panels:
        if p.height != target_h:
            ratio = target_h / p.height
            p = p.resize((int(p.width * ratio), target_h), Image.LANCZOS)
        imgs.append((name, p))

    total_w = sum(p.width for _, p in imgs) + 6 * (len(imgs) - 1)
    total_h = target_h + label_h
    panel = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(panel)
    x = 0
    for name, p in imgs:
        panel.paste(p, (x, label_h))
        draw.text((x + p.width // 2 - 50, 6), name, fill="white", font=font)
        x += p.width + 6
    return panel


def _dab_no_offsets_placeholder(image: Image.Image) -> Image.Image:
    """Return a plain image with an informational message."""
    from PIL import ImageDraw, ImageFont
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    msg = "Sampling offset visualization is not available for DAB-DETR\n(uses dense attention, not deformable)"
    draw.rectangle([10, 10, img.width - 10, 80], fill=(40, 40, 40))
    draw.text((18, 18), msg, fill="orange", font=font)
    return img


# ── build UI ─────────────────────────────────────────────────────────────────

def build_head_choices(n: int = 8) -> list[str]:
    return ["All (average)"] + [f"Head {i}" for i in range(n)]


EXAMPLE_IMAGES = sorted(
    [
        f"data/coco/val2017/{f}"
        for f in os.listdir("data/coco/val2017")
        if f.endswith(".jpg")
    ]
)[:10]


def build_app():
    with gr.Blocks(title="detr-lens", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# detr-lens\n"
            "Interactive attention visualization for **Deformable DETR**, **RT-DETR**, and **DAB-DETR**. "
            "Upload an image (or pick an example below), choose a model and visualization mode."
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input image", type="numpy", height=300)
                gr.Examples(
                    examples=EXAMPLE_IMAGES[:5],
                    inputs=input_image,
                    label="COCO examples",
                )

                model_choice = gr.Radio(
                    ["Deformable DETR", "RT-DETR", "DAB-DETR", "Both", "Compare all three"],
                    value="RT-DETR",
                    label="Model",
                )
                viz_mode = gr.Radio(
                    ["Heatmap", "Sampling Offsets", "Rollout", "Head Diversity", "Side-by-Side"],
                    value="Heatmap",
                    label="Visualization mode",
                )
                with gr.Accordion("Parameters", open=True):
                    layer_idx = gr.Slider(-6, -1, value=-1, step=1, label="Decoder layer (negative index)")
                    head_choice = gr.Dropdown(
                        build_head_choices(8), value="All (average)", label="Attention head"
                    )
                    query_idx = gr.Slider(0, 49, value=0, step=1, label="Object query index")
                    cmap = gr.Dropdown(
                        ["jet", "turbo", "viridis", "plasma", "hot", "coolwarm"],
                        value="jet", label="Colormap"
                    )
                    alpha = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Heatmap opacity")

                run_btn = gr.Button("Visualize", variant="primary")

            with gr.Column(scale=2):
                vis_out = gr.Image(label="Attention visualization", type="numpy")
                diversity_out = gr.Image(label="Head diversity heatmap", type="numpy")
                info_out = gr.Markdown()

        run_btn.click(
            fn=process,
            inputs=[input_image, model_choice, viz_mode, layer_idx,
                    head_choice, query_idx, cmap, alpha],
            outputs=[vis_out, diversity_out, info_out],
        )

        # Also trigger on image upload
        input_image.change(
            fn=process,
            inputs=[input_image, model_choice, viz_mode, layer_idx,
                    head_choice, query_idx, cmap, alpha],
            outputs=[vis_out, diversity_out, info_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
