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
from viz.heatmap import render_heatmap, render_rollout, render_head_diversity, compute_head_diversity_score
from viz.offset_grid import render_sampling_offsets, render_per_head_grid
from viz.side_by_side import render_side_by_side

# ── global model singletons (loaded on first use) ────────────────────────────

_deformable_detr: Optional[DeformableDETRWrapper] = None
_rtdetr: Optional[RTDETRWrapper] = None


def get_model(name: str):
    global _deformable_detr, _rtdetr
    if name == "Deformable DETR":
        if _deformable_detr is None:
            _deformable_detr = DeformableDETRWrapper()
        _deformable_detr.load()
        return _deformable_detr
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

    models_to_run = (
        ["Deformable DETR", "RT-DETR"]
        if model_choice == "Both"
        else [model_choice]
    )

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
        n_heads = out["cross_attn_weights"][0].shape[2] if out["cross_attn_weights"] else 0
        div = compute_head_diversity_score(out)
        info_lines.append(
            f"**{mname}** — {n_det} detections, {n_layers} decoder layers, "
            f"{n_heads} heads, head diversity: {div:.3f}"
        )
    info = "\n\n".join(info_lines)

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
            mode="heatmap" if viz_mode in ("Heatmap", "Rollout") else "offsets"
            if viz_mode == "Sampling Offsets" else "rollout"
            if viz_mode == "Rollout" else "heatmap",
            cmap=cmap,
            alpha=alpha,
        )
        # Diversity side-by-side
        div_a = render_head_diversity(outputs["Deformable DETR"], layer_idx=layer, query_idx=query_idx)
        div_b = render_head_diversity(outputs["RT-DETR"], layer_idx=layer, query_idx=query_idx)
        total_w = div_a.width + div_b.width + 10
        total_h = max(div_a.height, div_b.height)
        panel = Image.new("RGB", (total_w, total_h), (20, 20, 20))
        panel.paste(div_a, (0, 0))
        panel.paste(div_b, (div_a.width + 10, 0))
        vis_diversity = panel

    else:
        mname = models_to_run[0]
        out = outputs[mname]

        if viz_mode == "Heatmap":
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

    return (
        np.array(vis_main) if vis_main else None,
        np.array(vis_diversity) if vis_diversity else None,
        info,
    )


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
            "Interactive attention visualization for **Deformable DETR** and **RT-DETR**. "
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
                    ["Deformable DETR", "RT-DETR", "Both"],
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
