"""
Deformable DETR wrapper with hook-based attention extraction.

Captures cross-attention weights and deformable sampling locations per decoder
layer without modifying model weights or architecture.

Attention weights shape: (bs, n_queries, n_heads, n_levels, n_points)
Sampling locations shape: (bs, n_queries, n_heads, n_levels, n_points, 2)
"""

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

CHECKPOINT = "SenseTime/deformable-detr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DeformableDETRWrapper:
    def __init__(self, checkpoint: str = CHECKPOINT, device: str = DEVICE):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.processor = None
        self._hooks: list = []
        self._cross_attn_weights: list[torch.Tensor] = []
        self._self_attn_weights: list[torch.Tensor] = []
        self._sampling_locations: list[torch.Tensor] = []

    def load(self):
        if self.model is not None:
            return
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.checkpoint,
            ignore_mismatched_sizes=True,
            attn_implementation="eager",
        )
        self.model.eval()
        self.model.to(self.device)
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()
        decoder = self._find_decoder(self.model)
        if decoder is None:
            return
        for layer in decoder.layers:
            if hasattr(layer, "encoder_attn"):
                # Pre-hook on the inner CUDA op to capture sampling_locations
                inner = getattr(layer.encoder_attn, "attn", None)
                if inner is not None:
                    h = inner.register_forward_pre_hook(self._hook_sampling_locs_pre)
                    self._hooks.append(h)
                h = layer.encoder_attn.register_forward_hook(self._hook_cross_attn)
                self._hooks.append(h)
            if hasattr(layer, "self_attn"):
                h = layer.self_attn.register_forward_hook(self._hook_self_attn)
                self._hooks.append(h)

    def _find_decoder(self, model):
        for name, module in model.named_modules():
            if "Decoder" in type(module).__name__ and hasattr(module, "layers"):
                return module
        return None

    def _hook_sampling_locs_pre(self, module, args):
        """Pre-hook on MSDeformAttention: args are (value, shapes, shapes_list, start_idx, sampling_locs, attn_weights, im2col_step)."""
        if len(args) >= 5 and args[4] is not None:
            # args[4] = sampling_locations shape: (bs, n_queries, n_heads, n_levels, n_points, 2)
            self._sampling_locations.append(args[4].detach().cpu())

    def _hook_cross_attn(self, module, inputs, output):
        """Outer deformable cross-attention: output = (attn_out, attn_weights)."""
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            # attn_weights: (bs, n_queries, n_heads, n_levels, n_points)
            self._cross_attn_weights.append(output[1].detach().cpu())

    def _hook_self_attn(self, module, inputs, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self._self_attn_weights.append(output[1].detach().cpu())

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _clear_cache(self):
        self._cross_attn_weights.clear()
        self._self_attn_weights.clear()
        self._sampling_locations.clear()

    @torch.no_grad()
    def forward(self, image: Image.Image) -> dict:
        """
        Run inference and return attention data.

        Returns dict:
          boxes, scores, labels, cross_attn_weights, self_attn_weights,
          sampling_offsets (sampling_locations), image_size.
        """
        self.load()
        self._clear_cache()

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        img_w, img_h = image.size

        outputs = self.model(**inputs, output_attentions=True)

        target_size = torch.tensor([[img_h, img_w]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.3, target_sizes=target_size
        )[0]

        cross_attn = self._cross_attn_weights.copy()
        self_attn = self._self_attn_weights.copy()
        sampling = self._sampling_locations.copy()

        # Fallback: model outputs fields
        if not cross_attn and hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
            cross_attn = [a.detach().cpu() for a in outputs.cross_attentions]
        if not self_attn and hasattr(outputs, "decoder_attentions") and outputs.decoder_attentions:
            self_attn = [a.detach().cpu() for a in outputs.decoder_attentions]

        return {
            "boxes": results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
            "cross_attn_weights": cross_attn,
            "self_attn_weights": self_attn,
            "sampling_offsets": sampling,
            "image_size": (img_h, img_w),
        }

    def __del__(self):
        self._remove_hooks()
