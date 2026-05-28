"""
DAB-DETR wrapper with hook-based attention extraction.

DAB-DETR uses standard dense multi-head cross-attention (not deformable):
  - Cross-attention shape: (bs, n_heads, n_queries, seq_len) where seq_len = H_feat * W_feat
  - No sampling offsets — offset grid visualization is not applicable
  - Spatial heatmaps: reshape seq_len → (H_feat, W_feat) directly, no Gaussian splatting

Checkpoint: IDEA-Research/dab-detr-resnet-50
"""

import math
import torch
from PIL import Image
from transformers import AutoImageProcessor, DabDetrForObjectDetection

CHECKPOINT = "IDEA-Research/dab-detr-resnet-50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DABDETRWrapper:
    def __init__(self, checkpoint: str = CHECKPOINT, device: str = DEVICE):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.processor = None
        self._hooks: list = []
        self._cross_attn_weights: list[torch.Tensor] = []
        self._self_attn_weights: list[torch.Tensor] = []
        self._feat_hw: tuple[int, int] | None = None

    def load(self):
        if self.model is not None:
            return
        self.processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = DabDetrForObjectDetection.from_pretrained(
            self.checkpoint,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()
        self.model.to(self.device)
        self._register_hooks()

    def _register_hooks(self):
        self._remove_hooks()
        inner = getattr(self.model, "model", None)
        decoder = getattr(inner, "decoder", None)
        if decoder is not None:
            for layer in decoder.layers:
                # DabDetrDecoderLayer has cross_attn (DabDetrDecoderLayerCrossAttention)
                if hasattr(layer, "cross_attn"):
                    h = layer.cross_attn.register_forward_hook(self._hook_cross_attn)
                    self._hooks.append(h)
                # and self_attn (DabDetrDecoderLayerSelfAttention)
                if hasattr(layer, "self_attn"):
                    h = layer.self_attn.register_forward_hook(self._hook_self_attn)
                    self._hooks.append(h)
        backbone = getattr(inner, "backbone", None)
        if backbone is not None:
            h = backbone.register_forward_hook(self._hook_backbone)
            self._hooks.append(h)

    def _hook_cross_attn(self, module, inputs, output):
        # DabDetrDecoderLayerCrossAttention → (hidden_states, cross_attn_weights)
        # cross_attn_weights: (bs, n_heads, n_queries, seq_len) or None
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self._cross_attn_weights.append(output[1].detach().cpu())

    def _hook_self_attn(self, module, inputs, output):
        # DabDetrDecoderLayerSelfAttention → (hidden_states, attn_weights)
        # attn_weights: (bs, n_heads, n_queries, n_queries) or None
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self._self_attn_weights.append(output[1].detach().cpu())

    def _hook_backbone(self, module, inputs, output):
        # DabDetrConvModel → (out, pos)
        # out[-1] = (feature_map, mask), feature_map: (bs, C, H_feat, W_feat)
        try:
            feat = output[0][-1][0]
            self._feat_hw = (int(feat.shape[2]), int(feat.shape[3]))
        except Exception:
            pass

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _clear_cache(self):
        self._cross_attn_weights.clear()
        self._self_attn_weights.clear()
        self._feat_hw = None

    @staticmethod
    def _infer_hw(seq_len: int) -> tuple[int, int]:
        """Factorize seq_len into (H, W) as close to square as possible."""
        h = int(math.isqrt(seq_len))
        while h > 1 and seq_len % h != 0:
            h -= 1
        return (h, seq_len // h)

    @torch.no_grad()
    def forward(self, image: Image.Image) -> dict:
        """
        Run inference and return attention data.

        Returns dict:
          boxes, scores, labels, cross_attn_weights, self_attn_weights,
          sampling_offsets (empty list), feat_hw, image_size.

        cross_attn_weights: list of (bs, n_heads, n_queries, seq_len) tensors,
          one per decoder layer.  Shape differs from deformable models.
        feat_hw: (H_feat, W_feat) — needed to reshape seq_len back to spatial grid.
        sampling_offsets: [] — DAB-DETR has no deformable sampling points.
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

        # Fallback: use model output fields if hooks captured nothing
        if not cross_attn and hasattr(outputs, "cross_attentions") and outputs.cross_attentions:
            cross_attn = [a.detach().cpu() for a in outputs.cross_attentions]
        if not self_attn and hasattr(outputs, "decoder_attentions") and outputs.decoder_attentions:
            self_attn = [a.detach().cpu() for a in outputs.decoder_attentions]

        feat_hw = self._feat_hw
        if feat_hw is None and cross_attn:
            feat_hw = self._infer_hw(cross_attn[0].shape[-1])

        return {
            "boxes": results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
            "cross_attn_weights": cross_attn,
            "self_attn_weights": self_attn,
            "sampling_offsets": [],
            "feat_hw": feat_hw,
            "image_size": (img_h, img_w),
        }

    def __del__(self):
        self._remove_hooks()
