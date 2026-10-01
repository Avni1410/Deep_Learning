"""
Phase 5 - CrossDomainCIRModel.

Wraps a (warm-started) Phase 4 AttributeAwareCIRModel WITHOUT modifying it.
Adds ONE new trainable module, visual_adapter, applied identically to:
  - the reference image's global embedding g (feeds fusion, same as Phase 4)
  - the target image's global embedding z (used for L_CIR)
  - DeepFashion2 user/shop global embeddings (used for L_domain)

This is required, not decorative: CLIP is frozen, so aligning its raw output
directly (as a literal reading of spec Sec 10 would suggest) gives L_domain
zero gradient path to any trainable parameter, and would give it zero
influence on retrieval even if it did train something. Sharing one adapter
across all three call sites is what lets L_domain's gradient actually reach
parameters (visual_adapter) that also shape q and z, so the experiment
("does alignment help or hurt CIR?") is answerable at all.

visual_adapter is residual (x + adapter(x)), then re-normalized, so at
initialization it is close to the identity - this preserves the Phase 4
warm-start quality at step 0 rather than corrupting it with an untrained head.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attribute_aware_cir import AttributeAwareCIRModel, masked_mean_pool


class DomainProjectionHead(nn.Module):
    """512 -> 512, GELU, 512 -> 512, residual, L2-normalized. Per spec Sec 12."""

    def __init__(self, dim: int = 512, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x + self.net(x), dim=-1)


def load_phase4_checkpoint(model: AttributeAwareCIRModel, ckpt_path, device) -> tuple[int | None, dict | None]:
    """Format-agnostic loader: handles either the lean 'trainable_state' format
    or the original full 'model_state' format, whichever epoch_2.pt actually is.
    Inspects the file rather than assuming."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    if "trainable_state" in ckpt:
        ts = ckpt["trainable_state"]
        model.cross_attn.load_state_dict(ts["cross_attn"])
        model.fusion.load_state_dict(ts["fusion"])
        if "attribute_heads" in ts and len(ts["attribute_heads"]) > 0:
            model.attribute_heads.load_state_dict(ts["attribute_heads"])
        print(f"Loaded {ckpt_path}: lean 'trainable_state' format.")
    elif "model_state" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"Loaded {ckpt_path}: full 'model_state' format. "
              f"missing={len(missing)} unexpected={len(unexpected)}")
    else:
        raise KeyError(
            f"Checkpoint has neither 'trainable_state' nor 'model_state'. "
            f"Top-level keys: {list(ckpt.keys())}"
        )
    return ckpt.get("epoch"), ckpt.get("config")


class CrossDomainCIRModel(nn.Module):
    def __init__(self, phase4_model: AttributeAwareCIRModel, adapter_hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.clip = phase4_model.clip                 # frozen, shared with Phase 4
        self.cross_attn = phase4_model.cross_attn      # warm-started from Phase 4, trainable
        self.fusion = phase4_model.fusion              # warm-started from Phase 4, trainable
        dim = self.clip.embed_dim
        self.visual_adapter = DomainProjectionHead(dim, adapter_hidden, dropout)  # NEW, trainable

    def encode_visual(self, images) -> torch.Tensor:
        """Shared path used for target images AND DeepFashion2 user/shop images."""
        raw = self.clip.encode_image(images).global_embed
        return self.visual_adapter(raw)

    def encode_query(self, reference_images, caption_1, caption_2, debug: bool = False) -> dict:
        img_out = self.clip.encode_image(reference_images)
        t1 = self.clip.encode_text(caption_1)
        t2 = self.clip.encode_text(caption_2)

        g = self.visual_adapter(img_out.global_embed)   # ADAPTED, shared with domain loss
        patches = img_out.patch_embed
        tokens = torch.cat([t1.token_embed, t2.token_embed], dim=1)
        mask = torch.cat([t1.attention_mask, t2.attention_mask], dim=1)

        H = self.cross_attn(tokens, patches)
        h = masked_mean_pool(H, mask)
        q = self.fusion(g, h)

        out = {"query_embedding": q}
        if debug:
            out.update({"reference_global_raw": img_out.global_embed, "reference_global_adapted": g,
                        "cross_attended_tokens": H})
        return out

    def encode_target(self, target_images) -> torch.Tensor:
        return self.encode_visual(target_images)

    def forward(self, reference_images, caption_1, caption_2, target_images, debug: bool = False) -> dict:
        out = self.encode_query(reference_images, caption_1, caption_2, debug=debug)
        out["target_embedding"] = self.encode_target(target_images)
        return out