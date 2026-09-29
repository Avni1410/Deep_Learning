"""
Phase 4 - attribute-aware multimodal composition.

CLIP_ViT-B/16 (frozen, reused from Phase 2)
      |
      +--> reference: global g [B,512], patches V_p [B,196,512]
      |
caption_1, caption_2 --> CLIP text tokens T1 [B,L1,512], T2 [B,L2,512]
      |
      T = concat(T1, T2)  [B, L1+L2, 512]  (see docs for the positional-embedding caveat)
      |
      v
Cross-attention (text=Query, image patches=Key/Value), 2 blocks
      |
      v
H [B, L1+L2, 512] --masked mean pool (real tokens only)--> h [B, 512]
      |
Fusion([g; h; g*h; g-h]) -> MLP -> normalize -> q [B, 512]
      |
      +--> retrieval embedding q
      +--> attribute heads (added in Part B, once docs/attribute_mapping.md is available)

Target images use the SAME frozen CLIP visual encoder -> z [B, 512], no target-side
cross-attention (per spec: shared weights, no separate target network).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.clip_encoder import CLIPEncoder


class CrossAttentionBlock(nn.Module):
    """Text-as-query, image-patches-as-key/value Transformer block.

    Image patches have no padding (always 196 tokens), so no key_padding_mask
    is needed on the attention call itself. Padded TEXT query positions still
    produce an (unused) output row; they must be excluded later by masked_mean_pool,
    not here.
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, text_tokens: torch.Tensor, image_patches: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query=text_tokens, key=image_patches, value=image_patches, need_weights=False)
        x = self.norm1(text_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttentionStack(nn.Module):
    def __init__(self, dim: int = 512, num_heads: int = 8, num_blocks: int = 2,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(dim, num_heads, ffn_dim, dropout) for _ in range(num_blocks)]
        )

    def forward(self, text_tokens: torch.Tensor, image_patches: torch.Tensor) -> torch.Tensor:
        x = text_tokens
        for block in self.blocks:
            x = block(x, image_patches)
        return x


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """hidden: [B, L, D]; mask: [B, L] with 1 = real token, 0 = padding."""
    m = mask.unsqueeze(-1).to(hidden.dtype)          # [B, L, 1]
    summed = (hidden * m).sum(dim=1)                 # [B, D]
    count = m.sum(dim=1).clamp(min=1e-6)              # [B, 1]
    return summed / count


class FusionMLP(nn.Module):
    """x = [g; h; g*h; g-h] (2048-d) -> 1024 -> 512, GELU, dropout, L2-normalized output."""

    def __init__(self, dim: int = 512, hidden: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, g: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        x = torch.cat([g, h, g * h, g - h], dim=-1)
        q = self.net(x)
        return F.normalize(q, dim=-1)


class AttributeAwareCIRModel(nn.Module):
    """
    attribute_vocab: dict[attribute_name -> num_classes]. None or {} means no
    attribute heads exist yet (Part A / ablation B). Passing a real vocabulary
    (from docs/attribute_mapping.md) adds heads for ablation C.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        num_heads: int = 8,
        num_blocks: int = 2,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        attribute_vocab: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if not clip_encoder.config.freeze:
            raise ValueError(
                "AttributeAwareCIRModel requires a frozen CLIPEncoder "
                "(CLIPEncoderConfig(freeze=True)); Phase 4 does not fine-tune CLIP yet."
            )
        self.clip = clip_encoder
        dim = clip_encoder.embed_dim
        self.cross_attn = CrossAttentionStack(dim, num_heads, num_blocks, ffn_dim, dropout)
        self.fusion = FusionMLP(dim, hidden=1024, dropout=dropout)
        self.attribute_heads = nn.ModuleDict(
            {name: nn.Linear(dim, n_classes) for name, n_classes in (attribute_vocab or {}).items()}
        )

    def encode_query(self, reference_images, caption_1, caption_2, debug: bool = False) -> dict:
        img_out = self.clip.encode_image(reference_images)   # frozen internally
        t1 = self.clip.encode_text(caption_1)
        t2 = self.clip.encode_text(caption_2)

        g = img_out.global_embed                              # [B, 512], L2-normalized
        patches = img_out.patch_embed                         # [B, 196, 512]
        tokens = torch.cat([t1.token_embed, t2.token_embed], dim=1)          # [B, L1+L2, 512]
        mask = torch.cat([t1.attention_mask, t2.attention_mask], dim=1)     # [B, L1+L2]

        H = self.cross_attn(tokens, patches)
        h = masked_mean_pool(H, mask)
        q = self.fusion(g, h)

        attribute_logits = {name: head(q) for name, head in self.attribute_heads.items()}

        out = {"query_embedding": q, "attribute_logits": attribute_logits}
        if debug:
            out.update({
                "reference_global": g, "reference_patches": patches,
                "text_tokens": tokens, "text_mask": mask, "cross_attended_tokens": H,
            })
        return out

    def encode_target(self, target_images) -> torch.Tensor:
        return self.clip.encode_image(target_images).global_embed  # [B, 512], L2-normalized

    def forward(self, reference_images, caption_1, caption_2, target_images, debug: bool = False) -> dict:
        out = self.encode_query(reference_images, caption_1, caption_2, debug=debug)
        out["target_embedding"] = self.encode_target(target_images)
        return out