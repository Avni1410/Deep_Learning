"""
Phase 2 - frozen pretrained CLIP ViT-B/16 representation layer.

Wraps the Hugging Face `transformers` CLIP implementation of the OpenAI
ViT-B/16 checkpoint and exposes:

  image: global embedding, CLS/patch hidden states, projected patch embeddings
  text : global embedding, token-level embeddings, attention mask

No training, fusion, cross-attention or retrieval logic lives here.

Normalization summary
  ImageOutput.global_embed  -> L2-normalized by default (normalize=True)
  ImageOutput.patch_embed   -> NOT normalized
  ImageOutput.*_hidden      -> NOT normalized (raw transformer states)
  TextOutput.global_embed   -> L2-normalized by default (normalize=True)
  TextOutput.token_embed    -> NOT normalized
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPModel

ImageInput = Union[Image.Image, torch.Tensor, Sequence[Union[Image.Image, torch.Tensor]]]
TextInput = Union[str, Sequence[str]]

_DTYPES = {"float32": torch.float32, "float16": torch.float16}


@dataclass(frozen=True)
class CLIPEncoderConfig:
    model_name: str = "openai/clip-vit-base-patch16"
    device: str = "auto"      # "auto" -> cuda if available else cpu
    dtype: str = "float32"    # "float32" or "float16" (float16 needs CUDA)
    freeze: bool = True       # Phase 2: always frozen
    max_text_length: int = 77


@dataclass
class ImageOutput:
    global_embed: torch.Tensor   # [B, 512]      CLS -> post-LN -> projection (normalized if requested)
    cls_hidden: torch.Tensor     # [B, 768]      raw CLS state (before post-LN / projection)
    patch_hidden: torch.Tensor   # [B, 196, 768] raw patch states
    patch_embed: torch.Tensor    # [B, 196, 512] patch states -> post-LN -> projection (not normalized)


@dataclass
class TextOutput:
    global_embed: torch.Tensor    # [B, 512]    EOS state -> projection (normalized if requested)
    token_embed: torch.Tensor     # [B, L, 512] every token -> projection (not normalized)
    token_hidden: torch.Tensor    # [B, L, 512] raw token states (before projection)
    attention_mask: torch.Tensor  # [B, L]      1 = real token, 0 = padding
    input_ids: torch.Tensor       # [B, L]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class CLIPEncoder(nn.Module):
    def __init__(self, config: CLIPEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or CLIPEncoderConfig()
        if self.config.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        device = _resolve_device(self.config.device)
        dtype = _DTYPES[self.config.dtype]
        if device.type == "cpu" and dtype == torch.float16:
            raise ValueError("float16 is only supported on CUDA; use float32 on CPU.")

        self.model = CLIPModel.from_pretrained(self.config.model_name, use_safetensors=True)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model.to(device=device, dtype=dtype)

        self._frozen = self.config.freeze
        if self._frozen:
            for p in self.model.parameters():
                p.requires_grad = False
        self.eval()

    # ------------------------------------------------------------------ utils
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def embed_dim(self) -> int:
        return self.model.config.projection_dim

    @property
    def vision_width(self) -> int:
        return self.model.config.vision_config.hidden_size

    @property
    def text_width(self) -> int:
        return self.model.config.text_config.hidden_size

    @property
    def num_patches(self) -> int:
        vc = self.model.config.vision_config
        return (vc.image_size // vc.patch_size) ** 2

    def describe(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "embed_dim": self.embed_dim,
            "vision_width": self.vision_width,
            "text_width": self.text_width,
            "num_patches": self.num_patches,
            "total_params": total,
            "trainable_params": trainable,
        }

    def train(self, mode: bool = True) -> "CLIPEncoder":
        # A frozen backbone must stay in eval mode (dropout off) even if a
        # parent module calls .train().
        super().train(mode)
        if getattr(self, "_frozen", False):
            self.model.eval()
        return self

    def _ctx(self):
        return torch.no_grad() if self._frozen else nullcontext()

    # ------------------------------------------------------------ preprocessing
    def preprocess_images(self, images: ImageInput) -> torch.Tensor:
        """PIL image / list of PIL images / already-preprocessed tensor -> [B,3,H,W] on device.

        Tensors are assumed to have been produced by this checkpoint's own
        preprocessing (they are NOT re-normalized).
        """
        if isinstance(images, torch.Tensor):
            x = images if images.ndim == 4 else images.unsqueeze(0)
        else:
            if isinstance(images, Image.Image):
                images = [images]
            images = list(images)
            if len(images) > 0 and isinstance(images[0], torch.Tensor):
                x = torch.stack(images)
            else:
                rgb = [im.convert("RGB") for im in images]
                x = self.image_processor(images=rgb, return_tensors="pt")["pixel_values"]
        return x.to(device=self.device, dtype=self.dtype)

    def tokenize(self, texts: TextInput) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        tok = self.tokenizer(
            list(texts),
            padding=True,               # pad to longest in batch, not always 77
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors="pt",
        )
        return tok["input_ids"].to(self.device), tok["attention_mask"].to(self.device)

    # ------------------------------------------------------------------ vision
    def project_visual_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """CLIP's own final LayerNorm + visual projection (768 -> 512)."""
        with self._ctx():
            return self.model.visual_projection(self.model.vision_model.post_layernorm(hidden))

    def encode_image(self, images: ImageInput, normalize: bool = True) -> ImageOutput:
        pixel_values = self.preprocess_images(images)
        with self._ctx():
            out = self.model.vision_model(pixel_values=pixel_values)
            hidden = out.last_hidden_state                     # [B, 1 + P, 768]
            cls_hidden = hidden[:, 0]                          # [B, 768]
            patch_hidden = hidden[:, 1:]                       # [B, P, 768]
            global_embed = self.model.visual_projection(out.pooler_output)  # [B, 512]
            patch_embed = self.project_visual_tokens(patch_hidden)          # [B, P, 512]
        if normalize:
            global_embed = F.normalize(global_embed, dim=-1)
        return ImageOutput(global_embed, cls_hidden, patch_hidden, patch_embed)

    def encode_image_global(self, images: ImageInput, normalize: bool = True) -> torch.Tensor:
        return self.encode_image(images, normalize=normalize).global_embed

    def encode_image_patches(self, images: ImageInput) -> torch.Tensor:
        return self.encode_image(images).patch_embed

    # -------------------------------------------------------------------- text
    def encode_text(self, texts: TextInput, normalize: bool = True) -> TextOutput:
        input_ids, attention_mask = self.tokenize(texts)
        with self._ctx():
            out = self.model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            token_hidden = out.last_hidden_state                              # [B, L, 512]
            global_embed = self.model.text_projection(out.pooler_output)      # [B, 512]
            token_embed = self.model.text_projection(token_hidden)            # [B, L, 512]
        if normalize:
            global_embed = F.normalize(global_embed, dim=-1)
        return TextOutput(global_embed, token_embed, token_hidden, attention_mask, input_ids)

    def encode_text_tokens(self, texts: TextInput) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encode_text(texts)
        return out.token_embed, out.attention_mask