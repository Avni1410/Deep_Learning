"""
Phase 2 verification for src.models.clip_encoder.CLIPEncoder.

Run from the repository root, e.g.:
  python scripts/test_clip_encoder.py --fiq-csv <path to a cleaned FashionIQ CSV>

Sanity cosines printed here are NOT retrieval performance.
"""
from __future__ import annotations

import argparse
import inspect
import platform
from random import sample
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from PIL import Image
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402

REF_KEYS = ("reference_image", "ref_image", "source_image", "candidate_image", "reference", "candidate")
TGT_KEYS = ("target_image", "target")
CAP_KEYS = ("caption", "captions", "modification", "text", "modification_text")
USER_KEYS = ("user_image", "consumer_image", "user")
SHOP_KEYS = ("shop_image", "commercial_image", "shop")


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.checks: list[tuple[str, bool, str]] = []

    def log(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, bool(ok), detail))
        self.log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def find_field(sample, names, kind):
    if not isinstance(sample, dict):
        raise SystemExit(
            f"Dataset returned {type(sample).__name__}, expected a dict. "
            "Paste the loader's __getitem__ and I will adapt the test."
        )
    for n in names:
        if n not in sample:
            continue
        v = sample[n]
        if kind == "image" and isinstance(v, (Image.Image, torch.Tensor)):
            return v
        if kind == "text" and isinstance(v, str):
            return v
        if kind == "text" and isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
            return " ".join(v)
    raise SystemExit(f"No {kind} field found among {names}. Sample keys: {list(sample.keys())}")


def build_dataset(cls, rules):
    """Fill constructor args by matching parameter names against simple keyword rules."""
    sig = inspect.signature(cls.__init__)
    print(f"{cls.__name__}{sig}")
    kwargs = {}
    for name, p in sig.parameters.items():
        if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        low = name.lower()
        for keys, value in rules:
            if value is not None and any(k in low for k in keys):
                kwargs[name] = value
                break
    missing = [
        n for n, p in sig.parameters.items()
        if n != "self" and p.default is p.empty and n not in kwargs
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if missing:
        raise SystemExit(f"Could not fill {cls.__name__} arguments {missing}. Paste the loader file.")
    print(f"  built with: {kwargs}")
    return cls(**kwargs)


def collate_list(batch):
    return batch


def finite(*tensors) -> bool:
    return all(bool(torch.isfinite(t).all()) for t in tensors)


def maxdiff(a, b) -> float:
    return float((a.float() - b.float()).abs().max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--split", default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--skip-df2", action="store_true")
    ap.add_argument("--df2-csv", type=Path,
                    default=REPO_ROOT / "data/processed/deepfashion2/domain_alignment_pairs_5000.csv")
    ap.add_argument("--df2-root", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/images")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/phase_2_verified_output.md",
                    help="where to write measured results ('' to disable)")
    args = ap.parse_args()

    rep = Report()
    rep.log("=" * 50)
    rep.log("Phase 2 CLIP Encoder Test")
    rep.log("=" * 50)
    rep.log(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")

    # ---- A. load ----------------------------------------------------------
    t0 = time.time()
    enc = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    info = enc.describe()
    rep.log(f"Device: {info['device']}" + (f" ({torch.cuda.get_device_name(0)})" if enc.device.type == "cuda" else ""))
    rep.log(f"Model: {info['model_name']} | load time {time.time() - t0:.1f}s")
    rep.log(f"embed_dim={info['embed_dim']} vision_width={info['vision_width']} "
            f"text_width={info['text_width']} num_patches={info['num_patches']}")
    rep.log(f"params total={info['total_params']:,} trainable={info['trainable_params']:,}")
    rep.check("CLIP frozen (0 trainable params)", info["trainable_params"] == 0)
    rep.check("Backbone in eval mode", not enc.model.training)
    rep.check("Matches project spec (196 patches, 768 width, 512 embed)",
              (info["num_patches"], info["vision_width"], info["embed_dim"]) == (196, 768, 512))
    if enc.device.type == "cuda":
        rep.log(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MiB")

    # ---- B. single FashionIQ example ---------------------------------------
    rep.log("\n--- FashionIQ single example ---")
    fiq_rules = [
        (("csv", "manifest", "annotation"), args.fiq_csv),
        (("root", "image_dir", "img_dir", "images"), args.fiq_root),
        (("split",), args.split),
    ]
    ds = build_dataset(FashionIQCIRDataset, fiq_rules)
    rep.log(f"Dataset length: {len(ds)}")
    s0, s1 = ds[args.index], ds[args.index + 1]
    rep.log(f"Sample keys: { {k: type(v).__name__ for k, v in s0.items()} }")
    for k, v in s0.items():
        if isinstance(v, str) and len(v) < 80:
            rep.log(f"  {k}: {v}")

    ref_img = find_field(s0, REF_KEYS, "image")
    tgt_img = find_field(s0, TGT_KEYS, "image")
    def build_caption(sample: dict) -> str:
        if "caption_1" in sample and "caption_2" in sample:
            return f"{sample['caption_1']}, {sample['caption_2']}"
        return find_field(sample, CAP_KEYS, "text")

    caption = build_caption(s0)
    other_tgt = find_field(s1, TGT_KEYS, "image")
    if isinstance(ref_img, torch.Tensor):
        rep.log("  WARNING: loader returns tensors; they must come from CLIP's own preprocessing.")
    rep.log(f"Caption: {caption}")

    ref, tgt, oth = enc.encode_image(ref_img), enc.encode_image(tgt_img), enc.encode_image(other_tgt)
    txt = enc.encode_text(caption)

    rep.log(f"Reference global embedding : {list(ref.global_embed.shape)}")
    rep.log(f"Reference CLS hidden       : {list(ref.cls_hidden.shape)}")
    rep.log(f"Reference patch hidden     : {list(ref.patch_hidden.shape)}")
    rep.log(f"Reference patch embeddings : {list(ref.patch_embed.shape)}")
    rep.log(f"Target global embedding    : {list(tgt.global_embed.shape)}")
    rep.log(f"Text global embedding      : {list(txt.global_embed.shape)}")
    rep.log(f"Text token embeddings      : {list(txt.token_embed.shape)} "
            f"(real tokens: {int(txt.attention_mask.sum())})")

    D, P, W = enc.embed_dim, enc.num_patches, enc.vision_width
    rep.check("Global shape [1, D]", list(ref.global_embed.shape) == [1, D])
    rep.check("Patch embed shape [1, P, D]", list(ref.patch_embed.shape) == [1, P, D])
    rep.check("Patch hidden shape [1, P, W]", list(ref.patch_hidden.shape) == [1, P, W])
    rep.check("Text tokens shape [1, L, D]",
              txt.token_embed.ndim == 3 and txt.token_embed.shape[0] == 1 and txt.token_embed.shape[-1] == D)
    rep.check("Finite: reference", finite(ref.global_embed, ref.patch_embed, ref.patch_hidden, ref.cls_hidden))
    rep.check("Finite: target", finite(tgt.global_embed, tgt.patch_embed))
    rep.check("Finite: text", finite(txt.global_embed, txt.token_embed))

    tol = 1e-3
    rn, tn, xn = (ref.global_embed.norm(dim=-1).item(), tgt.global_embed.norm(dim=-1).item(),
                  txt.global_embed.norm(dim=-1).item())
    rep.log(f"Norms: reference={rn:.6f} target={tn:.6f} text={xn:.6f}; "
            f"mean patch-embed norm={ref.patch_embed.norm(dim=-1).mean().item():.3f} (not normalized)")
    rep.check("Global embeddings L2-normalized", all(abs(n - 1) < tol for n in (rn, tn, xn)))
    rep.check("No autograd graph on outputs", not ref.global_embed.requires_grad and not txt.token_embed.requires_grad)

    recomputed = F.normalize(enc.project_visual_tokens(ref.cls_hidden), dim=-1)
    d = maxdiff(recomputed, ref.global_embed)
    rep.check("Manual CLS projection matches library pooled path", d < 1e-4, f"max diff {d:.2e}")

    ref2, txt2 = enc.encode_image(ref_img), enc.encode_text(caption)
    d_img, d_txt = maxdiff(ref.global_embed, ref2.global_embed), maxdiff(txt.token_embed, txt2.token_embed)
    rep.check("Deterministic (image)", d_img <= 1e-5, f"max diff {d_img:.2e}")
    rep.check("Deterministic (text)", d_txt <= 1e-5, f"max diff {d_txt:.2e}")

    rep.log("Sanity cosines (NOT retrieval performance):")
    rep.log(f"  reference vs target        : {(ref.global_embed @ tgt.global_embed.T).item():.4f}")
    rep.log(f"  reference vs other target  : {(ref.global_embed @ oth.global_embed.T).item():.4f}")
    rep.log(f"  text vs reference          : {(txt.global_embed @ ref.global_embed.T).item():.4f}")
    rep.log(f"  text vs target             : {(txt.global_embed @ tgt.global_embed.T).item():.4f}")

    # ---- C. batch -----------------------------------------------------------
    rep.log(f"\n--- FashionIQ batch (B={args.batch_size}) ---")
    if enc.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=collate_list)))
    refs = [find_field(s, REF_KEYS, "image") for s in batch]
    tgts = [find_field(s, TGT_KEYS, "image") for s in batch]
    caps = [build_caption(s) for s in batch]
    B = len(batch)
    rb, tb, xb = enc.encode_image(refs), enc.encode_image(tgts), enc.encode_text(caps)
    rep.log(f"reference global   : {list(rb.global_embed.shape)}")
    rep.log(f"target global      : {list(tb.global_embed.shape)}")
    rep.log(f"reference patches  : {list(rb.patch_embed.shape)}")
    rep.log(f"text tokens        : {list(xb.token_embed.shape)}  mask {list(xb.attention_mask.shape)}")
    rep.check("Batch shapes correct",
              list(rb.global_embed.shape) == [B, D] and list(tb.global_embed.shape) == [B, D]
              and list(rb.patch_embed.shape) == [B, P, D]
              and xb.token_embed.shape[0] == B and xb.token_embed.shape[-1] == D)
    rep.check("Batch outputs finite", finite(rb.global_embed, rb.patch_embed, tb.global_embed, xb.token_embed))
    single0 = enc.encode_image(refs[0]).global_embed
    d = maxdiff(single0, rb.global_embed[:1])
    rep.check("Batch element matches single encode", d < 1e-3, f"max diff {d:.2e}")
    if enc.device.type == "cuda":
        rep.log(f"Peak GPU memory during batch: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MiB")

    # ---- D. DeepFashion2 ----------------------------------------------------
    if not args.skip_df2:
        rep.log("\n--- DeepFashion2 compatibility ---")
        try:
            from src.data.deepfashion2_dataset import DeepFashion2DomainDataset
            df2_rules = [
                (("csv", "manifest", "pairs"), args.df2_csv),
                (("root", "image_dir", "img_dir", "images"), args.df2_root),
            ]
            ds2 = build_dataset(DeepFashion2DomainDataset, df2_rules)
            u, s = ds2[0], ds2[1]
            user_imgs = [find_field(u, USER_KEYS, "image"), find_field(s, USER_KEYS, "image")]
            shop_imgs = [find_field(u, SHOP_KEYS, "image"), find_field(s, SHOP_KEYS, "image")]
            uo, so = enc.encode_image(user_imgs), enc.encode_image(shop_imgs)
            rep.log(f"user global {list(uo.global_embed.shape)} | shop global {list(so.global_embed.shape)}")
            rep.log(f"user patches {list(uo.patch_embed.shape)} | shop patches {list(so.patch_embed.shape)}")
            rep.log(f"pair cosines (sanity only): {(uo.global_embed * so.global_embed).sum(-1).tolist()}")
            rep.check("DeepFashion2 user/shop encoded with same encoder",
                      finite(uo.global_embed, so.global_embed)
                      and list(uo.patch_embed.shape) == [2, P, D] and list(so.patch_embed.shape) == [2, P, D])
        except BaseException as e:  # report, never hide
            rep.check("DeepFashion2 compatibility", False, repr(e))

    # ---- summary ------------------------------------------------------------
    n_fail = sum(1 for _, ok, _ in rep.checks if not ok)
    rep.log("\n" + "=" * 50)
    rep.log(f"{len(rep.checks) - n_fail}/{len(rep.checks)} checks passed")
    rep.log("=" * 50)

    if str(args.report):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            "# Phase 2 - measured output\n\nGenerated by `scripts/test_clip_encoder.py`; "
            "nothing here is hand-written.\n\n```text\n" + "\n".join(rep.lines) + "\n```\n",
            encoding="utf-8",
        )
        print(f"Report written to {args.report}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()