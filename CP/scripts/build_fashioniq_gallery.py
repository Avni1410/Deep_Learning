"""
Phase 3 - build a cached CLIP embedding gallery for one FashionIQ category+split.

Gallery = the set of unique TARGET images appearing in the given CSV.
Reference/candidate images are deliberately excluded so a query can never
trivially retrieve its own reference image.

Usage:
  python scripts/build_fashioniq_gallery.py \
      --fiq-csv data/processed/fashioniq/val_queries_dress.csv \
      --category dress --split val
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/processed/fashioniq_embeddings")
    args = ap.parse_args()

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    print(f"Loaded {len(ds)} queries from {args.fiq_csv}")

    # NOTE: this decodes every reference + target image while scanning, even
    # though only targets are kept, because __getitem__ loads both together.
    # Fine for validation-sized CSVs; would need optimizing for train-sized ones.
    unique_targets: dict[str, Image.Image] = {}
    for sample in tqdm(ds, desc="Scanning for unique targets"):
        tid = sample["target"]
        if tid not in unique_targets:
            unique_targets[tid] = sample["target_image"]

    ids = list(unique_targets.keys())
    images = [unique_targets[i] for i in ids]
    print(f"Unique target images (gallery size): {len(ids)}")
    if not ids:
        raise SystemExit("No target images found - check --fiq-csv.")

    enc = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    print(f"Device: {enc.device}")

    all_embeds = []
    t0 = time.time()
    for start in tqdm(range(0, len(images), args.batch_size), desc="Encoding gallery"):
        batch = images[start:start + args.batch_size]
        out = enc.encode_image(batch)  # global_embed is already L2-normalized
        all_embeds.append(out.global_embed.detach().cpu())
    embeddings = torch.cat(all_embeds, dim=0)
    elapsed = time.time() - t0
    print(f"Encoded {len(ids)} images in {elapsed:.1f}s ({len(ids)/max(elapsed, 1e-9):.1f} img/s)")

    assert embeddings.shape == (len(ids), enc.embed_dim), \
        f"expected ({len(ids)}, {enc.embed_dim}), got {tuple(embeddings.shape)}"
    norms = embeddings.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), \
        f"gallery embeddings not unit-norm (min={norms.min():.4f}, max={norms.max():.4f})"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.category}_{args.split}_gallery.pt"
    torch.save(
        {
            "ids": ids,
            "embeddings": embeddings,       # [N, 512], float32, L2-normalized
            "category": args.category,
            "split": args.split,
            "model": args.model,
            "embed_dim": enc.embed_dim,
            "source_csv": str(args.fiq_csv),
        },
        out_path,
    )
    print(f"Saved gallery cache to {out_path}")


if __name__ == "__main__":
    main()