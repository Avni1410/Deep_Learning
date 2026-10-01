"""
Phase 6 - build ONE combined, all-categories gallery (unlike Phase 3/4's
per-category galleries) so Strategy A (unrestricted similarity mining) can
actually surface cross-category candidates for Strategy B to then exclude.
Uses the frozen Phase 4 checkpoint's encoder (plain CLIP, no adapter -
Phase 6 does not use Phase 5's visual_adapter, per the decision to start
from Phase 4, not Phase 5).
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
    ap.add_argument("--fiq-csvs", type=Path, nargs="+", required=True,
                    help="e.g. train_queries_dress.csv train_queries_shirt.csv train_queries_toptee.csv")
    ap.add_argument("--categories", nargs="+", required=True, help="matching category name per CSV, same order")
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--split", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase6/cache")
    args = ap.parse_args()
    if len(args.fiq_csvs) != len(args.categories):
        raise SystemExit("--fiq-csvs and --categories must have the same length, in matching order.")

    all_ids: list[str] = []
    all_categories: list[str] = []
    all_images: list[Image.Image] = []
    seen_ids: set[str] = set()

    for csv_path, category in zip(args.fiq_csvs, args.categories):
        ds = FashionIQCIRDataset(csv_path=csv_path, image_root=args.fiq_root)
        print(f"Scanning {csv_path.name} ({category}): {len(ds)} queries")
        for sample in tqdm(ds, desc=f"  unique targets ({category})"):
            tid = sample["target"]
            if tid in seen_ids:
                continue  # dedupe across categories defensively, per spec Sec 11/34
            seen_ids.add(tid)
            all_ids.append(tid)
            all_categories.append(category)
            all_images.append(sample["target_image"])

    print(f"Combined gallery size (all categories, deduped): {len(all_ids)}")

    enc = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    all_embeds = []
    t0 = time.time()
    for start in tqdm(range(0, len(all_images), args.batch_size), desc="Encoding combined gallery"):
        batch = all_images[start:start + args.batch_size]
        out = enc.encode_image(batch)
        all_embeds.append(out.global_embed.detach().cpu())
    embeddings = torch.cat(all_embeds, dim=0)
    print(f"Encoded in {time.time() - t0:.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"combined_{args.split}_gallery.pt"
    torch.save({
        "ids": all_ids, "categories": all_categories, "embeddings": embeddings,
        "model": args.model, "split": args.split, "embed_dim": enc.embed_dim,
    }, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()