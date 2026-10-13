"""
Phase 8 - semi-hard offline hard-negative mining.

Differs from Phase 6's scripts/mine_fashioniq_hard_negatives.py in exactly
two ways, both deliberate and documented:
  1. Negatives must have similarity STRICTLY BELOW the positive's similarity
     at mining time (semi-hard), not simply top-K regardless of rank.
  2. The duplicate-group lookup CSV is loaded ONCE, not per query.

Everything else (duplicate exclusion, category/strategy logic, gallery
format) is reused as-is from the existing Phase 6 implementation.

Usage:
  python scripts/mine_phase8_hard_negatives.py \
      --fiq-csv data/processed/fashioniq/train_queries_dress.csv --category dress \
      --checkpoint experiments/phase4/phase4_deliverable/checkpoints/epoch_2.pt \
      --combined-gallery experiments/phase6/cache/combined_train_gallery.pt \
      --strategy similarity --n-hard 3 --out-dir experiments/phase8/hard_negatives
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import load_phase4_checkpoint  # noqa: E402
from src.retrieval.hard_negative_miner import load_duplicate_lookup  # noqa: E402
from src.retrieval.semi_hard_miner import mine_semi_hard_for_query  # noqa: E402
from src.utils.checkpoint_compat import patch_path_compat  # noqa: E402

patch_path_compat()


def mine_category(
    model, fiq_csv: Path, fiq_root: Path, category: str,
    combined_gallery_path: Path, strategy: str, k_retrieval: int, n_hard: int,
    margin_buffer: float, limit: int | None, duplicate_lookup_path: Path,
    checkpoint_path: Path, seed: int,
) -> list:
    ds = FashionIQCIRDataset(csv_path=fiq_csv, image_root=fiq_root)
    n = len(ds) if limit is None else min(limit, len(ds))
    print(f"Mining {n} of {len(ds)} queries | category={category} | strategy={strategy} | semi-hard")

    combined = torch.load(combined_gallery_path, map_location="cpu", weights_only=False)
    gallery_ids, gallery_categories, gallery_embeds = (
        combined["ids"], combined["categories"], combined["embeddings"]
    )
    asin_to_group, group_to_asins = load_duplicate_lookup(duplicate_lookup_path)

    rows = []
    n_shortfall = 0
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"Mining {category} (semi-hard)"):
            s = ds[i]
            q = model.encode_query([s["reference_image"]], [s["caption_1"]], [s["caption_2"]])["query_embedding"]
            q = q.squeeze(0).cpu()
            m = mine_semi_hard_for_query(
                q, s["query_id"], s["category"], s["target"], s["candidate"],
                gallery_ids, gallery_embeds, gallery_categories,
                asin_to_group, group_to_asins,
                k_retrieval, n_hard, strategy, margin_buffer,
            )
            if m.shortfall > 0:
                n_shortfall += 1
            rows.append((m, checkpoint_path, seed))
    if n_shortfall:
        print(f"  NOTE: {n_shortfall}/{n} queries had fewer than {n_hard} semi-hard candidates available "
              f"(too few gallery items rank below the positive) - these rows have shorter negative lists, "
              f"not padded with anything invalid.")
    return rows


def write_csv(rows: list, out_path: Path) -> None:
    max_n = max((len(m.negative_ids) for m, _, _ in rows), default=0)
    fieldnames = ["query_id", "category", "target", "candidate", "strategy", "k_retrieval",
                  "mining_checkpoint", "seed", "mined_at", "positive_similarity",
                  "n_requested", "n_found", "shortfall"] + \
                 [f"negative_{i+1}" for i in range(max_n)] + \
                 [f"similarity_{i+1}" for i in range(max_n)] + \
                 [f"negative_{i+1}_category" for i in range(max_n)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m, ckpt, seed in rows:
            row = {
                "query_id": m.query_id, "category": m.category, "target": m.target, "candidate": m.candidate,
                "strategy": m.strategy, "k_retrieval": m.k_retrieval,
                "mining_checkpoint": str(ckpt), "seed": seed,
                "mined_at": datetime.datetime.utcnow().isoformat() + "Z",
                "positive_similarity": m.positive_similarity,
                "n_requested": m.n_requested, "n_found": m.n_found, "shortfall": m.shortfall,
            }
            for i in range(max_n):
                row[f"negative_{i+1}"] = m.negative_ids[i] if i < len(m.negative_ids) else ""
                row[f"similarity_{i+1}"] = m.negative_similarities[i] if i < len(m.negative_similarities) else ""
                row[f"negative_{i+1}_category"] = m.negative_categories[i] if i < len(m.negative_categories) else ""
            writer.writerow(row)
    print(f"Saved {len(rows)} mined rows to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="checkpoint to mine from (Phase 4 initially, then the latest Phase 8 checkpoint "
                         "on subsequent re-mining rounds)")
    ap.add_argument("--combined-gallery", type=Path, required=True)
    ap.add_argument("--duplicate-lookup", type=Path,
                    default=REPO_ROOT / "experiments/phase6/fashioniq_duplicate_lookup.csv",
                    help="reused from Phase 6 - not regenerated")
    ap.add_argument("--strategy", default="similarity", choices=["similarity", "category_similarity"])
    ap.add_argument("--k-retrieval", type=int, default=50)
    ap.add_argument("--n-hard", type=int, default=3)
    ap.add_argument("--margin-buffer", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase8/hard_negatives")
    args = ap.parse_args()

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)
    epoch, _ = load_phase4_checkpoint(model, args.checkpoint, clip.device)
    model.to(clip.device)
    model.eval()
    print(f"Mining with checkpoint (epoch field={epoch}): {args.checkpoint}")

    rows = mine_category(
        model, args.fiq_csv, args.fiq_root, args.category, args.combined_gallery,
        args.strategy, args.k_retrieval, args.n_hard, args.margin_buffer, args.limit,
        args.duplicate_lookup, args.checkpoint, args.seed,
    )
    out_path = args.out_dir / f"{args.category}_train_{args.strategy}_semihard_negatives.csv"
    write_csv(rows, out_path)


if __name__ == "__main__":
    main()
