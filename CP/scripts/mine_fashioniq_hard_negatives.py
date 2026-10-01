"""
Phase 6 - offline hard-negative mining (Steps 5-11 of the dev order).

Produces a reproducible CSV per (category, split, strategy), traceable to
the exact checkpoint, K, N_hard, and seed used. Logs positive_similarity
and each negative's actual category alongside the mined IDs, so margin
analysis (spec Sec 30) and cross-category drift auditing (spec Sec 12) can
be done directly from the CSV without re-running retrieval.

FashionIQ image path convention (confirmed against the live
FashionIQCIRDataset._image_path implementation):
    data/raw/fashioniq/<category>/<ASIN>.jpg
"""
from __future__ import annotations

import argparse
import csv
import datetime
import random
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
from src.retrieval.hard_negative_miner import mine_for_query, random_negatives_for_query  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True, help="training CSV for ONE category")
    ap.add_argument("--category", required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True,
                    help="checkpoint used for mining AND Phase 6's starting point (Phase 4, per decision)")
    ap.add_argument("--combined-gallery", type=Path, required=True,
                    help="output of build_fashioniq_combined_gallery.py")
    ap.add_argument("--strategy", required=True, choices=["similarity", "category_similarity", "random"])
    ap.add_argument("--k-retrieval", type=int, default=50)
    ap.add_argument("--n-hard", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase6/hard_negatives")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Mining {n} of {len(ds)} queries | category={args.category} | strategy={args.strategy}")

    rows = []
    if args.strategy == "random":
        combined = torch.load(args.combined_gallery, map_location="cpu", weights_only=False)
        gallery_ids_all, gallery_categories_all = combined["ids"], combined["categories"]
        gallery_ids = [gid for gid, c in zip(gallery_ids_all, gallery_categories_all) if c == args.category]
        for i in tqdm(range(n), desc="Random negatives"):
            s = ds[i]
            m = random_negatives_for_query(s["query_id"], s["category"], s["target"], s["candidate"],
                                            gallery_ids, args.n_hard, rng,
                                            gallery_categories=None)  # gallery_ids is already single-category
            rows.append(m)
    else:
        clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
        model = AttributeAwareCIRModel(clip)
        epoch, _ = load_phase4_checkpoint(model, args.phase4_checkpoint, clip.device)
        model.to(clip.device)
        model.eval()
        print(f"Mining with Phase 4 checkpoint (epoch {epoch})")

        combined = torch.load(args.combined_gallery, map_location="cpu", weights_only=False)
        gallery_ids, gallery_categories, gallery_embeds = (
            combined["ids"], combined["categories"], combined["embeddings"]
        )

        with torch.no_grad():
            for i in tqdm(range(n), desc=f"Mining ({args.strategy})"):
                s = ds[i]
                q = model.encode_query([s["reference_image"]], [s["caption_1"]], [s["caption_2"]])["query_embedding"]
                q = q.squeeze(0).cpu()
                m = mine_for_query(
                    q, s["query_id"], s["category"], s["target"], s["candidate"],
                    gallery_ids, gallery_embeds, gallery_categories,
                    args.k_retrieval, args.n_hard, args.strategy,
                )
                rows.append(m)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.category}_train_{args.strategy}_negatives.csv"
    max_n = max((len(r.negative_ids) for r in rows), default=0)
    fieldnames = ["query_id", "category", "target", "candidate", "strategy", "k_retrieval",
                  "mining_checkpoint", "seed", "mined_at", "positive_similarity"] + \
                 [f"negative_{i+1}" for i in range(max_n)] + \
                 [f"similarity_{i+1}" for i in range(max_n)] + \
                 [f"negative_{i+1}_category" for i in range(max_n)]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row = {
                "query_id": r.query_id, "category": r.category, "target": r.target, "candidate": r.candidate,
                "strategy": r.strategy, "k_retrieval": r.k_retrieval,
                "mining_checkpoint": str(args.phase4_checkpoint), "seed": args.seed,
                "mined_at": datetime.datetime.utcnow().isoformat() + "Z",
                "positive_similarity": r.positive_similarity,
            }
            for i in range(max_n):
                row[f"negative_{i+1}"] = r.negative_ids[i] if i < len(r.negative_ids) else ""
                row[f"similarity_{i+1}"] = r.negative_similarities[i] if i < len(r.negative_similarities) else ""
                row[f"negative_{i+1}_category"] = r.negative_categories[i] if i < len(r.negative_categories) else ""
            writer.writerow(row)
    print(f"Saved {len(rows)} mined rows to {out_path}")


if __name__ == "__main__":
    main()