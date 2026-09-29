"""
Phase 4 evaluation - reuses Phase 3's gallery caches (same protocol, same
frozen CLIP target embeddings) so Phase 3 vs Phase 4 numbers are comparable.

Usage:
  python scripts/evaluate_phase4.py \
    --checkpoint experiments/phase4/ablation_b_cross_attn_fusion/checkpoints/epoch_4.pt \
    --fiq-csv data/processed/fashioniq/val_queries_dress.csv \
    --gallery data/processed/fashioniq_embeddings/dress_val_gallery.pt \
    --category dress --split val
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.retrieval.cosine_retrieval import rank_of_target, retrieve_top_k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--gallery", type=Path, required=True)
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase4")
    args = ap.parse_args()

    gallery = torch.load(args.gallery, map_location="cpu", weights_only=False)
    gallery_ids: list[str] = gallery["ids"]
    gallery_embeds: torch.Tensor = gallery["embeddings"]
    if gallery["category"] != args.category or gallery["split"] != args.split:
        raise SystemExit("Gallery category/split does not match --category/--split.")
    print(f"Loaded gallery: {gallery['category']}/{gallery['split']} | {len(gallery_ids)} images")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip, attribute_vocab=ckpt.get("attribute_vocab")).to(clip.device)
    model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
    model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
    model.attribute_heads.load_state_dict(ckpt["trainable_state"]["attribute_heads"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt['epoch']})")

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Running {n} of {len(ds)} queries")

    results = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc="Retrieving (Phase 4)"):
            s = ds[i]
            out = model.encode_query([s["reference_image"]], [s["caption_1"]], [s["caption_2"]])
            q = out["query_embedding"].squeeze(0).cpu()
            full = retrieve_top_k(q, gallery_embeds, gallery_ids, k=None)
            rank = rank_of_target(full.ranked_ids, {s["target"]})
            results.append({
                "query_id": s["query_id"], "category": s["category"], "target": s["target"],
                "target_rank": rank,
                "top_k_ids": full.ranked_ids[:args.top_k],
                "top_k_scores": [float(x) for x in full.scores[:args.top_k].tolist()],
            })
    print(f"Done in {time.time() - t0:.1f}s")

    out_dir = args.out_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.category}_{args.split}_phase4_ranked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "checkpoint": str(args.checkpoint),
            "category": args.category, "split": args.split,
            "variant": "phase4_cross_attention_fusion",
            "gallery_size": len(gallery_ids), "num_queries": n, "queries": results,
        }, f, indent=2)
    print(f"Saved per-query results to {out_path}")
    print("Run scripts/evaluate_baseline.py on this file to get Recall@K "
          "(it reads the same 'queries'/'target_rank' schema).")


if __name__ == "__main__":
    main()