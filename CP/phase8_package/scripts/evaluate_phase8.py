"""
Phase 8 evaluation - IDENTICAL protocol to Phase 3/4: same galleries
(data/processed/fashioniq_embeddings/<category>_val_gallery.pt, built by
scripts/build_fashioniq_gallery.py), same AttributeAwareCIRModel class, no
visual_adapter (Phase 8 does not use Phase 5's cross-domain alignment).
Only cross_attn/fusion weights differ from Phase 4's checkpoint.

Output JSON schema matches scripts/evaluate_baseline.py's expected input,
so Recall@K is computed with the exact same scorer used for every prior
phase.
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
from src.models.composed_retrieval_model import load_phase4_checkpoint  # noqa: E402
from src.retrieval.cosine_retrieval import rank_of_target, retrieve_top_k  # noqa: E402
from src.utils.checkpoint_compat import patch_path_compat  # noqa: E402

patch_path_compat()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase8-checkpoint", type=Path, required=True)
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--gallery", type=Path, required=True, help="existing Phase 3/4 *_val_gallery.pt")
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase8/results")
    args = ap.parse_args()

    gallery = torch.load(args.gallery, map_location="cpu", weights_only=False)
    gallery_ids, gallery_embeds = gallery["ids"], gallery["embeddings"]
    if gallery["category"] != args.category:
        raise SystemExit(f"Gallery is for {gallery['category']}, but --category is {args.category}.")
    print(f"Loaded gallery: {args.category}/{args.split} | {len(gallery_ids)} images")

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)

    ckpt = torch.load(args.phase8_checkpoint, map_location=clip.device, weights_only=False)
    model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
    model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
    if ckpt["trainable_state"].get("attribute_heads"):
        model.attribute_heads.load_state_dict(ckpt["trainable_state"]["attribute_heads"])
    model.to(clip.device)
    model.eval()
    print(f"Loaded Phase 8 checkpoint (round {ckpt.get('round')}): {args.phase8_checkpoint}")

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    results = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"Retrieving (Phase 8, {args.category})"):
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.category}_{args.split}_phase8_ranked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "variant": "phase8_semihard",
            "checkpoint": str(args.phase8_checkpoint),
            "category": args.category, "split": args.split,
            "gallery_size": len(gallery_ids), "num_queries": n, "queries": results,
        }, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
