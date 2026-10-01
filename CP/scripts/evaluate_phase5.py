"""
Phase 5 evaluation - uses a Phase-5-specific gallery (see build_phase5_gallery.py)
since target embeddings now pass through visual_adapter.
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
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402
from src.retrieval.cosine_retrieval import rank_of_target, retrieve_top_k  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--phase5-checkpoint", type=Path, required=True)
    ap.add_argument("--variant", required=True, choices=["pair", "coral", "combined"])
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--gallery", type=Path, required=True, help="output of build_phase5_gallery.py")
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase5")
    args = ap.parse_args()

    gallery = torch.load(args.gallery, map_location="cpu", weights_only=False)
    gallery_ids, gallery_embeds = gallery["ids"], gallery["embeddings"]
    if gallery["category"] != args.category or gallery["variant"] != args.variant:
        raise SystemExit("Gallery category/variant does not match --category/--variant.")
    print(f"Loaded Phase 5 gallery: {args.category}/{args.variant} | {len(gallery_ids)} images")

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    phase4_model = AttributeAwareCIRModel(clip)
    load_phase4_checkpoint(phase4_model, args.phase4_checkpoint, clip.device)
    model = CrossDomainCIRModel(phase4_model).to(clip.device)

    ckpt = torch.load(args.phase5_checkpoint, map_location=clip.device, weights_only=False)
    model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
    model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
    model.visual_adapter.load_state_dict(ckpt["trainable_state"]["visual_adapter"])
    model.eval()
    print(f"Loaded Phase 5 [{args.variant}] checkpoint (epoch {ckpt['epoch']})")

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))

    results = []
    t0 = time.time()
    with torch.no_grad():
        for i in tqdm(range(n), desc=f"Retrieving (Phase 5 {args.variant})"):
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
    out_path = out_dir / f"{args.category}_{args.split}_phase5_{args.variant}_ranked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model, "variant": f"phase5_{args.variant}",
            "checkpoint": str(args.phase5_checkpoint),
            "category": args.category, "split": args.split,
            "gallery_size": len(gallery_ids), "num_queries": n, "queries": results,
        }, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()