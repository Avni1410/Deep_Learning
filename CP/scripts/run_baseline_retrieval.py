"""
Phase 3 - run baseline composed-image-retrieval over FashionIQ.

Composition (BASELINE-A, the main baseline):
  r  = normalize(CLIP_image(reference))
  t1 = normalize(CLIP_text(caption_1))
  t2 = normalize(CLIP_text(caption_2))
  t  = normalize((t1 + t2) / 2)
  q  = normalize(r + t)

Diagnostic variants (--variant caption1_only / caption2_only) drop one
caption; they are NOT the main baseline (see project instructions).

Retrieval: score_i = q . z_i over a precomputed, L2-normalized gallery.

This script only produces per-query ranks and top-k results. Run
scripts/evaluate_baseline.py afterwards to turn this into Recall@K.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.retrieval.cosine_retrieval import rank_of_target, retrieve_top_k  # noqa: E402

VARIANTS = {
    "mean": lambda t1, t2: (t1 + t2) / 2,   # BASELINE-A (main baseline)
    "caption1_only": lambda t1, t2: t1,     # diagnostic Variant B
    "caption2_only": lambda t1, t2: t2,     # diagnostic Variant C
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--gallery", type=Path, required=True,
                    help="path to a *_gallery.pt produced by build_fashioniq_gallery.py")
    ap.add_argument("--category", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--variant", default="mean", choices=list(VARIANTS.keys()))
    ap.add_argument("--limit", type=int, default=None, help="only run the first N queries (sanity runs)")
    ap.add_argument("--top-k", type=int, default=50, help="how many top results to store per query")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/baseline")
    args = ap.parse_args()

    gallery = torch.load(args.gallery, map_location="cpu", weights_only=False)
    gallery_ids: list[str] = gallery["ids"]
    gallery_embeds: torch.Tensor = gallery["embeddings"]
    if gallery["category"] != args.category or gallery["split"] != args.split:
        raise SystemExit(
            f"Gallery is for {gallery['category']}/{gallery['split']}, "
            f"but --category/--split is {args.category}/{args.split}."
        )
    print(f"Loaded gallery: {gallery['category']}/{gallery['split']} | "
          f"{len(gallery_ids)} images | model={gallery['model']}")

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"Running {n} of {len(ds)} queries | variant={args.variant} | top_k={args.top_k}")

    enc = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    combine = VARIANTS[args.variant]

    results = []
    missing_target_in_gallery = 0
    t0 = time.time()
    for i in tqdm(range(n), desc="Retrieving"):
        s = ds[i]
        r = enc.encode_image(s["reference_image"]).global_embed   # [1, 512]
        t1 = enc.encode_text(s["caption_1"]).global_embed         # [1, 512]
        t2 = enc.encode_text(s["caption_2"]).global_embed         # [1, 512]
        t = F.normalize(combine(t1, t2), dim=-1)
        q = F.normalize(r + t, dim=-1).squeeze(0).cpu()           # [512]

        full = retrieve_top_k(q, gallery_embeds, gallery_ids, k=None)
        rank = rank_of_target(full.ranked_ids, {s["target"]})
        if rank is None:
            missing_target_in_gallery += 1

        results.append({
            "query_id": s["query_id"],
            "category": s["category"],
            "candidate": s["candidate"],
            "target": s["target"],
            "caption_1": s["caption_1"],
            "caption_2": s["caption_2"],
            "target_rank": rank,
            "top_k_ids": full.ranked_ids[:args.top_k],
            "top_k_scores": [float(x) for x in full.scores[:args.top_k].tolist()],
        })
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.2f} queries/s)")
    if missing_target_in_gallery:
        print(f"WARNING: target missing from gallery for {missing_target_in_gallery}/{n} queries "
              "(counted as a miss at every K, not excluded from the denominator) - "
              "this usually means the gallery was built from a different CSV than --fiq-csv.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.category}_{args.split}_{args.variant}_ranked.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "category": args.category,
            "split": args.split,
            "variant": args.variant,
            "composition": ("q = normalize(r + mean(t1, t2))" if args.variant == "mean"
                            else f"q = normalize(r + t_{args.variant})"),
            "gallery_source": str(args.gallery),
            "gallery_size": len(gallery_ids),
            "num_queries": n,
            "device": str(enc.device),
            "queries": results,
        }, f, indent=2)
    print(f"Saved per-query results to {out_path}")


if __name__ == "__main__":
    main()