"""
Phase 3 - compute Recall@K from per-query ranked results.

Usage (one variant/split at a time, all its categories together):
  python scripts/evaluate_baseline.py \
      --ranked experiments/baseline/dress_val_mean_ranked.json \
               experiments/baseline/shirt_val_mean_ranked.json \
               experiments/baseline/toptee_val_mean_ranked.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import recall_at_k_by_category  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", type=Path, nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 50])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "experiments/baseline/results.json")
    args = ap.parse_args()

    ranks: list[int | None] = []
    categories: list[str] = []
    meta = {"model": None, "variant": None, "split": None, "gallery_sizes": {}}
    for path in args.ranked:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for m in ("model", "variant", "split"):
            if meta[m] is None:
                meta[m] = data[m]
            elif meta[m] != data[m]:
                raise SystemExit(
                    f"{path} has {m}={data[m]!r}, expected {meta[m]!r}. "
                    "Evaluate one variant/split at a time."
                )
        meta["gallery_sizes"][data["category"]] = data["gallery_size"]
        for q in data["queries"]:
            ranks.append(q["target_rank"])
            categories.append(q["category"])

    per_category = recall_at_k_by_category(ranks, categories, args.k)

    print("=" * 50)
    print("Phase 3 Baseline CIR - Evaluation")
    print("=" * 50)
    print(f"Model:   {meta['model']}")
    print(f"Variant: {meta['variant']}")
    print(f"Split:   {meta['split']}")
    print(f"Queries: {len(ranks)}")
    for cat, size in meta["gallery_sizes"].items():
        print(f"Gallery ({cat}): {size} images")
    print()
    for cat, r in per_category.items():
        label = "Overall" if cat == "overall" else cat.capitalize()
        print(f"{label}:")
        for k, v in r.items():
            print(f"  R@{k:<3} = {v:.4f}")
    print("=" * 50)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": meta["model"],
        "composition": ("q = normalize(r + mean(t1, t2))" if meta["variant"] == "mean"
                        else f"q = normalize(r + t_{meta['variant']})"),
        "dataset": "FashionIQ",
        "split": meta["split"],
        "variant": meta["variant"],
        "categories": sorted(meta["gallery_sizes"].keys()),
        "num_queries": len(ranks),
        "gallery_sizes": meta["gallery_sizes"],
        "metrics": {
            ("overall" if cat == "overall" else cat): {f"R@{k}": v for k, v in r.items()}
            for cat, r in per_category.items()
        },
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }
    existing = []
    if args.out.exists():
        with open(args.out, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if isinstance(existing, dict):
                existing = [existing]
    existing.append(payload)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print(f"\nAppended results to {args.out}")


if __name__ == "__main__":
    main()