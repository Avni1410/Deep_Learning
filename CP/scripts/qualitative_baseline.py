"""
Phase 3 - qualitative inspection: reference / target / top-N retrieved images.

Saves a SMALL number of image grids (default 5) under
experiments/baseline/qualitative/ - not the full validation set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", type=Path, required=True)
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--num-examples", type=int, default=5)
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/baseline/qualitative")
    args = ap.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("matplotlib is required for this script: pip install matplotlib")

    with open(args.ranked, "r", encoding="utf-8") as f:
        data = json.load(f)

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    by_query_id = {ds[i]["query_id"]: i for i in range(len(ds))}
    target_image_by_id = {}
    for i in range(len(ds)):
        s = ds[i]
        target_image_by_id.setdefault(s["target"], s["target_image"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = min(args.num_examples, len(data["queries"]))
    for q in data["queries"][:n]:
        sample = ds[by_query_id[q["query_id"]]]
        top_ids = q["top_k_ids"][:args.top_n]
        top_scores = q["top_k_scores"][:args.top_n]

        cols = 2 + len(top_ids)
        fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3.5))
        axes[0].imshow(sample["reference_image"]); axes[0].set_title("Reference"); axes[0].axis("off")
        axes[1].imshow(sample["target_image"]); axes[1].set_title(f"Target\n({q['target']})"); axes[1].axis("off")
        for j, (tid, score) in enumerate(zip(top_ids, top_scores), start=2):
            img = target_image_by_id.get(tid)
            if img is not None:
                axes[j].imshow(img)
            hit = " [HIT]" if tid == q["target"] else ""
            axes[j].set_title(f"#{j - 1} {tid}\n{score:.3f}{hit}")
            axes[j].axis("off")
        fig.suptitle(f"{q['query_id']}: \"{q['caption_1']}\" + \"{q['caption_2']}\" "
                     f"(rank of target = {q['target_rank']})")
        fig.tight_layout()
        out_path = args.out_dir / f"{q['query_id']}.png"
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()