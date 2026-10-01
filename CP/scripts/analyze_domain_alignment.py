"""
Phase 5 diagnostics (spec Sec 18, 19, 36):
  - mean cosine(user, shop) BEFORE (raw CLIP) vs AFTER (model.encode_visual) alignment
  - category-wise breakdown
  - representation-collapse check: mean pairwise cosine among a batch of
    DISTINCT images (high similarity among unrelated images suggests collapse)

Uses the HELD-OUT alignment_val.csv split, not the training pairs.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.deepfashion2_dataset import DeepFashion2DomainDataset  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def mean_pairwise_cosine(embeds: torch.Tensor) -> float:
    sims = embeds @ embeds.T
    n = sims.shape[0]
    off_diag = sims[~torch.eye(n, dtype=torch.bool)]
    return off_diag.mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df2-val-csv", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/alignment_val.csv")
    ap.add_argument("--df2-root", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/images")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--phase5-checkpoint", type=Path, required=True)
    ap.add_argument("--variant", required=True, choices=["pair", "coral", "combined"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    args = ap.parse_args()

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    phase4_model = AttributeAwareCIRModel(clip)
    load_phase4_checkpoint(phase4_model, args.phase4_checkpoint, clip.device)
    model = CrossDomainCIRModel(phase4_model).to(clip.device)

    ckpt = torch.load(args.phase5_checkpoint, map_location=clip.device, weights_only=False)
    model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
    model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
    model.visual_adapter.load_state_dict(ckpt["trainable_state"]["visual_adapter"])
    model.eval()

    ds = DeepFashion2DomainDataset(csv_path=args.df2_val_csv, image_root=args.df2_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    print(f"Held-out alignment val pairs: {len(ds)}")

    before_cos, after_cos = [], []
    by_cat_before, by_cat_after = defaultdict(list), defaultdict(list)
    all_before_user, all_after_user = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Analyzing"):
            raw_u = clip.encode_image(batch["user_image"]).global_embed
            raw_s = clip.encode_image(batch["shop_image"]).global_embed
            adapt_u = model.encode_visual(batch["user_image"])
            adapt_s = model.encode_visual(batch["shop_image"])

            cb = F.cosine_similarity(raw_u, raw_s, dim=-1)
            ca = F.cosine_similarity(adapt_u, adapt_s, dim=-1)
            before_cos.extend(cb.tolist())
            after_cos.extend(ca.tolist())
            all_before_user.append(raw_u.cpu())
            all_after_user.append(adapt_u.cpu())
            for cat, b, a in zip(batch["category_name"], cb.tolist(), ca.tolist()):
                by_cat_before[cat].append(b)
                by_cat_after[cat].append(a)

    print("=" * 60)
    print(f"Phase 5 Domain Alignment Diagnostics - variant: {args.variant}")
    print("=" * 60)
    print(f"Mean cosine(user, shop) BEFORE alignment: {sum(before_cos)/len(before_cos):.4f}")
    print(f"Mean cosine(user, shop) AFTER  alignment: {sum(after_cos)/len(after_cos):.4f}")
    print()
    print(f"{'Category':<20} {'N':>5} {'Before':>8} {'After':>8}")
    for cat in sorted(by_cat_before):
        b = sum(by_cat_before[cat]) / len(by_cat_before[cat])
        a = sum(by_cat_after[cat]) / len(by_cat_after[cat])
        print(f"{cat:<20} {len(by_cat_before[cat]):>5} {b:>8.4f} {a:>8.4f}")

    before_stack = torch.cat(all_before_user, dim=0)
    after_stack = torch.cat(all_after_user, dim=0)
    print()
    print("Representation collapse check (mean pairwise cosine among DISTINCT user images):")
    print(f"  Before alignment: {mean_pairwise_cosine(F.normalize(before_stack, dim=-1)):.4f}")
    print(f"  After  alignment: {mean_pairwise_cosine(after_stack):.4f}")
    print("  (A large jump toward 1.0 after alignment would indicate collapse - investigate if so.)")


if __name__ == "__main__":
    main()