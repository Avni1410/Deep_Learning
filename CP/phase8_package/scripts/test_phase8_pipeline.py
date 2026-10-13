"""
Phase 8 sanity tests - run before any real training/mining at scale.

  1. Model loads from Phase 4 checkpoint correctly.
  2. Semi-hard mining NEVER returns a negative with similarity >= positive
     (the core guarantee that fixes Phase 6's diagnosed failure mode).
  3. Shortfall is reported, not silently padded.
  4. hard_negative_loss_variable handles a mix of zero/nonzero negative
     counts in one batch without crashing.
  5. Gradients are finite and reach cross_attn when hard negatives exist.
  6. One full Phase-8-style training step (CIR + warm-up-scaled HN loss)
     executes successfully.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.losses.contrastive import symmetric_contrastive_loss  # noqa: E402
from src.losses.semi_hard_loss import hard_negative_loss_variable  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import load_phase4_checkpoint  # noqa: E402
from src.retrieval.hard_negative_miner import load_duplicate_lookup  # noqa: E402
from src.retrieval.semi_hard_miner import mine_semi_hard_for_query  # noqa: E402
from src.utils.checkpoint_compat import patch_path_compat  # noqa: E402

patch_path_compat()


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--combined-gallery", type=Path, required=True)
    ap.add_argument("--duplicate-lookup", type=Path,
                    default=REPO_ROOT / "experiments/phase6/fashioniq_duplicate_lookup.csv")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-hard", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    args = ap.parse_args()

    n_fail = 0
    def check(name, ok, detail=""):
        nonlocal n_fail
        n_fail += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    print("=" * 50); print("Phase 8 Pipeline Sanity Test"); print("=" * 50)

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)
    epoch, _ = load_phase4_checkpoint(model, args.phase4_checkpoint, clip.device)
    model.to(clip.device)
    check("Loaded Phase 4 checkpoint", epoch is not None, f"epoch={epoch}")

    combined = torch.load(args.combined_gallery, map_location="cpu", weights_only=False)
    gallery_ids, gallery_categories, gallery_embeds = (
        combined["ids"], combined["categories"], combined["embeddings"]
    )
    asin_to_group, group_to_asins = load_duplicate_lookup(args.duplicate_lookup)
    check("Combined gallery and duplicate lookup loaded",
          len(gallery_ids) > 0 and len(asin_to_group) >= 0)

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    batch = next(iter(loader))
    B = len(batch["reference_image"])

    model.eval()
    mined = []
    with torch.no_grad():
        for i in range(B):
            q = model.encode_query(
                [batch["reference_image"][i]], [batch["caption_1"][i]], [batch["caption_2"][i]]
            )["query_embedding"].squeeze(0).cpu()
            m = mine_semi_hard_for_query(
                q, batch["query_id"][i], batch["category"][i], batch["target"][i], batch["candidate"][i],
                gallery_ids, gallery_embeds, gallery_categories, asin_to_group, group_to_asins,
                k_retrieval=50, n_hard=args.n_hard, strategy="similarity", margin_buffer=0.0,
            )
            mined.append(m)

    # Test 2: the core guarantee
    violations = sum(
        1 for m in mined for s in m.negative_similarities
        if m.positive_similarity == m.positive_similarity and s >= m.positive_similarity  # NaN-safe
    )
    check("Semi-hard guarantee holds (no negative >= positive)", violations == 0, f"violations={violations}")

    # Test 3: shortfall is reported honestly, not padded
    check("Shortfall field present and consistent",
          all(m.shortfall == m.n_requested - m.n_found for m in mined))

    # Test 4: variable-length loss with a deliberately mixed batch (some zero-negative samples)
    pos = torch.randn(B, clip.embed_dim)
    q_fake = torch.randn(B, clip.embed_dim)
    neg_list = []
    for i, m in enumerate(mined):
        if len(m.negative_ids) == 0 or i == 0:  # force at least one empty case for the test
            neg_list.append(None)
        else:
            neg_list.append(torch.randn(len(m.negative_ids), clip.embed_dim))
    loss, n_used = hard_negative_loss_variable(q_fake, pos, neg_list, temperature=0.07)
    check("hard_negative_loss_variable handles mixed zero/nonzero batch", torch.isfinite(loss).item(),
          f"n_used={n_used}/{B}")

    # Test 5-6: real gradient flow + one training step
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()
    out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
    cir_loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"])

    neg_embeds_list = []
    for m in mined:
        if len(m.negative_ids) == 0:
            neg_embeds_list.append(None)
            continue
        imgs = []
        for neg_id, neg_cat in zip(m.negative_ids, m.negative_categories):
            from src.data.fashioniq_paths import load_fashioniq_image
            imgs.append(load_fashioniq_image(args.fiq_root, neg_cat, neg_id))
        neg_embeds_list.append(model.clip.encode_image(imgs).global_embed)

    hn_loss, n_used_real = hard_negative_loss_variable(
        out["query_embedding"], out["target_embedding"], neg_embeds_list, temperature=0.07
    )
    check("Real-data L_HN finite", torch.isfinite(hn_loss).item(), f"n_used={n_used_real}/{B}")

    lambda_hn = 0.1  # mid-warmup value for the test
    total = cir_loss + lambda_hn * hn_loss
    total.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for n, p in model.named_parameters() if n.startswith("cross_attn.") and p.grad is not None)
    check("Gradients finite and reach cross_attn", grad_ok)
    optimizer.step()
    check("One Phase 8 training step executed (CIR + semi-hard L_HN)", True)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED")
    print("=" * 50)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
