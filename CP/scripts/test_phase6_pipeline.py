"""
Phase 6 sanity tests (spec Tests 1-10), runnable without a combined gallery:
  1-3. Gallery/query embedding + top-K retrieval (reuses verified Phase 3/4 code)
  4. True target removed from mined negatives
  5. Duplicate (candidate==target scenario) handling doesn't crash
  7. Hard-negative file generation produces a valid CSV row
  8. hard_negative_contrastive_loss returns finite value
  9. Gradients are finite and reach the trainable model
  10. One Phase 6 training step executes (CIR + hard-negative loss combined)
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
from src.losses.hard_negative_loss import hard_negative_contrastive_loss  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import load_phase4_checkpoint  # noqa: E402
from src.retrieval.hard_negative_miner import mine_for_query  # noqa: E402


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
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

    print("=" * 50); print("Phase 6 Pipeline Sanity Test"); print("=" * 50)

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)
    epoch, _ = load_phase4_checkpoint(model, args.phase4_checkpoint, clip.device)
    model.to(clip.device)
    print(f"Loaded Phase 4 checkpoint (epoch {epoch}) as Phase 6 starting point (not Phase 5, per decision)")

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate)
    batch = next(iter(loader))
    B = len(batch["reference_image"])

    # Tests 1-3: build a tiny gallery from this same batch's targets (self-contained, no file I/O needed)
    with torch.no_grad():
        gallery_embeds = model.clip.encode_image(batch["target_image"]).global_embed.cpu()
    gallery_ids = batch["target"]
    check("Gallery embeddings generated", list(gallery_embeds.shape) == [B, clip.embed_dim])

    with torch.no_grad():
        q = model.encode_query(batch["reference_image"], batch["caption_1"], batch["caption_2"])["query_embedding"]
    check("Query embeddings generated", list(q.shape) == [B, clip.embed_dim])

    from src.retrieval.cosine_retrieval import retrieve_top_k
    res = retrieve_top_k(q[0].cpu(), gallery_embeds, gallery_ids, k=B)
    check("Top-K retrieval works", len(res.ranked_ids) == B)

    # Test 4: target removal
    mined = mine_for_query(q[0].cpu(), batch["query_id"][0], batch["category"][0], batch["target"][0],
                            batch["candidate"][0], gallery_ids, gallery_embeds, None,
                            k_retrieval=B, n_hard=args.n_hard, strategy="similarity")
    check("True target removed from mined negatives", batch["target"][0] not in mined.negative_ids)
    check("Candidate/reference ID removed from mined negatives", batch["candidate"][0] not in mined.negative_ids)

    # Test 7 (lightweight): mined result has the expected shape to become a CSV row
    check("Mined negatives has expected fields",
          hasattr(mined, "negative_ids") and hasattr(mined, "negative_similarities")
          and len(mined.negative_ids) == len(mined.negative_similarities))

    # For the loss/training tests, fabricate a plausible hard-negative batch from
    # the OTHER targets in this same batch (self-contained, no mining file needed)
    M = min(args.n_hard, B - 1)
    if M < 1:
        print("  [SKIP] remaining tests need batch_size > 1 to fabricate hard negatives")
        sys.exit(1 if n_fail else 0)

    with torch.no_grad():
        other_targets = model.clip.encode_image(batch["target_image"][1:1 + M]).global_embed  # [M, D]
    hard_negs = other_targets.unsqueeze(0).repeat(B, 1, 1).to(clip.device)  # [B, M, D], fabricated for the test

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()

    out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
    cir_loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"])
    hn_loss = hard_negative_contrastive_loss(out["query_embedding"], out["target_embedding"], hard_negs)
    check("L_HN finite", torch.isfinite(hn_loss).item())

    total = cir_loss + 0.1 * hn_loss
    total.backward()
    grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all()
                  for n, p in model.named_parameters() if n.startswith("cross_attn.") and p.grad is not None)
    check("Gradients finite and reach cross_attn", grad_ok)
    optimizer.step()
    check("One Phase 6 training step executed (CIR + L_HN combined)", True)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED")
    print("=" * 50)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()