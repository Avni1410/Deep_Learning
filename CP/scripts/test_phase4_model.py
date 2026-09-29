"""
Phase 4 sanity tests (runnable without attribute labels):

1. Tensor shapes through cross-attention / fusion / query / target embeddings.
2. Gradients reach cross-attention and fusion parameters.
3. CLIP parameters do NOT change after an optimizer step (frozen-backbone check).
4. Tiny-batch overfit: loss decreases when repeatedly training on the same
   small batch. This is a pipeline check, NOT a reported research result.

Usage:
  python scripts/test_phase4_model.py --fiq-csv data/processed/fashioniq/train_queries_dress.csv
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
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402


def cir_collate(batch: list[dict]) -> dict:
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--overfit-steps", type=int, default=30)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    args = ap.parse_args()

    n_fail = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal n_fail
        n_fail += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    print("=" * 50)
    print("Phase 4 Model Sanity Test")
    print("=" * 50)

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip)
    model.to(clip.device)
    print(f"Device: {clip.device}")

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    check("CLIP parameters are frozen", all(n.startswith("clip.") for n in frozen) and len(frozen) > 0)
    check("Cross-attention/fusion parameters are trainable",
          any(n.startswith("cross_attn.") for n in trainable) and any(n.startswith("fusion.") for n in trainable))

    ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=cir_collate)
    batch = next(iter(loader))
    B = len(batch["reference_image"])
    print(f"Batch size: {B}")

    out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"], debug=True)
    D = clip.embed_dim
    check("query_embedding shape [B, D]", list(out["query_embedding"].shape) == [B, D])
    check("target_embedding shape [B, D]", list(out["target_embedding"].shape) == [B, D])
    check("query_embedding finite", bool(torch.isfinite(out["query_embedding"]).all()))
    qn = out["query_embedding"].norm(dim=-1)
    check("query_embedding L2-normalized", bool(torch.allclose(qn, torch.ones_like(qn), atol=1e-3)))
    check("attribute_heads empty (Part A, no attribute_mapping.md yet)", len(out["attribute_logits"]) == 0)
    L = out["text_tokens"].shape[1]
    check("cross_attended_tokens shape [B, L1+L2, D]", list(out["cross_attended_tokens"].shape) == [B, L, D])

    # --- gradient flow ---
    model.train()
    for p in model.clip.parameters():
        assert not p.requires_grad  # sanity on the sanity check itself
    clip_hashes_before = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("clip.")}

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4
    )
    out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
    loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"])
    print(f"Initial loss (random batch): {loss.item():.4f}")
    optimizer.zero_grad()
    loss.backward()

    cross_attn_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if n.startswith("cross_attn.")
    )
    fusion_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters() if n.startswith("fusion.")
    )
    clip_grad_none = all(
        p.grad is None for n, p in model.named_parameters() if n.startswith("clip.")
    )
    check("Cross-attention receives non-zero gradients", cross_attn_grad)
    check("Fusion receives non-zero gradients", fusion_grad)
    check("CLIP parameters receive NO gradients", clip_grad_none)

    optimizer.step()
    clip_unchanged = all(
        torch.equal(clip_hashes_before[n], p) for n, p in model.named_parameters() if n.startswith("clip.")
    )
    check("CLIP parameters unchanged after optimizer step", clip_unchanged)

    # --- tiny overfit sanity (pipeline check only, NOT a reported result) ---
    print(f"\n--- Tiny overfit sanity ({args.overfit_steps} steps on {B} repeated examples) ---")
    losses = []
    for step in range(args.overfit_steps):
        out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
        loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == args.overfit_steps - 1:
            print(f"  step {step:>2}: loss = {loss.item():.4f}")
    check("Loss decreased over the tiny-batch overfit run", losses[-1] < losses[0],
          f"first={losses[0]:.4f} last={losses[-1]:.4f}")

    print("\n" + "=" * 50)
    print(f"{'ALL CHECKS PASSED' if n_fail == 0 else f'{n_fail} CHECK(S) FAILED'}")
    print("=" * 50)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()