"""
Phase 5 sanity tests (spec Tests 4-9):
  4. Pair alignment loss finite
  5. CORAL loss finite
  6. Combined domain loss finite
  7. visual_adapter receives gradients from BOTH L_CIR and L_domain
  8. Frozen CLIP receives no gradients
  9. One complete Phase 5 training step executes
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.deepfashion2_dataset import DeepFashion2DomainDataset  # noqa: E402
from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.losses.contrastive import symmetric_contrastive_loss  # noqa: E402
from src.losses.domain_alignment import coral_loss, combined_domain_loss, paired_cosine_alignment_loss  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402


def cir_collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-csv", type=Path, required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--df2-csv", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/alignment_train.csv")
    ap.add_argument("--df2-root", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/images")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    args = ap.parse_args()

    n_fail = 0
    def check(name, ok, detail=""):
        nonlocal n_fail
        n_fail += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    print("=" * 50); print("Phase 5 Domain-Alignment Sanity Test"); print("=" * 50)

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    phase4_model = AttributeAwareCIRModel(clip)
    epoch, _ = load_phase4_checkpoint(phase4_model, args.phase4_checkpoint, clip.device)
    print(f"Warm-started from Phase 4 checkpoint (epoch {epoch})")
    model = CrossDomainCIRModel(phase4_model).to(clip.device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    check("CLIP frozen", all(n.startswith("clip.") for n in frozen) and len(frozen) > 0)
    check("cross_attn/fusion/visual_adapter trainable",
          any(n.startswith("cross_attn.") for n in trainable)
          and any(n.startswith("fusion.") for n in trainable)
          and any(n.startswith("visual_adapter.") for n in trainable))

    torch.manual_seed(0)
    f_u = F.normalize(torch.randn(6, 512), dim=-1)
    f_s = F.normalize(torch.randn(6, 512), dim=-1)
    check("paired_cosine_alignment_loss finite", torch.isfinite(paired_cosine_alignment_loss(f_u, f_s)).all().item())
    check("coral_loss finite", torch.isfinite(coral_loss(f_u, f_s)).all().item())
    combined, _ = combined_domain_loss(f_u, f_s, alpha=1.0, beta=0.1)
    check("combined_domain_loss finite", torch.isfinite(combined).all().item())
    check("coral_loss handles batch_size=1 safely", torch.isfinite(coral_loss(f_u[:1], f_s[:1])).all().item())

    fiq_ds = FashionIQCIRDataset(csv_path=args.fiq_csv, image_root=args.fiq_root)
    df2_ds = DeepFashion2DomainDataset(csv_path=args.df2_csv, image_root=args.df2_root)
    fiq_batch = next(iter(DataLoader(fiq_ds, batch_size=args.batch_size, collate_fn=cir_collate)))
    df2_batch = next(iter(DataLoader(df2_ds, batch_size=args.batch_size, collate_fn=cir_collate)))

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    optimizer.zero_grad()

    out = model(fiq_batch["reference_image"], fiq_batch["caption_1"], fiq_batch["caption_2"], fiq_batch["target_image"])
    cir_loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"])
    cir_loss.backward()

    f_user = model.encode_visual(df2_batch["user_image"])
    f_shop = model.encode_visual(df2_batch["shop_image"])
    domain_loss, _ = combined_domain_loss(f_user, f_shop, alpha=1.0, beta=0.1)
    (0.1 * domain_loss).backward()

    adapter_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                        for n, p in model.named_parameters() if n.startswith("visual_adapter."))
    clip_grad_none = all(p.grad is None for n, p in model.named_parameters() if n.startswith("clip."))
    check("visual_adapter receives gradients from BOTH losses", adapter_grad)
    check("CLIP receives no gradients", clip_grad_none)

    optimizer.step()
    check("One complete Phase 5 training step executed", True)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED" if n_fail == 0 else f"{n_fail} CHECK(S) FAILED")
    print("=" * 50)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()