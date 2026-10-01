"""
Phase 5 diagnostic (NOT a training script, NOT a fix) - determine whether
CORAL's near-zero values at batch_size=2 are explained by (a) the batch size
being too small to estimate covariance meaningfully, (b) a scale mismatch
between the 4d^2 CORAL normalizer and L2-normalized unit-sphere features, or
both. Uses the already-trained exp3_coral checkpoint, frozen, and real
DeepFashion2 pairs. No parameters are updated.
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
from src.losses.domain_alignment import coral_loss, paired_cosine_alignment_loss  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def coral_raw_frobenius(user_feats: torch.Tensor, shop_feats: torch.Tensor):
    """Same math as coral_loss but also returns the UN-normalized (pre-4d^2)
    squared Frobenius difference and both traces, for diagnosis."""
    B_u, d = user_feats.shape
    u = user_feats - user_feats.mean(dim=0, keepdim=True)
    s = shop_feats - shop_feats.mean(dim=0, keepdim=True)
    cov_u = (u.T @ u) / (B_u - 1)
    cov_s = (s.T @ s) / (shop_feats.shape[0] - 1)
    diff = cov_u - cov_s
    raw_sq_frob = (diff * diff).sum()
    return raw_sq_frob, cov_u.trace().item(), cov_s.trace().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--df2-csv", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/alignment_train.csv")
    ap.add_argument("--df2-root", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/images")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--phase5-checkpoint", type=Path, required=True,
                    help="the TRAINED exp3_coral checkpoint - weights only, not updated here")
    ap.add_argument("--encode-batch-size", type=int, default=8,
                    help="DataLoader batch for encoding (memory-safe), decoupled from the B values tested below")
    ap.add_argument("--max-samples", type=int, default=64)
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
    print(f"Loaded exp3_coral checkpoint (epoch {ckpt['epoch']}) - INFERENCE ONLY, no weights will change.")

    ds = DeepFashion2DomainDataset(csv_path=args.df2_csv, image_root=args.df2_root)
    loader = DataLoader(ds, batch_size=args.encode_batch_size, shuffle=True, collate_fn=collate)

    all_user, all_shop = [], []
    with torch.no_grad():
        for batch in loader:
            all_user.append(model.encode_visual(batch["user_image"]).cpu())
            all_shop.append(model.encode_visual(batch["shop_image"]).cpu())
            if sum(t.shape[0] for t in all_user) >= args.max_samples:
                break
    user_feats = torch.cat(all_user, dim=0)[: args.max_samples]
    shop_feats = torch.cat(all_shop, dim=0)[: args.max_samples]
    print(f"Encoded {user_feats.shape[0]} real DeepFashion2 pairs through the trained visual_adapter.\n")

    print(f"{'B':>4} {'L_pair':>10} {'raw ||diff||_F^2':>18} {'L_CORAL (/4d^2)':>18} {'trace(cov_u)':>14} {'trace(cov_s)':>14}")
    for B in [2, 8, 16, 32, 64]:
        if B > user_feats.shape[0]:
            continue
        u, s = user_feats[:B], shop_feats[:B]
        l_pair = paired_cosine_alignment_loss(u, s).item()
        l_coral = coral_loss(u, s).item()
        raw_frob, tr_u, tr_s = coral_raw_frobenius(u, s)
        print(f"{B:>4} {l_pair:>10.4f} {raw_frob.item():>18.6e} {l_coral:>18.6e} {tr_u:>14.6f} {tr_s:>14.6f}")

    print("\nInterpretation guide:")
    print("- If L_CORAL grows toward a 'normal' magnitude (e.g. >= 1e-3) by B=32-64:")
    print("    batch_size=2 was the dominant problem -> fix: decouple a larger df2 batch")
    print("    specifically for CORAL (e.g. accumulate 32-64 DeepFashion2 samples before")
    print("    computing coral_loss, independent of the FashionIQ batch size).")
    print("- If L_CORAL stays << L_pair (e.g. < 1e-4) even at B=64:")
    print("    the 4d^2 normalizer is scale-mismatched to L2-normalized unit-sphere features")
    print("    -> fix: either drop the /4d^2 term (let beta carry the scale instead) or")
    print("    compute CORAL on the pre-L2-norm adapter output specifically.")
    print("- 'raw ||diff||_F^2' isolates the numerator alone - compare its growth rate")
    print("  against trace(cov_u)/trace(cov_s) to see if cancellation (correlated rank-1")
    print("  directions) is adding to the effect beyond pure batch-size noise.")


if __name__ == "__main__":
    main()