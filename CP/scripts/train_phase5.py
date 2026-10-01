"""
Phase 5 training - alternating FashionIQ/DeepFashion2 batches, gradients
accumulated into one optimizer step (see chat discussion: no retain_graph
needed, the two forward passes are independent graphs).

--domain-loss selects Experiment 2 (pair), 3 (coral), or 4 (combined).
Warm-starts cross_attn/fusion from the Phase 4 checkpoint; visual_adapter
is new and initializes near-identity (residual + LayerNorm-free small MLP).
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.deepfashion2_dataset import DeepFashion2DomainDataset  # noqa: E402
from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.losses.contrastive import symmetric_contrastive_loss  # noqa: E402
from src.losses.domain_alignment import coral_loss, paired_cosine_alignment_loss  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402
from src.models.composed_retrieval_model import CrossDomainCIRModel, load_phase4_checkpoint  # noqa: E402


def collate(batch):
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def compute_domain_loss(kind: str, f_user, f_shop, alpha: float, beta: float):
    if kind == "pair":
        return paired_cosine_alignment_loss(f_user, f_shop), {"pair": None}
    if kind == "coral":
        return coral_loss(f_user, f_shop), {"coral": None}
    if kind == "combined":
        l_pair = paired_cosine_alignment_loss(f_user, f_shop)
        l_coral = coral_loss(f_user, f_shop)
        return alpha * l_pair + beta * l_coral, {"pair": l_pair.item(), "coral": l_coral.item()}
    raise ValueError(kind)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiq-train-csv", type=Path, nargs="+", required=True)
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--df2-train-csv", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/alignment_train.csv")
    ap.add_argument("--df2-root", type=Path, default=REPO_ROOT / "data/processed/deepfashion2/images")
    ap.add_argument("--phase4-checkpoint", type=Path, required=True)
    ap.add_argument("--domain-loss", required=True, choices=["pair", "coral", "combined"])
    ap.add_argument("--lambda-domain", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--fiq-batch-size", type=int, default=4)
    ap.add_argument("--df2-batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase5")
    ap.add_argument("--run-name", required=True, help="e.g. exp2_pair, exp3_coral, exp4_combined")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    phase4_model = AttributeAwareCIRModel(clip)
    epoch4, _ = load_phase4_checkpoint(phase4_model, args.phase4_checkpoint, clip.device)
    print(f"Warm-started cross_attn/fusion from Phase 4 checkpoint (epoch {epoch4})")
    model = CrossDomainCIRModel(phase4_model).to(clip.device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params (cross_attn+fusion+visual_adapter): {n_trainable:,} | Frozen (CLIP): {n_frozen:,}")

    fiq_datasets = []
    for p in args.fiq_train_csv:
        ds = FashionIQCIRDataset(csv_path=p, image_root=args.fiq_root)
        if args.limit is not None:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        fiq_datasets.append(ds)
        print(f"  FashionIQ {p.name}: {len(ds)} queries")
    fiq_loader = DataLoader(ConcatDataset(fiq_datasets), batch_size=args.fiq_batch_size,
                             shuffle=True, collate_fn=collate)

    df2_ds = DeepFashion2DomainDataset(csv_path=args.df2_train_csv, image_root=args.df2_root)
    if args.limit is not None:
        df2_ds = torch.utils.data.Subset(df2_ds, range(min(args.limit, len(df2_ds))))
    print(f"  DeepFashion2 alignment train: {len(df2_ds)} pairs")
    df2_loader = DataLoader(df2_ds, batch_size=args.df2_batch_size, shuffle=True, collate_fn=collate)
    df2_iter = itertools.cycle(df2_loader)  # DeepFashion2 (4000) is smaller than combined FashionIQ train

    print(f"Batches/epoch (FashionIQ-driven): {len(fiq_loader)} | Domain loss: {args.domain_loss}")

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=args.lr, weight_decay=args.weight_decay)

    out_dir = args.out_dir / args.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "loss_cir", "loss_domain", "loss_pair", "loss_coral", "elapsed_s"])

    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        for step, fiq_batch in enumerate(fiq_loader):
            df2_batch = next(df2_iter)
            optimizer.zero_grad()

            out = model(fiq_batch["reference_image"], fiq_batch["caption_1"],
                        fiq_batch["caption_2"], fiq_batch["target_image"])
            cir_loss = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"], args.temperature)
            cir_loss.backward()

            f_user = model.encode_visual(df2_batch["user_image"])
            f_shop = model.encode_visual(df2_batch["shop_image"])
            domain_loss, sub = compute_domain_loss(args.domain_loss, f_user, f_shop, args.alpha, args.beta)
            (args.lambda_domain * domain_loss).backward()

            optimizer.step()

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {step}/{len(fiq_loader)} "
                      f"cir={cir_loss.item():.4f} domain={domain_loss.item():.4f} "
                      f"pair={sub.get('pair', 'n/a')} coral={sub.get('coral', 'n/a')} "
                      f"elapsed={elapsed:.0f}s")
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([epoch, step, cir_loss.item(), domain_loss.item(),
                                             sub.get("pair"), sub.get("coral"), elapsed])

        ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "trainable_state": {
                "cross_attn": model.cross_attn.state_dict(),
                "fusion": model.fusion.state_dict(),
                "visual_adapter": model.visual_adapter.state_dict(),
            },
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
        }, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    print(f"Training complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()