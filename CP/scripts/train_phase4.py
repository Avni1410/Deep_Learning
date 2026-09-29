"""
Phase 4 training - ablation B (cross-attention + fusion, no attribute loss yet).

Passing --attribute-vocab-json later (once docs/attribute_mapping.md is
available) adds attribute heads and ablation C's attribute loss; with no
vocab, attribute_logits is empty and the attribute loss term is skipped
automatically (loss_attr = 0), which is exactly ablation B.

CLIP stays frozen throughout; only cross-attention, fusion, and (if present)
attribute heads train.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset  # noqa: E402
from src.losses.attribute import combined_attribute_loss  # noqa: E402
from src.losses.contrastive import symmetric_contrastive_loss  # noqa: E402
from src.models.attribute_aware_cir import AttributeAwareCIRModel  # noqa: E402
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig  # noqa: E402


def cir_collate(batch: list[dict]) -> dict:
    return {k: [b[k] for b in batch] for k in batch[0].keys()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, nargs="+", required=True,
                    help="one or more train_queries_<category>.csv files (combined for training)")
    ap.add_argument("--fiq-root", type=Path, default=REPO_ROOT / "data/raw/fashioniq")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--lambda-cir", type=float, default=1.0)
    ap.add_argument("--lambda-attr", type=float, default=0.5)
    ap.add_argument("--attribute-vocab-json", type=Path, default=None,
                    help="JSON {attribute_name: num_classes}; omit for ablation B")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    ap.add_argument("--limit", type=int, default=None, help="cap queries per csv, for quick smoke runs")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "experiments/phase4")
    ap.add_argument("--run-name", default="ablation_b_cross_attn_fusion")
    ap.add_argument("--resume-from", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    attribute_vocab = None
    if args.attribute_vocab_json is not None:
        attribute_vocab = json.loads(args.attribute_vocab_json.read_text())
        print(f"Loaded attribute vocabulary: {attribute_vocab}")
    else:
        print("No --attribute-vocab-json given: running WITHOUT attribute heads (ablation B).")

    clip = CLIPEncoder(CLIPEncoderConfig(model_name=args.model, device=args.device))
    model = AttributeAwareCIRModel(clip, attribute_vocab=attribute_vocab).to(clip.device)
    print(f"Device: {clip.device}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {n_trainable:,} | Frozen (CLIP) params: {n_frozen:,}")

    datasets = []
    for csv_path in args.train_csv:
        ds = FashionIQCIRDataset(csv_path=csv_path, image_root=args.fiq_root)
        if args.limit is not None:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
        datasets.append(ds)
        print(f"  {csv_path.name}: {len(ds)} queries")
    train_ds = ConcatDataset(datasets)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=cir_collate, num_workers=0)
    print(f"Total training queries: {len(train_ds)} | batches/epoch: {len(loader)}")

    use_amp = clip.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay
    )

    start_epoch = 0
    if args.resume_from is not None:
        ckpt = torch.load(args.resume_from, map_location=clip.device, weights_only=False)
        model.cross_attn.load_state_dict(ckpt["trainable_state"]["cross_attn"])
        model.fusion.load_state_dict(ckpt["trainable_state"]["fusion"])
        model.attribute_heads.load_state_dict(ckpt["trainable_state"]["attribute_heads"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {args.resume_from}, starting at epoch {start_epoch}")

    out_dir = args.out_dir / args.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "loss_total", "loss_cir", "loss_attr", "elapsed_s"])

    model.train()
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad()
        for step, batch in enumerate(loader):
            with torch.autocast(device_type=clip.device.type, enabled=use_amp):
                out = model(batch["reference_image"], batch["caption_1"], batch["caption_2"], batch["target_image"])
                loss_cir = symmetric_contrastive_loss(out["query_embedding"], out["target_embedding"], args.temperature)
                loss_attr, _ = (None, {}) if not attribute_vocab else combined_attribute_loss(
                    out["attribute_logits"], {}  # attribute LABELS not wired yet - Part B
                )
                loss_attr_value = loss_attr if loss_attr is not None else torch.zeros((), device=clip.device)
                loss = args.lambda_cir * loss_cir + args.lambda_attr * loss_attr_value
                loss_to_backward = loss / args.grad_accum

            scaler.scale(loss_to_backward).backward()
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(f"epoch {epoch} step {step}/{len(loader)} "
                      f"loss={loss.item():.4f} cir={loss_cir.item():.4f} "
                      f"attr={float(loss_attr_value):.4f} elapsed={elapsed:.0f}s")
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [epoch, step, loss.item(), loss_cir.item(), float(loss_attr_value), elapsed]
                    )

        trainable_state = {
            "cross_attn": model.cross_attn.state_dict(),
            "fusion": model.fusion.state_dict(),
            "attribute_heads": model.attribute_heads.state_dict(),
        }
        ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "trainable_state": trainable_state,
            "optimizer_state": optimizer.state_dict(),
            "config": vars(args),
            "attribute_vocab": attribute_vocab,
        }, ckpt_path)


    print(f"Training complete in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()