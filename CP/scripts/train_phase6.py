"""
Phase 6 training: Fine-grained contrastive learning with hard negatives.

Starts from the Phase 4 epoch-2 checkpoint and further trains:
    - cross-attention
    - fusion

CLIP remains frozen.

Training data:
    Original FashionIQ query CSVs provide:
        candidate/reference image
        target image
        caption_1
        caption_2

    Phase 6 mined-negative CSVs provide:
        negative_1
        negative_2
        negative_3
        negative categories

Objective:
    L_total = L_CIR + lambda_hn * L_hard_negative
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.losses.contrastive import symmetric_contrastive_loss
from src.losses.phase6_loss import compute_phase6_loss
from src.models.attribute_aware_cir import AttributeAwareCIRModel
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig


class Phase6Dataset(Dataset):
    """
    Joins original FashionIQ query metadata with mined hard negatives.
    """

    def __init__(
        self,
        query_csvs,
        negative_csvs,
        image_root,
        quality_exclusions=None,
        limit=None,
    ):
        self.image_root = Path(image_root)

        query_frames = []
        for path in query_csvs:
            df = pd.read_csv(path)
            query_frames.append(df)

        query_df = pd.concat(
            query_frames,
            ignore_index=True,
        )

        negative_frames = []
        for path in negative_csvs:
            df = pd.read_csv(path)
            negative_frames.append(df)

        negative_df = pd.concat(
            negative_frames,
            ignore_index=True,
        )

        required_query = {
            "query_id",
            "category",
            "candidate",
            "target",
            "caption_1",
            "caption_2",
        }

        required_negative = {
            "query_id",
            "negative_1",
            "negative_2",
            "negative_3",
            "negative_1_category",
            "negative_2_category",
            "negative_3_category",
        }

        missing_query = required_query - set(query_df.columns)
        missing_negative = required_negative - set(negative_df.columns)

        if missing_query:
            raise ValueError(
                f"Missing query columns: {sorted(missing_query)}"
            )

        if missing_negative:
            raise ValueError(
                f"Missing negative columns: {sorted(missing_negative)}"
            )

        negative_columns = [
            "query_id",
            "negative_1",
            "negative_2",
            "negative_3",
            "negative_1_category",
            "negative_2_category",
            "negative_3_category",
        ]

        negative_df = negative_df[negative_columns]

        self.df = query_df.merge(
            negative_df,
            on="query_id",
            how="inner",
            validate="one_to_one",
        )

        if quality_exclusions:
            quality_exclusions = set(quality_exclusions)

            self.df = self.df[
                ~self.df["query_id"].isin(quality_exclusions)
            ]

        self.df = self.df.reset_index(drop=True)

        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        print(
            f"Phase 6 dataset: {len(self.df)} queries"
        )

    def __len__(self):
        return len(self.df)

    def _path(self, category, asin):
        return (
            self.image_root
            / str(category)
            / f"{asin}.jpg"
        )

    def _load_image(self, path):
        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {path}"
            )

        return Image.open(path).convert("RGB")

    def __getitem__(self, index):
        row = self.df.iloc[index]

        category = str(row["category"])

        candidate = str(row["candidate"])
        target = str(row["target"])

        negative_ids = [
            str(row["negative_1"]),
            str(row["negative_2"]),
            str(row["negative_3"]),
        ]

        negative_categories = [
            str(row["negative_1_category"]),
            str(row["negative_2_category"]),
            str(row["negative_3_category"]),
        ]

        reference_path = self._path(
            category,
            candidate,
        )

        target_path = self._path(
            category,
            target,
        )

        negative_paths = [
            self._path(cat, asin)
            for asin, cat in zip(
                negative_ids,
                negative_categories,
            )
        ]

        return {
            "query_id": str(row["query_id"]),
            "category": category,
            "reference_image": self._load_image(
                reference_path
            ),
            "target_image": self._load_image(
                target_path
            ),
            "negative_images": [
                self._load_image(path)
                for path in negative_paths
            ],
            "caption_1": str(row["caption_1"]),
            "caption_2": str(row["caption_2"]),
            "candidate": candidate,
            "target": target,
            "negatives": negative_ids,
        }


def collate_phase6(batch):
    return {
        "query_id": [x["query_id"] for x in batch],
        "category": [x["category"] for x in batch],
        "reference_image": [
            x["reference_image"] for x in batch
        ],
        "target_image": [
            x["target_image"] for x in batch
        ],
        "negative_images": [
            x["negative_images"] for x in batch
        ],
        "caption_1": [
            x["caption_1"] for x in batch
        ],
        "caption_2": [
            x["caption_2"] for x in batch
        ],
        "candidate": [
            x["candidate"] for x in batch
        ],
        "target": [
            x["target"] for x in batch
        ],
        "negatives": [
            x["negatives"] for x in batch
        ],
    }


def load_phase4_checkpoint(
    model,
    checkpoint_path,
    device,
):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    trainable_state = checkpoint["trainable_state"]

    model.cross_attn.load_state_dict(
        trainable_state["cross_attn"]
    )

    model.fusion.load_state_dict(
        trainable_state["fusion"]
    )

    if "attribute_heads" in trainable_state:
        model.attribute_heads.load_state_dict(
            trainable_state["attribute_heads"]
        )

    print(
        f"Loaded Phase 4 checkpoint: {checkpoint_path}"
    )

    print(
        f"Starting from Phase 4 epoch "
        f"{checkpoint.get('epoch', 'unknown')}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query-csv",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--negative-csv",
        type=Path,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--fiq-root",
        type=Path,
        default=(
            REPO_ROOT
            / "data/raw/fashioniq"
        ),
    )

    parser.add_argument(
        "--phase4-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--quality-exclusions",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/phase6/"
            "quality_exclusions.txt"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )

    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch16",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments/phase6"
        ),
    )

    parser.add_argument(
        "--run-name",
        default="phase6_hard_negative",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    else:
        device = args.device

    print(f"Device: {device}")

    quality_exclusions = []

    if args.quality_exclusions.exists():
        quality_exclusions = [
            line.strip()
            for line in args.quality_exclusions
            .read_text()
            .splitlines()
            if line.strip()
        ]

    print(
        f"Quality exclusions: "
        f"{len(quality_exclusions)}"
    )

    dataset = Phase6Dataset(
        query_csvs=args.query_csv,
        negative_csvs=args.negative_csv,
        image_root=args.fiq_root,
        quality_exclusions=quality_exclusions,
        limit=args.limit,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_phase6,
        num_workers=0,
    )

    print(
        f"Queries: {len(dataset)}"
    )

    print(
        f"Batches/epoch: {len(loader)}"
    )

    clip = CLIPEncoder(
        CLIPEncoderConfig(
            model_name=args.model,
            device=device,
            freeze=True,
        )
    )

    model = AttributeAwareCIRModel(
        clip,
        attribute_vocab=None,
    ).to(clip.device)

    load_phase4_checkpoint(
        model,
        args.phase4_checkpoint,
        clip.device,
    )

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    n_trainable = sum(
        p.numel()
        for p in trainable_params
    )

    n_frozen = sum(
        p.numel()
        for p in model.parameters()
        if not p.requires_grad
    )

    print(
        f"Trainable params: "
        f"{n_trainable:,}"
    )

    print(
        f"Frozen params: "
        f"{n_frozen:,}"
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = (
        clip.device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    out_dir = (
        args.out_dir
        / args.run_name
    )

    checkpoint_dir = (
        out_dir / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        out_dir / "train_log.csv"
    )

    with open(
        log_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "step",
                "loss_total",
                "loss_cir",
                "loss_hard_negative",
                "elapsed_s",
            ]
        )

    model.train()

    t0 = time.time()

    optimizer.zero_grad()

    for epoch in range(args.epochs):

        for step, batch in enumerate(loader):

            with torch.autocast(
                device_type=clip.device.type,
                enabled=use_amp,
            ):

                output = model(
                    batch["reference_image"],
                    batch["caption_1"],
                    batch["caption_2"],
                    batch["target_image"],
                )

                query_embeddings = (
                    output["query_embedding"]
                )

                positive_embeddings = (
                    output["target_embedding"]
                )

                negative_embeddings = []

                for i in range(
                    len(batch["negative_images"])
                ):

                    negatives_for_query = (
                        batch["negative_images"][i]
                    )

                    encoded = []

                    for image in (
                        negatives_for_query
                    ):

                        encoded.append(
                            model.encode_target(
                                [image]
                            )
                        )

                    encoded = torch.cat(
                        encoded,
                        dim=0,
                    )

                    negative_embeddings.append(
                        encoded
                    )

                negative_embeddings = torch.stack(
                    negative_embeddings,
                    dim=0,
                )

                cir_loss = (
                    symmetric_contrastive_loss(
                        query_embeddings,
                        positive_embeddings,
                        args.temperature,
                    )
                )

                losses = compute_phase6_loss(
                    query_embeddings=(
                        query_embeddings
                    ),
                    positive_embeddings=(
                        positive_embeddings
                    ),
                    negative_embeddings=(
                        negative_embeddings
                    ),
                    cir_loss=cir_loss,
                    margin=args.margin,
                    hard_negative_weight=(
                        args.hard_negative_weight
                    ),
                )

                loss = losses["total_loss"]

                loss_to_backward = (
                    loss / args.grad_accum
                )

            scaler.scale(
                loss_to_backward
            ).backward()

            if (
                (step + 1)
                % args.grad_accum
                == 0
                or (step + 1)
                == len(loader)
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad()

            if step % 20 == 0:

                elapsed = (
                    time.time() - t0
                )

                print(
                    f"epoch {epoch} "
                    f"step {step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"cir={losses['cir_loss'].item():.4f} "
                    f"hard={losses['hard_negative_loss'].item():.4f} "
                    f"elapsed={elapsed:.0f}s"
                )

                with open(
                    log_path,
                    "a",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    csv.writer(f).writerow(
                        [
                            epoch,
                            step,
                            loss.item(),
                            losses[
                                "cir_loss"
                            ].item(),
                            losses[
                                "hard_negative_loss"
                            ].item(),
                            elapsed,
                        ]
                    )

        checkpoint = {
            "epoch": epoch,
            "trainable_state": {
                "cross_attn":
                    model.cross_attn.state_dict(),
                "fusion":
                    model.fusion.state_dict(),
                "attribute_heads":
                    model.attribute_heads.state_dict(),
            },
            "optimizer_state":
                optimizer.state_dict(),
            "config":
                vars(args),
        }

        checkpoint_path = (
            checkpoint_dir
            / f"epoch_{epoch}.pt"
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Saved checkpoint: "
            f"{checkpoint_path}"
        )

    print(
        f"Training complete in "
        f"{time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()