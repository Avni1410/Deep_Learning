"""
Phase 7 - End-to-end composed image retrieval evaluation.

Pipeline:
reference image + modification captions
        ↓
Phase 4 Attribute-Aware CIR model
        ↓
query embedding
        ↓
precomputed FashionIQ gallery
        ↓
cosine similarity retrieval
        ↓
Recall@1 / Recall@5 / Recall@10 / Recall@50

Phase 4 is intentionally used as the frozen retrieval model because
it is the strongest validated checkpoint available.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data.fashioniq_dataset import FashionIQCIRDataset
from src.models.attribute_aware_cir import AttributeAwareCIRModel
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig
from src.retrieval.cosine_retrieval import rank_of_target, retrieve_top_k


def load_phase4_model(
    checkpoint_path: Path,
    model_name: str,
    device: str,
) -> AttributeAwareCIRModel:

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    clip = CLIPEncoder(
        CLIPEncoderConfig(
            model_name=model_name,
            device=device,
        )
    )

    model = AttributeAwareCIRModel(
        clip,
        attribute_vocab=checkpoint.get("attribute_vocab"),
    ).to(clip.device)

    model.cross_attn.load_state_dict(
        checkpoint["trainable_state"]["cross_attn"]
    )

    model.fusion.load_state_dict(
        checkpoint["trainable_state"]["fusion"]
    )

    model.attribute_heads.load_state_dict(
        checkpoint["trainable_state"]["attribute_heads"]
    )

    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Device: {clip.device}")

    return model


def recall_at_k(ranks: list[int | None], k: int) -> float:
    valid = [r for r in ranks if r is not None]
    if not valid:
        return 0.0

    return sum(r <= k for r in valid) / len(ranks)


def evaluate_category(
    model: AttributeAwareCIRModel,
    csv_path: Path,
    gallery_path: Path,
    fiq_root: Path,
    category: str,
    split: str,
    limit: int | None,
    top_k: int,
):
    print()
    print("=" * 70)
    print(f"Category: {category} | Split: {split}")
    print("=" * 70)

    gallery = torch.load(
        gallery_path,
        map_location="cpu",
        weights_only=False,
    )

    gallery_ids = gallery["ids"]
    gallery_embeddings = gallery["embeddings"]

    print(f"Gallery size: {len(gallery_ids)}")
    print(f"Embedding shape: {tuple(gallery_embeddings.shape)}")

    dataset = FashionIQCIRDataset(
        csv_path=csv_path,
        image_root=fiq_root,
    )

    n = len(dataset) if limit is None else min(limit, len(dataset))

    print(f"Queries: {n}/{len(dataset)}")

    results = []
    ranks = []

    start_time = time.time()

    with torch.no_grad():

        for i in tqdm(
            range(n),
            desc=f"Retrieving {category}",
        ):

            sample = dataset[i]

            output = model.encode_query(
                [sample["reference_image"]],
                [sample["caption_1"]],
                [sample["caption_2"]],
            )

            query_embedding = (
                output["query_embedding"]
                .squeeze(0)
                .cpu()
            )

            retrieval = retrieve_top_k(
                query_embedding=query_embedding,
                gallery_embeddings=gallery_embeddings,
                gallery_ids=gallery_ids,
                k=None,
            )

            rank = rank_of_target(
                retrieval.ranked_ids,
                {sample["target"]},
            )

            ranks.append(rank)

            results.append(
                {
                    "query_id": sample["query_id"],
                    "category": sample["category"],
                    "reference": str(sample["reference_image"]),
                    "target": sample["target"],
                    "caption_1": sample["caption_1"],
                    "caption_2": sample["caption_2"],
                    "target_rank": rank,
                    "top_k_ids": retrieval.ranked_ids[:top_k],
                    "top_k_scores": [
                        float(x)
                        for x in retrieval.scores[:top_k].tolist()
                    ],
                }
            )

    elapsed = time.time() - start_time

    metrics = {
        "R@1": recall_at_k(ranks, 1),
        "R@5": recall_at_k(ranks, 5),
        "R@10": recall_at_k(ranks, 10),
        "R@50": recall_at_k(ranks, 50),
    }

    print()
    print(f"Completed in {elapsed:.1f}s")
    print(f"R@1  = {metrics['R@1']:.4f}")
    print(f"R@5  = {metrics['R@5']:.4f}")
    print(f"R@10 = {metrics['R@10']:.4f}")
    print(f"R@50 = {metrics['R@50']:.4f}")

    return {
        "category": category,
        "split": split,
        "gallery_size": len(gallery_ids),
        "num_queries": n,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "queries": results,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            REPO_ROOT
            / "experiments"
            / "phase4"
            / "phase4_deliverable"
            / "checkpoints"
            / "epoch_2.pt"
        ),
    )

    parser.add_argument(
        "--fiq-root",
        type=Path,
        default=REPO_ROOT / "data/raw/fashioniq",
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=REPO_ROOT / "data/processed/fashioniq",
    )

    parser.add_argument(
        "--gallery-root",
        type=Path,
        default=REPO_ROOT / "data/processed/fashioniq_embeddings",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "experiments/phase7",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch16",
    )

    args = parser.parse_args()

    categories = [
        "dress",
        "shirt",
        "toptee",
    ]

    model = load_phase4_model(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        device=args.device,
    )

    all_results = {}

    for category in categories:

        csv_path = (
            args.data_root
            / f"val_queries_{category}.csv"
        )

        gallery_path = (
            args.gallery_root
            / f"{category}_val_gallery.pt"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing query CSV: {csv_path}"
            )

        if not gallery_path.exists():
            raise FileNotFoundError(
                f"Missing gallery: {gallery_path}"
            )

        result = evaluate_category(
            model=model,
            csv_path=csv_path,
            gallery_path=gallery_path,
            fiq_root=args.fiq_root,
            category=category,
            split="val",
            limit=args.limit,
            top_k=args.top_k,
        )

        all_results[category] = result

    output = {
        "phase": 7,
        "model": args.model,
        "checkpoint": str(args.checkpoint),
        "protocol": (
            "Phase 4 attribute-aware composed query "
            "embedding against precomputed FashionIQ "
            "target gallery using cosine similarity"
        ),
        "categories": all_results,
    }

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.out_dir
        / "phase7_retrieval_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print("=" * 70)
    print("PHASE 7 COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()