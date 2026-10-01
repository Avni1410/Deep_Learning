from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from src.data.fashioniq_dataset import FashionIQCIRDataset
from src.models.attribute_aware_cir import AttributeAwareCIRModel
from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig
from src.retrieval.cosine_retrieval import retrieve_top_k


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "experiments"
    / "phase4"
    / "phase4_deliverable"
    / "checkpoints"
    / "epoch_2.pt"
)

GALLERY_DIR = (
    ROOT
    / "data"
    / "processed"
    / "fashioniq_embeddings"
)

IMAGE_ROOT = ROOT / "data" / "raw" / "fashioniq"


def load_model(device: str):
    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    clip = CLIPEncoder(
        CLIPEncoderConfig(
            model_name="openai/clip-vit-base-patch32",
            device=device,
        )
    )

    model = AttributeAwareCIRModel(
        clip,
        attribute_vocab=ckpt.get("attribute_vocab"),
    ).to(clip.device)

    model.cross_attn.load_state_dict(
        ckpt["trainable_state"]["cross_attn"]
    )

    model.fusion.load_state_dict(
        ckpt["trainable_state"]["fusion"]
    )

    model.attribute_heads.load_state_dict(
        ckpt["trainable_state"]["attribute_heads"]
    )

    model.eval()

    return model


def find_image(image_id: str) -> Path | None:
    candidates = list(
        IMAGE_ROOT.rglob(f"{image_id}.jpg")
    )

    if candidates:
        return candidates[0]

    candidates = list(
        IMAGE_ROOT.rglob(f"{image_id}.png")
    )

    if candidates:
        return candidates[0]

    return None


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        choices=["dress", "shirt", "toptee"],
        default="dress",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    args = parser.parse_args()

    print("Loading model...")

    model = load_model(args.device)

    # ---------------------------------------------------------
    # Load gallery
    # ---------------------------------------------------------

    gallery_path = (
        GALLERY_DIR
        / f"{args.category}_val_gallery.pt"
    )

    gallery = torch.load(
        gallery_path,
        map_location="cpu",
        weights_only=False,
    )

    gallery_ids = gallery["ids"]
    gallery_embeddings = gallery["embeddings"]

    print(
        f"Gallery size: {len(gallery_ids)}"
    )

    # ---------------------------------------------------------
    # Load FashionIQ validation queries
    # ---------------------------------------------------------

    csv_path = (
        ROOT
        / "data"
        / "processed"
        / "fashioniq"
        / f"val_queries_{args.category}.csv"
       
    )

    dataset = FashionIQCIRDataset(
        csv_path=csv_path,
        image_root=IMAGE_ROOT,
    )

    print(
        f"Validation queries: {len(dataset)}"
    )

    # ---------------------------------------------------------
    # Select query
    # ---------------------------------------------------------

    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(
            f"Query index {args.index} is outside "
            f"the valid range 0-{len(dataset)-1}"
        )

    sample = None
    actual_index = args.index

    for idx in range(args.index, len(dataset)):
        try:
            sample = dataset[idx]
            actual_index = idx
            break
        except FileNotFoundError:
            continue

    if sample is None:
        raise RuntimeError(
            f"No valid query found from index {args.index} onward."
        )

    if actual_index != args.index:
        print(
            f"Requested index {args.index} has a missing image. "
            f"Using valid query index {actual_index} instead."
        )

    print()
    print("=" * 70)
    print("PHASE 7 QUALITATIVE RETRIEVAL")
    print("=" * 70)

    print(
        f"Category : {args.category}"
    )

    print(
        f"Query    : {actual_index}"
    )

    print(
        f"Target   : {sample['target']}"
    )

    print(
        f"Caption 1: {sample['caption_1']}"
    )

    print(
        f"Caption 2: {sample['caption_2']}"
    )

    # ---------------------------------------------------------
    # Encode composed query
    # ---------------------------------------------------------

    with torch.no_grad():

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

    # ---------------------------------------------------------
    # Retrieve top-K
    # ---------------------------------------------------------

    retrieval = retrieve_top_k(
        query_embedding,
        gallery_embeddings,
        gallery_ids,
        k=args.top_k,
    )

    ranked_ids = retrieval.ranked_ids

    target = sample["target"]

    print()
    print("Retrieved:")

    for rank, image_id in enumerate(
        ranked_ids,
        start=1,
    ):

        marker = (
            " <-- TARGET"
            if image_id == target
            else ""
        )

        print(
            f"{rank}. {image_id}{marker}"
        )

    # ---------------------------------------------------------
    # Create visualization
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(18, 9)
    )

    # ---------------------------------------------------------
    # Reference image
    # ---------------------------------------------------------

    ax = plt.subplot(
        2,
        3,
        1,
    )

    reference = sample["reference_image"]

    if isinstance(reference, Image.Image):

        ax.imshow(reference)

    else:

        ax.imshow(
            Image.open(reference).convert("RGB")
        )

    ax.set_title(
        "Reference Image"
    )

    ax.axis("off")

    # ---------------------------------------------------------
    # Modification text
    # ---------------------------------------------------------

    modification = (
        f"{sample['caption_1']} "
        f"{sample['caption_2']}"
    )

    fig.text(
        0.5,
        0.02,
        "Modification: " + modification,
        ha="center",
        fontsize=12,
        wrap=True,
    )

    # ---------------------------------------------------------
    # Retrieved images
    # ---------------------------------------------------------

    for i, image_id in enumerate(
        ranked_ids
    ):

        ax = plt.subplot(
            2,
            3,
            i + 2,
        )

        image_path = find_image(
            image_id
        )

        if image_path is not None:

            image = Image.open(
                image_path
            ).convert("RGB")

            ax.imshow(image)

        else:

            ax.text(
                0.5,
                0.5,
                "Image not found",
                ha="center",
                va="center",
            )

        title = (
            f"Rank {i + 1}\n"
            f"{image_id}"
        )

        if image_id == target:

            title += "\nTARGET"

        ax.set_title(title)

        ax.axis("off")

    # ---------------------------------------------------------
    # Figure title
    # ---------------------------------------------------------

    plt.suptitle(
        "Attribute-Aware Composed Image Retrieval"
        f" — {args.category}",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[0, 0.05, 1, 0.95]
    )

    # ---------------------------------------------------------
    # Save visualization
    # ---------------------------------------------------------

    output_dir = (
        ROOT
        / "experiments"
        / "phase7"
        / "qualitative"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{args.category}_query_{actual_index}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
    )

    plt.show()

    print()
    print(
        "Saved visualization to:"
    )

    print(output_path)


if __name__ == "__main__":
    main()