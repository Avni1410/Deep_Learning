import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image

from src.models.clip_encoder import CLIPEncoder, CLIPEncoderConfig
from src.models.composed_retrieval_model import (
    CrossDomainCIRModel,
    load_phase4_checkpoint,
)
from src.losses.phase6_loss import compute_phase6_loss
from src.data.phase6_hard_negative_dataset import (
    FashionIQHardNegativeDataset,
    load_quality_exclusions,
)
from src.models.composed_retrieval_model import AttributeAwareCIRModel


ROOT = Path.cwd()
DEVICE = torch.device("cpu")
CATEGORY = "dress"

CSV_PATH = (
    ROOT
    / "experiments"
    / "phase6"
    / "hard_negatives"
    / f"{CATEGORY}_train_similarity_negatives.csv"
)

IMAGE_ROOT = ROOT / "data" / "raw" / "fashioniq"

EXCLUSION_FILE = (
    ROOT
    / "experiments"
    / "phase6"
    / "quality_exclusions.txt"
)

CHECKPOINT = (
    ROOT
    / "experiments"
    / "phase4"
    / "phase4_deliverable"
    / "checkpoints"
    / "epoch_2.pt"
)


print("=" * 70)
print("PHASE 6 REAL TRAINING SMOKE TEST")
print("=" * 70)


# ---------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------

exclusions = load_quality_exclusions(EXCLUSION_FILE)

dataset = FashionIQHardNegativeDataset(
    csv_path=CSV_PATH,
    image_root=IMAGE_ROOT,
    quality_exclusions=exclusions,
)

item = dataset[0]

print("Dataset size:", len(dataset))
print("Query:", item["query_id"])
print("Target:", item["target"])
print("Candidate:", item["candidate"])
print("Negatives:", item["negatives"])


# ---------------------------------------------------------
# 2. Load CLIP
# ---------------------------------------------------------

print()
print("Loading CLIP encoder...")

clip_config = CLIPEncoderConfig(
    model_name="openai/clip-vit-base-patch16",
    device="cpu",
    dtype="float32",
    freeze=True,
)

clip_encoder = CLIPEncoder(
    config=clip_config
).to(DEVICE)

print("CLIP loaded.")


# ---------------------------------------------------------
# 3. Load Phase 4 model
# ---------------------------------------------------------

print()
print("Loading Phase 4 checkpoint...")

phase4_model = AttributeAwareCIRModel(
    clip_encoder=clip_encoder
).to(DEVICE)

epoch, metadata = load_phase4_checkpoint(
    phase4_model,
    CHECKPOINT,
    DEVICE,
)

print("Phase 4 checkpoint loaded.")
print("Checkpoint epoch:", epoch)


# ---------------------------------------------------------
# 4. Build CrossDomainCIRModel
# ---------------------------------------------------------

model = CrossDomainCIRModel(
    phase4_model=phase4_model
).to(DEVICE)

model.eval()

print("CrossDomainCIRModel created.")


# ---------------------------------------------------------
# 5. Load actual images
# ---------------------------------------------------------

def load_image(path):
    return Image.open(path).convert("RGB")


reference_image = load_image(
    item["candidate_path"]
)

target_image = load_image(
    item["target_path"]
)

negative_images = [
    load_image(path)
    for path in item["negative_paths"]
]

print()
print("Reference image loaded.")
print("Target image loaded.")
print("3 hard-negative images loaded.")


# ---------------------------------------------------------
# 6. Test image encoding
# ---------------------------------------------------------

print()
print("Testing image encoding...")

with torch.no_grad():

    target_embedding = model.encode_target(
        [target_image]
    )

    negative_embeddings = torch.stack([
        model.encode_target([img])[0]
        for img in negative_images
    ])

print(
    "Target embedding:",
    tuple(target_embedding.shape)
)

print(
    "Negative embeddings:",
    tuple(negative_embeddings.shape)
)


# ---------------------------------------------------------
# 7. Test composed query
# ---------------------------------------------------------

print()
print("Testing composed-query encoding...")

caption_1 = "fashion modification"
caption_2 = "fashion modification"

with torch.no_grad():

    query_output = model.encode_query(
        reference_images=[reference_image],
        caption_1=[caption_1],
        caption_2=[caption_2],
    )

print(
    "Query output type:",
    type(query_output).__name__
)


# ---------------------------------------------------------
# 8. Extract query embedding
# ---------------------------------------------------------

if isinstance(query_output, dict):

    print(
        "Query output keys:",
        list(query_output.keys())
    )

    possible_keys = [
        "query_embedding",
        "embedding",
        "query",
        "composed_embedding",
    ]

    query_embedding = None

    for key in possible_keys:

        if key in query_output:

            query_embedding = query_output[key]

            print(
                "Using query key:",
                key
            )

            break

    if query_embedding is None:

        raise RuntimeError(
            "Could not find query embedding in "
            "encode_query output."
        )

else:

    query_embedding = query_output


# ---------------------------------------------------------
# 9. Normalize dimensions
# ---------------------------------------------------------

if query_embedding.ndim == 1:
    query_embedding = query_embedding.unsqueeze(0)

if target_embedding.ndim == 1:
    target_embedding = target_embedding.unsqueeze(0)

if negative_embeddings.ndim == 2:
    negative_embeddings = negative_embeddings.unsqueeze(0)


print()
print(
    "Query embedding:",
    tuple(query_embedding.shape)
)

print(
    "Target embedding:",
    tuple(target_embedding.shape)
)

print(
    "Negative embeddings:",
    tuple(negative_embeddings.shape)
)


# ---------------------------------------------------------
# 10. Phase 6 loss
# ---------------------------------------------------------

cir_loss = torch.tensor(
    0.5,
    requires_grad=True,
    device=DEVICE,
)

losses = compute_phase6_loss(
    query_embeddings=query_embedding,
    positive_embeddings=target_embedding,
    negative_embeddings=negative_embeddings,
    cir_loss=cir_loss,
    margin=0.10,
    hard_negative_weight=0.5,
)


print()
print("CIR loss:", losses["cir_loss"].item())

print(
    "Hard-negative loss:",
    losses["hard_negative_loss"].item(),
)

print(
    "Total loss:",
    losses["total_loss"].item(),
)

print()
print(
    "Finite:",
    torch.isfinite(
        losses["total_loss"]
    ).item()
)

print()
print("=" * 70)
print("SMOKE TEST RESULT")
print("=" * 70)

print(
    "PASS:",
    torch.isfinite(
        losses["total_loss"]
    ).item()
)
