import torch
import torch.nn.functional as F

from src.losses.fine_grained import (
    fine_grained_hard_negative_loss,
    combined_phase6_loss,
)


def compute_phase6_loss(
    query_embeddings,
    positive_embeddings,
    negative_embeddings,
    cir_loss,
    margin=0.10,
    hard_negative_weight=0.5,
):
    """
    Compute the complete Phase 6 objective.

    Args:
        query_embeddings:     [B, D]
        positive_embeddings:  [B, D]
        negative_embeddings:  [B, K, D]
        cir_loss:             scalar Phase 4 CIR InfoNCE loss
        margin:               hard-negative ranking margin
        hard_negative_weight: lambda for hard-negative loss

    Returns:
        Dictionary containing all Phase 6 loss components.
    """

    hard_negative_loss = fine_grained_hard_negative_loss(
        query_embeddings=query_embeddings,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
        margin=margin,
    )

    total_loss = combined_phase6_loss(
        cir_loss=cir_loss,
        hard_negative_loss=hard_negative_loss,
        hard_negative_weight=hard_negative_weight,
    )

    return {
        "cir_loss": cir_loss,
        "hard_negative_loss": hard_negative_loss,
        "total_loss": total_loss,
    }
