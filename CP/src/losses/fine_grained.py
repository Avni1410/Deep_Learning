import torch
import torch.nn.functional as F


def fine_grained_hard_negative_loss(
    query_embeddings,
    positive_embeddings,
    negative_embeddings,
    margin=0.10,
):
    """
    Fine-grained hard-negative ranking loss.

    query_embeddings:
        [B, D]

    positive_embeddings:
        [B, D]

    negative_embeddings:
        [B, K, D]

    The loss encourages:

        sim(query, positive) > sim(query, negative) + margin

    for every hard negative.
    """

    query_embeddings = F.normalize(query_embeddings, dim=-1)
    positive_embeddings = F.normalize(positive_embeddings, dim=-1)
    negative_embeddings = F.normalize(negative_embeddings, dim=-1)

    # Positive similarity: [B]
    positive_similarity = (
        query_embeddings * positive_embeddings
    ).sum(dim=-1)

    # Negative similarities: [B, K]
    negative_similarity = torch.einsum(
        "bd,bkd->bk",
        query_embeddings,
        negative_embeddings,
    )

    # Hinge ranking loss:
    #
    # max(0, margin - positive_similarity + negative_similarity)
    #
    # We want:
    # positive_similarity >= negative_similarity + margin
    loss = F.relu(
        margin
        - positive_similarity.unsqueeze(1)
        + negative_similarity
    )

    return loss.mean()


def combined_phase6_loss(
    cir_loss,
    hard_negative_loss,
    hard_negative_weight=0.5,
):
    """
    Phase 6 total objective:

        L_total = L_CIR + lambda_HN * L_HN
    """

    return (
        cir_loss
        + hard_negative_weight * hard_negative_loss
    )
