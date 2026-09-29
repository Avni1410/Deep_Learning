"""
Phase 4 - masked attribute classification loss.

Labels use `ignore_index` for missing annotations; a batch where an attribute
has no valid labels at all contributes nothing for that attribute (returns
None), rather than an artificial zero-labeled class.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_attribute_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor | None:
    valid = labels != ignore_index
    if valid.sum() == 0:
        return None
    return F.cross_entropy(logits[valid], labels[valid])


def combined_attribute_loss(
    attribute_logits: dict[str, torch.Tensor],
    attribute_labels: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
    weights = weights or {}
    per_attr: dict[str, torch.Tensor] = {}
    total: torch.Tensor | None = None
    for name, logits in attribute_logits.items():
        if name not in attribute_labels:
            continue
        loss = masked_attribute_loss(logits, attribute_labels[name], ignore_index)
        if loss is None:
            continue
        per_attr[name] = loss
        w = weights.get(name, 1.0)
        total = w * loss if total is None else total + w * loss
    return total, per_attr